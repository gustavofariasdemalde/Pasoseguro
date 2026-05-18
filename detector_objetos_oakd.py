#!/usr/bin/env python3
"""
Programa de detección de objetos con YOLO y cámara OAK-D Lite
Detecta objetos y calcula distancias usando la información de profundidad
"""

import os
import sys

# Si no hay DISPLAY, ejecutar con: ./ejecutar.sh (él arranca Xvfb y pone DISPLAY=:99)
import cv2
import numpy as np
import depthai as dai
from ultralytics import YOLO
import time
# pyttsx3 deshabilitado: su carga provoca Abortado (SIGABRT) en Jetson
import threading
import speech_recognition as sr
import queue
import re
import json
import os
import sys
import signal
import serial
import serial.tools.list_ports
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# Suprimir mensajes de ALSA (audio) desde el inicio usando variables de entorno
if sys.platform == 'linux':
    os.environ['PYTHONWARNINGS'] = 'ignore'
    # Suprimir mensajes de ALSA usando variables de entorno
    os.environ['ALSA_CARD'] = '0'
    
    import warnings
    warnings.filterwarnings('ignore')
    
    # Redirigir stderr para suprimir mensajes de ALSA de forma más agresiva
    class SuppressALSA:
        def __init__(self):
            self.original_stderr = sys.stderr
            try:
                self.devnull = open(os.devnull, 'w')
            except:
                self.devnull = None
            
        def write(self, message):
            if message:
                msg_str = str(message)
                # Filtrar todos los mensajes de ALSA y audio
                if any(x in msg_str for x in ['ALSA', 'pcm', 'snd_', 'Unknown PCM', 'Unable to find']):
                    if self.devnull:
                        self.devnull.write(msg_str)
                    return
            self.original_stderr.write(message)
                
        def flush(self):
            self.original_stderr.flush()
            if self.devnull:
                self.devnull.flush()
    
    # Aplicar supresión siempre
    sys.stderr = SuppressALSA()

class OAKDObjectDetector:
    def __init__(self, model_name='yolov8n.pt'):
        """
        Inicializa el detector de objetos con OAK-D Lite
        
        Args:
            model_name: Nombre del modelo YOLO (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
        """
        print("Inicializando YOLO...")
        self.model = YOLO(model_name)
        print(f"Modelo YOLO {model_name} cargado exitosamente")
        
        print("Inicializando cámara OAK-D Lite...")
        self.pipeline = self.create_pipeline()
        try:
            self.device = dai.Device(self.pipeline)
            print("Cámara OAK-D Lite conectada")
        except RuntimeError as e:
            if "ALREADY_IN_USE" in str(e) or "X_LINK" in str(e):
                print("\n" + "="*60)
                print("❌ ERROR: La cámara OAK-D Lite está en uso")
                print("="*60)
                print("Solución:")
                print("  1. Detén el programa anterior ejecutando:")
                print("     pkill -9 -f detector_objetos_oakd.py")
                print("  2. Espera 3 segundos y vuelve a ejecutar el programa")
                print("  3. O usa el script: ./detener.sh")
                print("="*60)
            raise
        
        # Colas pequeñas + no bloqueo: menos frames encolados = imagen más “actual”
        # (con maxSize alto, tryGet devuelve el frame más viejo y suma latencia).
        self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        self.q_depth = self.device.getOutputQueue(name="depth", maxSize=1, blocking=False)
        
        # Flag para controlar el loop
        self.running = True
        
        # Variables para detección de desniveles
        self.ultimo_desnivel_anunciado = None
        self.frame_desnivel_anterior = 0
        
        # Configurar manejo de señales para detener correctamente
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Obtener nombres de clases de YOLO
        self.class_names = self.model.names
        
        # Inicializar reconocimiento de voz (solo para cuando pregunte "¿Deseas preguntar algo?")
        self.recognizer = None
        self.microphone = None
        print("Inicializando reconocimiento de voz...")
        old_stderr = sys.stderr
        try:
            sys.stderr = open(os.devnull, 'w')
            try:
                self.recognizer = sr.Recognizer()
                mic_list = sr.Microphone.list_microphone_names()
                h390_index = None
                for i, mic_name in enumerate(mic_list):
                    if 'logi' in mic_name.lower() or 'h390' in mic_name.lower() or 'usb headset' in mic_name.lower():
                        h390_index = i
                        break
                self.microphone = sr.Microphone(device_index=h390_index if h390_index is not None else 0)
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=2)
                self.recognizer.energy_threshold = max(self.recognizer.energy_threshold * 0.6, 150)
                print("  ✓ Micrófono listo (se activa cuando pregunte '¿Deseas preguntar algo?')")
            except Exception as e:
                print(f"  ⚠ Micrófono no disponible: {e}")
                self.recognizer = None
                self.microphone = None
        finally:
            try:
                if sys.stderr is not old_stderr:
                    sys.stderr.close()
            except Exception:
                pass
            sys.stderr = old_stderr
        
        # Cola para comandos de voz
        self.voice_commands = queue.Queue()
        
        # Bandera para indicar cuando el usuario está hablando (para pausar síntesis de voz)
        self.user_speaking = False
        self.user_speaking_lock = threading.Lock()
        
        # Objetos detectados actualmente (para responder preguntas)
        self.current_objects = {}
        self.objects_lock = threading.Lock()

        # OCR automático de carteles (sin que el usuario diga nada)
        self.ocr_reader = None
        self.ocr_last_announce_time = 0.0
        self.ocr_last_text_key = ""
        self.ocr_check_every_n_frames = 30  # Reduce carga: OCR no se hace en cada frame
        self.ocr_cooldown_seconds = 30       # Evita repetir el mismo cartel seguido
        # Umbral más bajo: EasyOCR suele dar scores bajos en letras pequeñas / carteles
        self.ocr_min_confidence = 0.28
        self.ocr_max_chars = 120
        self.ocr_max_words = 18
        self.last_listen_end_time = 0.0  # Evita que el OCR hable justo después de escuchar al usuario
        self.ocr_in_progress = False
        self.ocr_thread_lock = threading.Lock()
        self.ocr_reader_lock = threading.Lock()
        self._ocr_voice_cancelled = False
        # Último frame útil para OCR cuando el usuario lo pide por voz
        self.last_frame_for_ocr = None
        self.last_frame_count_for_ocr = 0
        
        # Configurar auricular H390 como dispositivo de salida por defecto
        print("Configurando auricular H390 como dispositivo de audio...")
        try:
            import subprocess
            # Buscar el auricular H390
            result = subprocess.run(['pactl', 'list', 'short', 'sinks'], 
                                  capture_output=True, text=True, timeout=2)
            h390_sink = None
            for line in result.stdout.split('\n'):
                if 'logi' in line.lower() or 'h390' in line.lower():
                    h390_sink = line.split()[1]
                    break
            
            if h390_sink:
                # Configurar como dispositivo por defecto
                subprocess.run(['pactl', 'set-default-sink', h390_sink], 
                             capture_output=True, timeout=2)
                subprocess.run(['pactl', 'set-sink-volume', h390_sink, '100%'], 
                             capture_output=True, timeout=2)
                subprocess.run(['pactl', 'set-sink-mute', h390_sink, '0'], 
                             capture_output=True, timeout=2)
                print(f"  ✓ Auricular H390 configurado: {h390_sink}")
            else:
                print("  ⚠ Auricular H390 no encontrado, usando dispositivo por defecto")
        except Exception as e:
            print(f"  ⚠ No se pudo configurar auricular automáticamente: {e}")
            print("  Ejecuta: ./configurar_auricular.sh")
        
        # Síntesis de voz: en Jetson pyttsx3/espeak provoca Abortado (SIGABRT).
        # Deshabilitada aquí; puedes usar espeak por terminal si quieres voz:
        #   espeak -v es "Texto a decir"
        print("Inicializando síntesis de voz...")
        self.tts_engine = None  # Deshabilitado para evitar abort en Jetson
        self.tts_lock = threading.Lock()
        self.last_spoken = {}
        self.last_spoken_time = {}
        print("  ✓ Síntesis deshabilitada (evita abort). El programa mostrará detección por pantalla.")
        
        # Inicializar GPS
        print("Inicializando GPS...")
        self.gps_serial = None
        self.gps_location = None
        self.gps_lock = threading.Lock()
        self.gps_thread = None
        self.gps_probe_baud = None  # baud detectado al sondear NMEA (find_gps_port)
        self._gps_last_lazy_attempt = 0.0  # throttle para reintento al pedir ubicación
        self._gps_last_console_log = 0.0
        self._gps_last_logged_latlon = None
        
        # Inicializar geocodificador para obtener direcciones
        try:
            self.geolocator = Nominatim(user_agent="oakd_detector")
            print("  ✓ Geocodificador inicializado (para obtener direcciones)")
        except Exception as e:
            print(f"  ⚠ Error inicializando geocodificador: {e}")
            self.geolocator = None

        # Guía simple a destino (geocodificar + distancia / orientación por voz)
        self.nav_destination = None
        self.nav_lock = threading.Lock()
        self.home_location = self._load_home_location()
        print(
            f"  ✓ Casa guardada: {self.home_location['label']} "
            f"({self.home_location['latitude']:.6f}, {self.home_location['longitude']:.6f})"
        )
        
        try:
            # Intentar activar GPS automáticamente si no se encuentra
            gps_port = self.find_gps_port()
            if not gps_port:
                print("  Intentando activar GPS automáticamente...")
                try:
                    import subprocess
                    # Intentar usar el comando activar_gps (sin sudo después de configuración)
                    result = subprocess.run(['activar_gps'], 
                                          capture_output=True, text=True, timeout=8)
                    if result.returncode == 0:
                        print("  ✓ GPS activado automáticamente")
                        time.sleep(3)  # Esperar más tiempo a que aparezca el puerto
                        gps_port = self.find_gps_port()
                    else:
                        # Si falla, intentar con el script local
                        result = subprocess.run(['./activar_gps_simple.sh'], 
                                              capture_output=True, text=True, timeout=8)
                        if result.returncode == 0:
                            print("  ✓ GPS activado automáticamente")
                            time.sleep(3)
                            gps_port = self.find_gps_port()
                        else:
                            # Último intento: forzar activación con comandos directos
                            try:
                                subprocess.run(['modprobe', '-r', 'ch34x'], 
                                             capture_output=True, timeout=2)
                                time.sleep(1)
                                subprocess.run(['modprobe', 'ch34x'], 
                                             capture_output=True, timeout=2)
                                time.sleep(2)
                                # Intentar escribir en new_id (puede fallar sin sudo)
                                with open('/sys/bus/usb-serial/drivers/ch34x/new_id', 'w') as f:
                                    f.write('1a86 7523\n')
                                time.sleep(2)
                                gps_port = self.find_gps_port()
                            except:
                                pass
                except Exception as e:
                    print(f"  ⚠ No se pudo activar GPS automáticamente: {e}")
                    print("  Ejecuta manualmente: activar_gps")
                    print("  O si es la primera vez: sudo ./configurar_gps_permanente.sh")
            
            if gps_port:
                if self._open_gps_serial_and_thread(gps_port):
                    print("  ✓ GPS iniciado, leyendo datos...")
                    print("  ℹ El GPS necesita estar al aire libre para recibir señal de satélites")
                else:
                    self.gps_serial = None
                    print("  ⚠ No se pudo abrir el puerto GPS; se reintentará al preguntar dónde estás")
            else:
                print("  ⚠ GPS no encontrado, funcionalidad deshabilitada")
                print("  Conecta el módulo GPS por USB y ejecuta: ./activar_gps.sh")
                print(
                    "  ℹ Modem Huawei + GPS: el primer ttyUSB suele ser el módem. "
                    "Forzá el GPS con: export GPS_SERIAL_PORT=/dev/ttyUSB1"
                )
        except Exception as e:
            print(f"  ⚠ Error inicializando GPS: {e}")
            import traceback
            traceback.print_exc()
            print("  La funcionalidad GPS estará deshabilitada")
            self.gps_serial = None

        # EasyOCR: primera carga puede tardar minutos y bloquea el hilo → precarga en background
        def _preload_ocr():
            try:
                os.environ.setdefault("OMP_NUM_THREADS", "2")
                os.environ.setdefault("MKL_NUM_THREADS", "2")
                print("🧾 OCR: precargando modelo (la primera vez descarga pesos)...", flush=True)
                self._ensure_ocr_reader()
                print("🧾 OCR: precarga lista.", flush=True)
            except Exception as ex:
                print(f"🧾 OCR: precarga omitida o falló: {ex}", flush=True)

        if os.environ.get("OCR_PRELOAD", "1").strip().lower() not in (
            "0",
            "no",
            "false",
            "off",
        ):
            threading.Thread(target=_preload_ocr, daemon=True).start()

    def _gps_modem_tty_set(self):
        """tty* que no debemos usar como GPS (Huawei E3372, Quectel, etc.)."""
        skip = set()
        extra = os.environ.get("GPS_SKIP_PORTS", "").strip()
        if extra:
            for p in extra.split(","):
                p = p.strip()
                if p:
                    skip.add(p)
        try:
            for p in serial.tools.list_ports.comports():
                hw = (p.hwid or "").upper().replace(" ", "")
                if "VID:PID=12D1:" in hw or "VID_12D1&PID_" in hw:
                    skip.add(p.device)
                for vid in ("1199", "2C7C", "1BC7", "05C6", "1E0E"):
                    if f"VID:PID={vid}:" in hw or f"VID_{vid}&" in hw:
                        skip.add(p.device)
        except Exception:
            pass
        return skip

    def _probe_gps_nmea_baud(self, port):
        """Devuelve baud si en el puerto aparece NMEA (GGA/RMC); si no, None."""
        bauds = [9600, 115200, 38400, 4800, 57600]
        env_b = os.environ.get("GPS_PROBE_BAUDS", "").strip()
        if env_b:
            try:
                bauds = [int(x.strip()) for x in env_b.split(",") if x.strip()]
            except ValueError:
                pass
        try:
            per = float(os.environ.get("GPS_PROBE_SEC_PER_BAUD", "1.4"))
        except ValueError:
            per = 1.4
        per = max(0.35, min(4.5, per))
        for baud in bauds:
            ser = None
            try:
                ser = serial.Serial(port, baudrate=baud, timeout=0.12)
                try:
                    ser.reset_input_buffer()
                except Exception:
                    pass
                t_end = time.time() + per
                buf = ""
                while time.time() < t_end:
                    n = ser.in_waiting
                    if n:
                        buf += ser.read(n).decode("utf-8", errors="ignore")
                        # Sin fix aún puede no haber GGA/RMC completo; basta con tramas GPS típicas
                        if ("$GP" in buf or "$GN" in buf) and any(
                            t in buf
                            for t in (
                                "GGA",
                                "RMC",
                                "GSA",
                                "GSV",
                                "VTG",
                                "TXT",
                            )
                        ):
                            ser.close()
                            return baud
                    else:
                        time.sleep(0.02)
                ser.close()
            except Exception:
                try:
                    if ser is not None and getattr(ser, "is_open", False):
                        ser.close()
                except Exception:
                    pass
        return None

    def find_gps_port(self):
        """
        Encuentra el GPS por texto NMEA real. El Huawei E3372 abre como ttyUSB*
        pero no manda GGA/RMC; antes el primer ttyUSB ganaba y el GPS quedaba mal.
        """
        print("  Buscando puerto GPS (se ignoran modems LTE conocidos y se exige NMEA)...")
        import glob

        self.gps_probe_baud = None
        modem_skip = self._gps_modem_tty_set()
        if modem_skip:
            print(f"  Puertos no-GPS (modem / GPS_SKIP_PORTS): {sorted(modem_skip)}")

        forced = os.environ.get("GPS_SERIAL_PORT", "").strip()
        trust = os.environ.get("GPS_TRUST_PORT", "").strip().lower() in ("1", "yes", "true")

        if forced:
            if not os.path.exists(forced):
                print(f"  ✗ GPS_SERIAL_PORT={forced} no existe")
                return None
            if not os.access(forced, os.R_OK | os.W_OK):
                print(f"  ✗ Sin permisos en {forced} (sudo chmod 666 {forced})")
                return None
            if trust:
                print(f"  ✓ GPS_SERIAL_PORT={forced} (GPS_TRUST_PORT=1, sin sondeo NMEA)")
                return forced
            baud = self._probe_gps_nmea_baud(forced)
            if baud is not None:
                self.gps_probe_baud = baud
                print(f"  ✓ {forced} OK — NMEA a {baud} baud")
                return forced
            print(f"  ✗ {forced} no envió NMEA; revisá baud o cable")
            return None

        preferred = []
        try:
            for p in serial.tools.list_ports.comports():
                if p.device in modem_skip:
                    continue
                d = (p.description or "").lower()
                if any(
                    k in d
                    for k in (
                        "ch340",
                        "ch341",
                        "ch34",
                        "gps",
                        "ublox",
                        "serial adapter",
                        "cp210",
                        "ftdi",
                    )
                ):
                    if p.device not in preferred:
                        preferred.append(p.device)
        except Exception:
            pass

        candidates = []
        seen = set()
        for dev in preferred:
            if dev not in seen and os.path.exists(dev) and dev not in modem_skip:
                candidates.append(dev)
                seen.add(dev)
        for dev in sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")):
            if dev in modem_skip or dev in seen:
                continue
            candidates.append(dev)
            seen.add(dev)

        for intento in range(2):
            for port in candidates:
                try:
                    if not os.access(port, os.R_OK | os.W_OK):
                        if intento == 0:
                            print(f"  ⚠ {port} sin permisos (chmod 666 o regla udev)")
                        continue
                except OSError:
                    continue
                baud = self._probe_gps_nmea_baud(port)
                if baud is not None:
                    self.gps_probe_baud = baud
                    print(f"  ✓ GPS en {port} ({baud} baud)")
                    return port
            if intento == 0:
                time.sleep(0.9)

        print("  ✗ Ningún puerto devolvió tramas NMEA reconocibles")
        all_tty = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
        cand = [x for x in all_tty if x not in modem_skip]
        print(f"  Puertos USB/ACM en el sistema: {all_tty or '(ninguno)'}")
        print(f"  Candidatos GPS (sin contar modem): {cand or '(ninguno — revisá cable o permisos)'}")
        if not all_tty:
            print("  ⚠ Sin /dev/ttyUSB* con CH340: si dmesg dice «brltty» al enchufar:")
            print("     brltty lo lanza udev (no siempre aparece en systemctl). Solución:")
            print("       sudo apt purge brltty brltty-x11  &&  sudo killall brltty 2>/dev/null")
            print("       sudo udevadm control --reload-rules && sudo udevadm trigger")
            print("     O en este repo: ./fix_gps_brltty.sh   — luego desenchufá y enchufá el GPS.")
            print("     Luego: ls -la /dev/ttyUSB0")
            print("     (Huawei HiLink suele no usar ttyUSB; el GPS usa el CH340.)")
        print("  Sugerencias:")
        print("    lsusb | grep -i serial;  ls -la /dev/ttyUSB* /dev/ttyACM* 2>/dev/null")
        print("    (Si existe: ls -l /dev/serial/by-id/)")
        print("    export GPS_SERIAL_PORT=/dev/ttyUSB1   # el del GPS, no el Huawei")
        print("    export GPS_TRUST_PORT=1 && reiniciar el programa")
        print("    permisos: sudo chmod 666 /dev/ttyUSB*   o regla udev")
        return None

    def _open_gps_serial_and_thread(self, gps_port):
        """Abre el serial del GPS y arranca el hilo de lectura (arranque o reintento)."""
        if not gps_port or self.gps_serial is not None:
            return self.gps_serial is not None
        baudrates = [4800, 9600, 38400, 115200]
        if self.gps_probe_baud:
            baudrates = [self.gps_probe_baud] + [
                b for b in baudrates if b != self.gps_probe_baud
            ]
        for baudrate in baudrates:
            try:
                self.gps_serial = serial.Serial(gps_port, baudrate=baudrate, timeout=1)
                print(f"  ✓ GPS conectado en: {gps_port} (baudrate: {baudrate})", flush=True)
                self.gps_thread = threading.Thread(target=self.read_gps_data, daemon=True)
                self.gps_thread.start()
                return True
            except serial.SerialException as e:
                self.gps_serial = None
                if baudrate == baudrates[-1]:
                    print(f"  ✗ No se pudo abrir {gps_port}: {e}", flush=True)
                continue
        return False

    def try_connect_gps_lazy(self, reason=""):
        """
        Si al iniciar no hubo GPS (USB lento, sin fix, sondeo corto), reintenta al pedir ubicación.
        No toca cámara ni OCR.
        Returns:
            1 = conectado (o ya estaba)
            2 = esperá (throttle; reintentá en unos segundos)
            0 = falló la detección o apertura
        """
        if self.gps_serial is not None:
            return 1
        try:
            gap = float(os.environ.get("GPS_LAZY_RETRY_SEC", "5"))
        except ValueError:
            gap = 5.0
        gap = max(2.0, min(120.0, gap))
        now = time.time()
        if now - self._gps_last_lazy_attempt < gap and self._gps_last_lazy_attempt > 0:
            return 2
        self._gps_last_lazy_attempt = now
        print(f"📍 GPS: reintento de conexión ({reason})...", flush=True)
        gps_port = self.find_gps_port()
        if not gps_port:
            forced = os.environ.get("GPS_SERIAL_PORT", "").strip()
            trust = os.environ.get("GPS_TRUST_PORT", "").strip().lower() in (
                "1",
                "yes",
                "true",
            )
            if (
                forced
                and trust
                and os.path.exists(forced)
                and os.access(forced, os.R_OK | os.W_OK)
            ):
                print(
                    f"📍 GPS: abriendo {forced} con GPS_TRUST_PORT (sin sondeo NMEA)...",
                    flush=True,
                )
                gps_port = forced
            else:
                print(
                    "📍 GPS: no apareció ningún puerto con tramas NMEA. "
                    "Configurá GPS_SERIAL_PORT y GPS_TRUST_PORT y reiniciá.",
                    flush=True,
                )
                return 0
        if self._open_gps_serial_and_thread(gps_port):
            print(
                "📍 GPS: conectado. Si acabás de salir a la calle, esperá unos segundos al fix.",
                flush=True,
            )
            return 1
        return 0

    def _parse_nmea_lat_lon_from_dm(self, lat_str, ns, lon_str, ew):
        """lat_str / lon_str en formato NMEA DDMM.MMMM y DDDMM.MMMM."""
        if not lat_str or len(lat_str) < 4:
            return None
        lat_deg = float(lat_str[:2])
        lat_min = float(lat_str[2:])
        latitude = lat_deg + lat_min / 60.0
        if ns == "S":
            latitude = -latitude
        elif ns != "N":
            return None
        if not lon_str or len(lon_str) < 5:
            return None
        lon_deg = float(lon_str[:3])
        lon_min = float(lon_str[3:])
        longitude = lon_deg + lon_min / 60.0
        if ew == "W":
            longitude = -longitude
        elif ew != "E":
            return None
        return latitude, longitude

    def parse_nmea(self, nmea_sentence):
        """GGA / RMC, talkers GP y GN (módulos multi-constelación)."""
        try:
            line = nmea_sentence.strip()
            if not line.startswith("$"):
                return None
            parts = line.split(",")
            if len(parts) < 6:
                return None
            head = parts[0].upper()
            if head.endswith("GGA"):
                if len(parts) < 10 or not parts[2] or not parts[4]:
                    return None
                coords = self._parse_nmea_lat_lon_from_dm(
                    parts[2], parts[3], parts[4], parts[5]
                )
                if coords is None:
                    return None
                latitude, longitude = coords
                quality = int(parts[6]) if parts[6] else 0
                if quality <= 0:
                    return None
                return {
                    "latitude": latitude,
                    "longitude": longitude,
                    "quality": quality,
                    "timestamp": time.time(),
                }
            if head.endswith("RMC"):
                if len(parts) < 7:
                    return None
                if parts[2] != "A":
                    return None
                coords = self._parse_nmea_lat_lon_from_dm(
                    parts[3], parts[4], parts[5], parts[6]
                )
                if coords is None:
                    return None
                latitude, longitude = coords
                fix = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "quality": 1,
                    "timestamp": time.time(),
                }
                if len(parts) > 8 and parts[8]:
                    try:
                        fix["course_deg"] = float(parts[8]) % 360.0
                    except ValueError:
                        pass
                return fix
        except Exception:
            pass
        return None

    def _merge_gps_fix(self, location):
        """Combina fixes GGA/RMC sin perder el rumbo (RMC) al actualizar posición."""
        with self.gps_lock:
            prev = dict(self.gps_location) if self.gps_location else {}
            merged = {**prev, **location}
            if "course_deg" not in location and prev.get("course_deg") is not None:
                merged["course_deg"] = prev["course_deg"]
            self.gps_location = merged

    def _log_gps_fix_console(self, snap):
        extra = ""
        if snap.get("course_deg") is not None:
            extra = f", rumbo {snap['course_deg']:.0f}°"
        print(
            f"📍 GPS: Lat {snap['latitude']:.6f}, Lon {snap['longitude']:.6f} "
            f"(calidad: {snap['quality']}){extra}",
            flush=True,
        )

    def _maybe_log_gps_fix(self, snap):
        """
        El GPS sigue leyendo en memoria siempre; solo limita mensajes en consola.
        GPS_VERBOSE=1 → cada trama. Por defecto: 1.ª vez y luego cada ~30 s o si te movés ~15 m.
        """
        verbose = os.environ.get("GPS_VERBOSE", "").strip().lower() in (
            "1",
            "yes",
            "true",
        )
        if verbose:
            self._log_gps_fix_console(snap)
            return

        now = time.time()
        if self._gps_last_logged_latlon is None:
            self._log_gps_fix_console(snap)
            self._gps_last_console_log = now
            self._gps_last_logged_latlon = (snap["latitude"], snap["longitude"])
            return

        try:
            interval = float(os.environ.get("GPS_LOG_INTERVAL_SEC", "30"))
        except ValueError:
            interval = 30.0
        interval = max(10.0, min(300.0, interval))

        moved_m = 0.0
        try:
            moved_m = geodesic(
                self._gps_last_logged_latlon,
                (snap["latitude"], snap["longitude"]),
            ).meters
        except Exception:
            pass

        if moved_m >= 15.0 or (now - self._gps_last_console_log) >= interval:
            self._log_gps_fix_console(snap)
            self._gps_last_console_log = now
            self._gps_last_logged_latlon = (snap["latitude"], snap["longitude"])
    
    def read_gps_data(self):
        """Lee datos GPS en un thread separado"""
        if not self.gps_serial:
            return
        
        buffer = ""
        nmea_count = 0
        last_status_print = time.time()
        last_data_time = time.time()
        no_data_warning = False
        
        print("  📡 GPS: Iniciando lectura de datos...")
        
        while self.running and self.gps_serial:
            try:
                if self.gps_serial.in_waiting > 0:
                    data = self.gps_serial.read(self.gps_serial.in_waiting).decode('utf-8', errors='ignore')
                    buffer += data
                    last_data_time = time.time()
                    no_data_warning = False
                    
                    # Procesar líneas completas
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        p0 = line.split(",")[0] if "," in line else ""
                        if p0.startswith("$G") and ("GGA" in p0 or "RMC" in p0):
                            nmea_count += 1
                            if nmea_count == 1:
                                print(
                                    f"  ✓ GPS: NMEA ({p0[:8]}…): {line[:56]}...",
                                    flush=True,
                                )
                            location = self.parse_nmea(line)
                            if location:
                                self._merge_gps_fix(location)
                                with self.gps_lock:
                                    snap = dict(self.gps_location)
                                self._maybe_log_gps_fix(snap)
                            elif time.time() - last_status_print > 30:
                                print(
                                    "📡 GPS: tramas sin posición fija todavía (cielo abierto ayuda)",
                                    flush=True,
                                )
                                last_status_print = time.time()
                else:
                    # Verificar si no hay datos por mucho tiempo
                    if time.time() - last_data_time > 10 and not no_data_warning:
                        print("  ⚠ GPS: No se están recibiendo datos del puerto serial")
                        print("     Verifica que el GPS esté conectado y funcionando")
                        no_data_warning = True
                    time.sleep(0.1)
            except serial.SerialException as e:
                if self.running:
                    print(f"  ✗ Error de comunicación GPS: {e}")
                    print("     El GPS puede haberse desconectado")
                time.sleep(2)
            except Exception as e:
                if self.running:
                    print(f"  ✗ Error leyendo GPS: {e}")
                time.sleep(1)
    
    def get_location_text(self):
        """
        Devuelve (texto_para_voz, texto_extra_consola o None).
        El texto de voz va corto para no cortar espeak por timeout.
        """
        if not self.gps_serial:
            lazy_rc = self.try_connect_gps_lazy("pregunta de ubicación")
            if not self.gps_serial:
                if lazy_rc == 2:
                    return (
                        "Esperá unos segundos y preguntá otra vez dónde estoy.",
                        None,
                    )
                voz = "No detecto el GPS. Revisá la consola para configurar el puerto."
                consola = (
                    "  → export GPS_SERIAL_PORT=/dev/ttyUSB0 (o 1, 2: el módulo GPS, no el modem)\n"
                    "  → export GPS_TRUST_PORT=1\n"
                    "  → reiniciar el programa (identificá el GPS con: lsusb y ls -la /dev/ttyUSB*)"
                )
                return (voz, consola)

        with self.gps_lock:
            snap = dict(self.gps_location) if self.gps_location else None

        if not snap:
            return (
                "GPS sin señal de satélites todavía. Probá al aire libre unos segundos.",
                None,
            )

        lat = snap["latitude"]
        lon = snap["longitude"]

        direccion = None
        skip_geo = os.environ.get("GPS_SKIP_GEOCODE", "").strip().lower() in (
            "1",
            "yes",
            "true",
        )
        if self.geolocator and not skip_geo:
            try:
                try:
                    tmo = int(os.environ.get("GPS_GEOCODE_TIMEOUT", "2"))
                except ValueError:
                    tmo = 2
                tmo = max(1, min(8, tmo))
                location_info = self.geolocator.reverse((lat, lon), timeout=tmo, language="es")
                if location_info:
                    address = location_info.raw.get("address", {})
                    partes_direccion = []
                    if "road" in address or "street" in address:
                        calle = address.get("road") or address.get("street", "")
                        numero = address.get("house_number", "")
                        if numero:
                            partes_direccion.append(f"{calle} {numero}")
                        elif calle:
                            partes_direccion.append(calle)
                    if "suburb" in address:
                        partes_direccion.append(address["suburb"])
                    elif "neighbourhood" in address:
                        partes_direccion.append(address["neighbourhood"])
                    if "city" in address:
                        partes_direccion.append(address["city"])
                    elif "town" in address:
                        partes_direccion.append(address["town"])
                    elif "village" in address:
                        partes_direccion.append(address["village"])
                    if "state" in address:
                        partes_direccion.append(address["state"])
                    if partes_direccion:
                        direccion = ", ".join(partes_direccion)
            except Exception:
                pass

        lat_dir = "Norte" if lat >= 0 else "Sur"
        lon_dir = "Este" if lon >= 0 else "Oeste"
        lat_abs = abs(lat)
        lon_abs = abs(lon)
        if direccion:
            # Solo dirección por voz (truncar si supera límite de espeak).
            prefix = "Estoy cerca de "
            suffix = "."
            max_total = 220
            room = max_total - len(prefix) - len(suffix)
            if room < 10:
                room = 10
            if len(direccion) <= room:
                dir_out = direccion
            else:
                chunk = direccion[:room]
                last_comma = chunk.rfind(",")
                if last_comma > room // 3:
                    dir_out = chunk[:last_comma].strip()
                else:
                    dir_out = chunk.rstrip()
            voz = f"{prefix}{dir_out}{suffix}"
            return (voz, None)
        voz = (
            f"Ubicación por satélite: {lat_abs:.3f} grados {lat_dir}, "
            f"{lon_abs:.3f} grados {lon_dir}."
        )
        return (voz, None)

    def _nav_clean_address(self, raw):
        """Quita muletillas al final de la frase de destino."""
        t = re.sub(r"\s+", " ", (raw or "").strip())
        tail = (
            r"(?:\s+)?(?:por favor|gracias|guiame|guíame|guia|guía|"
            r"navega|navegar|llevame|llévame|ayudame|ayúdame|"
            r"me puedes guiar|puedes guiarme|pod[eé]s guiarme)$"
        )
        while True:
            n = re.sub(tail, "", t, flags=re.IGNORECASE).strip(" ,.")
            if n == t:
                break
            t = n
        return t.strip(" ,.")

    def _extract_nav_destination(self, command_lower):
        """Extrae calle/lugar de frases tipo 'quiero ir a … guíame'."""
        t = re.sub(r"\s+", " ", (command_lower or "").strip())
        patterns = [
            r"quiero ir a (.+)",
            r"quiero llegar a (.+)",
            r"me llevas a (.+)",
            r"llevame a (.+)",
            r"llévame a (.+)",
            r"gu[ií]ame a (.+)",
            r"gu[ií]a(?:me)? a (.+)",
            r"navega(?:r)? a (.+)",
            r"ir hasta (.+)",
            r"destino (.+)",
        ]
        for pat in patterns:
            m = re.search(pat, t)
            if m:
                dest = self._nav_clean_address(m.group(1))
                if len(dest) >= 4:
                    return dest
        return None

    def _format_distancia_voz(self, meters):
        """Distancia en español para espeak (objetos, GPS, guía)."""
        if meters is None or meters < 0:
            return None
        if meters < 1.0:
            return f"{max(1, int(round(meters * 100)))} centímetros"
        if meters < 10.0:
            d = round(meters, 1)
            if abs(d - round(d)) < 0.05:
                return f"{int(round(d))} metros"
            return f"{d:.1f} metros"
        if meters < 1000:
            return f"{int(round(meters))} metros"
        km = meters / 1000.0
        if km < 10:
            return f"{km:.1f} kilómetros"
        return f"{int(round(km))} kilómetros"

    def _nav_distance_text(self, meters):
        t = self._format_distancia_voz(meters)
        return t if t else "0 metros"

    def _nombre_objeto_es(self, nombre_en):
        traducciones = {
            "chair": "silla",
            "person": "persona",
            "car": "auto",
            "bicycle": "bicicleta",
            "dining table": "mesa",
            "couch": "sofá",
            "bed": "cama",
            "tv": "televisor",
            "laptop": "computadora",
            "cell phone": "teléfono",
            "bottle": "botella",
            "cup": "taza",
            "book": "libro",
            "clock": "reloj",
            "cat": "gato",
            "dog": "perro",
            "bench": "banco",
            "potted plant": "planta",
            "backpack": "mochila",
            "handbag": "cartera",
        }
        return traducciones.get(nombre_en, str(nombre_en).replace("_", " "))

    def _respuesta_objeto_detectado(self, nombre_en, distancia):
        """Respuesta por voz centrada en la distancia (útil para sentarse, esquivar, etc.)."""
        nombre_es = self._nombre_objeto_es(nombre_en)
        dist_txt = self._format_distancia_voz(distancia)
        if not dist_txt:
            return f"Veo {nombre_es}, pero no pude medir la distancia."
        return f"{nombre_es.capitalize()} está a {dist_txt}."

    def _buscar_objeto_actual(self, objeto_en):
        """Devuelve (nombre_en, info) del objeto pedido, el más cercano si hay varios."""
        if not objeto_en:
            return None
        key = str(objeto_en).lower().strip()
        with self.objects_lock:
            items = list(self.current_objects.items())
        best = None
        best_d = float("inf")
        for name, info in items:
            nl = name.lower()
            if key == nl or key in nl or nl in key:
                d = float(info.get("distancia", 9999))
                if d < best_d:
                    best_d = d
                    best = (name, info)
        return best

    def _bearing_deg(self, lat1, lon1, lat2, lon2):
        import math

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dlam = math.radians(lon2 - lon1)
        x = math.sin(dlam) * math.cos(phi2)
        y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
        return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0

    def _bearing_cardinal_es(self, bearing):
        dirs = (
            "norte",
            "noreste",
            "este",
            "sureste",
            "sur",
            "suroeste",
            "oeste",
            "noroeste",
        )
        return dirs[int((bearing + 22.5) / 45.0) % 8]

    def _nav_turn_hint(self, bearing_to_dest, course_deg):
        if course_deg is None:
            return None
        diff = (bearing_to_dest - course_deg + 540.0) % 360.0 - 180.0
        if abs(diff) <= 35.0:
            return "adelante"
        if diff > 35.0:
            return "a tu derecha"
        return "a tu izquierda"

    def _home_gps_path(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "mi_casa_gps.json")

    def _load_home_location(self):
        """Carga coordenadas de casa (archivo mi_casa_gps.json o variables HOME_LAT/LON)."""
        default = {
            "latitude": -31.260652,
            "longitude": -61.475046,
            "label": "casa",
        }
        path = self._home_gps_path()
        data = dict(default)
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    data.update(loaded)
            except Exception as e:
                print(f"  ⚠ No se pudo leer {path}: {e}")
        else:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(default, f, indent=2, ensure_ascii=False)
                print(f"  ✓ Creado {path} con las coordenadas de tu casa")
            except Exception as e:
                print(f"  ⚠ No se pudo crear {path}: {e}")

        try:
            data["latitude"] = float(os.environ.get("HOME_LAT", data["latitude"]))
            data["longitude"] = float(os.environ.get("HOME_LON", data["longitude"]))
        except ValueError:
            pass
        data["label"] = os.environ.get("HOME_LABEL", data.get("label", "casa"))
        return data

    def clear_nav_destination(self):
        with self.nav_lock:
            self.nav_destination = None

    def set_nav_destination_coords(self, lat, lon, label="casa"):
        """Fija destino por coordenadas (casa guardada, sin internet)."""
        dest = {
            "latitude": float(lat),
            "longitude": float(lon),
            "label": label,
            "query": f"coords:{lat:.6f},{lon:.6f}",
            "set_at": time.time(),
            "is_home": str(label).lower() == "casa",
        }
        with self.nav_lock:
            self.nav_destination = dest

        dist_txt = ""
        with self.gps_lock:
            snap = dict(self.gps_location) if self.gps_location else None
        if snap:
            try:
                d_m = geodesic(
                    (snap["latitude"], snap["longitude"]),
                    (dest["latitude"], dest["longitude"]),
                ).meters
                dist_txt = f" En línea recta son unos {self._nav_distance_text(d_m)}."
            except Exception:
                pass

        print(
            f"🧭 Destino: {label} → {dest['latitude']:.6f}, {dest['longitude']:.6f}"
        )
        return (
            f"Te guío hacia {label}.{dist_txt} "
            "Caminá y preguntá cómo voy o cuánto falta."
        )

    def set_nav_destination_home(self):
        h = self.home_location
        return self.set_nav_destination_coords(
            h["latitude"], h["longitude"], h.get("label", "casa")
        )

    def _voice_wants_go_home(self, text):
        t = re.sub(r"\s+", " ", (text or "").strip().lower())
        if not re.search(r"\bcasa\b", t):
            return False
        if re.search(
            r"(?:llev|llév|gui|guí|ir|volver|regres|quiero|naveg|and[aá]|camina)",
            t,
        ):
            return True
        return t in ("casa", "mi casa", "a casa", "ir casa", "volver casa")

    def set_nav_destination(self, address_text):
        """Geocodifica y guarda destino. Devuelve texto para espeak."""
        if not self.geolocator:
            return "No puedo buscar direcciones sin conexión a internet."
        addr = self._nav_clean_address(address_text)
        if not addr or len(addr) < 4:
            return "Decime la calle y el número, por ejemplo: San Martín 500."

        # Por defecto Rafaela (proyecto local); override: export NAV_GEOCODE_SUFFIX="..."
        default_suffix = ", Rafaela, Santa Fe, Argentina"
        suffix = os.environ.get("NAV_GEOCODE_SUFFIX", default_suffix).strip()
        query = f"{addr}{suffix}" if suffix else addr
        try:
            tmo = int(os.environ.get("NAV_GEOCODE_TIMEOUT", "6"))
        except ValueError:
            tmo = 6
        tmo = max(3, min(12, tmo))

        try:
            loc = self.geolocator.geocode(query, timeout=tmo, language="es")
        except Exception as e:
            print(f"🧭 Geocodificar destino falló: {e}")
            return "No pude buscar esa dirección. Revisá internet e intentá otra vez."

        if not loc:
            return (
                f"No encontré {addr} en Rafaela. "
                "Probá con calle y número, por ejemplo: San Martín 500."
            )

        label = (loc.address or addr).split(",")[0].strip()
        if len(label) > 80:
            label = label[:77] + "..."

        dest = {
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "label": label,
            "query": query,
            "set_at": time.time(),
        }
        with self.nav_lock:
            self.nav_destination = dest

        dist_txt = ""
        with self.gps_lock:
            snap = dict(self.gps_location) if self.gps_location else None
        if snap:
            try:
                d_m = geodesic(
                    (snap["latitude"], snap["longitude"]),
                    (dest["latitude"], dest["longitude"]),
                ).meters
                dist_txt = f" En línea recta son unos {self._nav_distance_text(d_m)}."
            except Exception:
                pass

        print(f"🧭 Destino: {label} → {dest['latitude']:.6f}, {dest['longitude']:.6f}")
        return (
            f"Te guío hacia {label}.{dist_txt} "
            "Caminá y preguntá cómo voy o cuánto falta."
        )

    def get_nav_guidance_text(self):
        """Distancia y orientación al destino activo."""
        with self.nav_lock:
            dest = dict(self.nav_destination) if self.nav_destination else None
        if not dest:
            return (
                "No hay destino guardado. "
                "Decí por ejemplo: quiero ir a San Martín 500, guíame."
            )

        if not self.gps_serial:
            self.try_connect_gps_lazy("guía de destino")
        if not self.gps_serial:
            return "No detecto el GPS. Conectalo y preguntá otra vez."

        with self.gps_lock:
            snap = dict(self.gps_location) if self.gps_location else None
        if not snap:
            return (
                "GPS sin señal todavía. Esperá unos segundos al aire libre y preguntá otra vez."
            )

        try:
            d_m = geodesic(
                (snap["latitude"], snap["longitude"]),
                (dest["latitude"], dest["longitude"]),
            ).meters
        except Exception:
            return "No pude calcular la distancia al destino."

        try:
            arrival_m = float(os.environ.get("NAV_ARRIVAL_M", "35"))
        except ValueError:
            arrival_m = 35.0
        arrival_m = max(15.0, min(80.0, arrival_m))

        label = dest.get("label") or "el destino"
        if d_m <= arrival_m:
            self.clear_nav_destination()
            return f"Llegaste cerca de {label}. Destino cumplido."

        dist = self._nav_distance_text(d_m)
        bearing = self._bearing_deg(
            snap["latitude"], snap["longitude"], dest["latitude"], dest["longitude"]
        )
        course = snap.get("course_deg")
        turn = self._nav_turn_hint(bearing, course)

        if turn == "adelante":
            return f"Vas bien hacia {label}. Te faltan unos {dist}, sigue adelante."
        if turn in ("a tu izquierda", "a tu derecha"):
            lado = "izquierda" if turn == "a tu izquierda" else "derecha"
            return (
                f"Hacia {label} te faltan unos {dist}. "
                f"El destino está más a tu {lado}."
            )

        card = self._bearing_cardinal_es(bearing)
        return (
            f"Hacia {label} te faltan unos {dist} en línea recta. "
            f"El destino queda hacia el {card}."
        )

    def _process_nav_voice_command(self, command, command_lower, after_listen_prompt=False):
        """Comandos: ir a / guíame / cómo voy / cancelar ruta."""
        t = re.sub(r"\s+", " ", (command_lower or "").strip())

        cancel_patterns = [
            r"\bcancelar (?:la )?(?:ruta|gu[ií]a|destino|navegaci[oó]n)\b",
            r"\bparar (?:la )?(?:ruta|gu[ií]a|destino)\b",
            r"\bborrar (?:el )?destino\b",
            r"\bno quiero (?:ir|seguir)\b",
            r"\bdejar de guiar\b",
        ]
        for pat in cancel_patterns:
            if re.search(pat, t):
                self.clear_nav_destination()
                msg = "Listo, cancelé la guía al destino."
                print(f"🧭 {msg}")
                self.speak_text(msg, force=True)
                with self.user_speaking_lock:
                    self.user_speaking = False
                return True

        status_patterns = [
            r"\bc[uú]anto falta\b",
            r"\bc[oó]mo voy\b",
            r"\bdistancia al destino\b",
            r"\bd[oó]nde queda\b",
            r"\bc[uú]anto me falta\b",
            r"\bestoy cerca\b",
            r"\bllegu[eé]\b",
        ]
        with self.nav_lock:
            has_dest = self.nav_destination is not None
        for pat in status_patterns:
            if has_dest and re.search(pat, t):
                msg = self.get_nav_guidance_text()
                print(f"🧭 Guía ({command.strip()}): {msg}")
                self.speak_text(msg, force=True)
                with self.user_speaking_lock:
                    self.user_speaking = False
                return True

        if self._voice_wants_go_home(t):
            print("🧭 Destino: casa (coordenadas guardadas)")
            msg = self.set_nav_destination_home()
            print(f"🧭 {msg}")
            self.speak_text(msg, force=True)
            with self.user_speaking_lock:
                self.user_speaking = False
            return True

        dest_phrase = self._extract_nav_destination(t)
        if dest_phrase and dest_phrase.strip().lower() in ("casa", "mi casa", "a casa"):
            print("🧭 Destino: casa (coordenadas guardadas)")
            msg = self.set_nav_destination_home()
            print(f"🧭 {msg}")
            self.speak_text(msg, force=True)
            with self.user_speaking_lock:
                self.user_speaking = False
            return True

        if dest_phrase:
            print(f"🧭 Destino pedido por voz: {dest_phrase}")
            try:
                self.speak_text_sync("Buscando el destino.")
            except Exception:
                pass
            msg = self.set_nav_destination(dest_phrase)
            print(f"🧭 {msg}")
            self.speak_text(msg, force=True)
            with self.user_speaking_lock:
                self.user_speaking = False
            return True

        if after_listen_prompt and has_dest:
            if t in ("guíame", "guiame", "guia", "guía", "navega", "navegar"):
                msg = self.get_nav_guidance_text()
                print(f"🧭 Guía ({command.strip()}): {msg}")
                self.speak_text(msg, force=True)
                with self.user_speaking_lock:
                    self.user_speaking = False
                return True

        return False

    def _voice_summary_and_maybe_listen(self, summary_cycle_count, resumen):
        """
        Ciclos 1–3: solo dice lo que ve. Ciclo 4: solo pregunta y escucha.
        Devuelve True si hay que salir del bucle principal.
        """
        if summary_cycle_count % 4 == 0:
            print(f"\n🔊 Ciclo {summary_cycle_count}: ¿Deseas preguntar algo?")
            self.speak_text_sync("¿Deseas preguntar algo?")
            print("🎤 Escuchando... (puedes preguntar o decir 'terminar'/'apagar')", flush=True)
            with self.user_speaking_lock:
                self.user_speaking = True
            user_text = self.listen_once(timeout_seconds=8)
            with self.user_speaking_lock:
                self.user_speaking = False
            self.last_listen_end_time = time.time()
            if user_text:
                print(f"\n🎤 Dijiste: {user_text}\n")
                self.process_voice_command(user_text, after_listen_prompt=True)
            return not self.running

        print(f"\n🔊 RESUMEN (ciclo {summary_cycle_count}): {resumen}")
        self.speak_text(resumen)
        return False
    
    def _signal_handler(self, signum, frame):
        """Maneja señales para detener el programa correctamente"""
        print("\n" + "="*60)
        print("SALIENDO... (Señal recibida - Ctrl+C)")
        print("="*60)
        self.running = False
        # Forzar salida si no responde
        import sys
        sys.exit(0)
        
    def create_pipeline(self):
        """Crea el pipeline de DepthAI para OAK-D Lite"""
        pipeline = dai.Pipeline()
        
        # Nodo de cámara RGB
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(640, 480)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        
        # Nodo de salida RGB
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)
        
        # Nodo de cámara de profundidad (stereo)
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        
        # Nodo de profundidad
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        
        # Nodo de salida de profundidad
        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)
        
        return pipeline

    def _map_rgb_box_to_depth(self, x1, y1, x2, y2, rgb_shape, depth_shape):
        """
        YOLO usa el frame RGB (preview); el mapa de profundidad suele ser OTRO tamaño
        aunque esté alineado al RGB. Sin escalar, el centro del bbox apunta al fondo
        equivocado (ej. persona a 40 cm y lectura de pared a 4 m).
        """
        rh, rw = int(rgb_shape[0]), int(rgb_shape[1])
        dh, dw = int(depth_shape[0]), int(depth_shape[1])
        if rw <= 0 or rh <= 0:
            return int(x1), int(y1), int(x2), int(y2)
        sx = dw / float(rw)
        sy = dh / float(rh)
        dx1 = int(np.clip(round(x1 * sx), 0, dw - 1))
        dy1 = int(np.clip(round(y1 * sy), 0, dh - 1))
        dx2 = int(np.clip(round(x2 * sx), 0, dw - 1))
        dy2 = int(np.clip(round(y2 * sy), 0, dh - 1))
        if dx2 < dx1:
            dx1, dx2 = dx2, dx1
        if dy2 < dy1:
            dy1, dy2 = dy2, dy1
        return dx1, dy1, dx2, dy2

    def get_distance_in_bbox(
        self, depth_frame, x1, y1, x2, y2, percentile=50, max_mm=15000
    ):
        """
        Distancia en metros a partir de todos los píxeles válidos dentro del bbox
        (coordenadas ya en espacio del depth frame). Percentil bajo (~30) prioriza
        superficies más cercanas (útil para persona vs fondo).
        """
        if depth_frame is None:
            return None
        x1, x2 = sorted([int(x1), int(x2)])
        y1, y2 = sorted([int(y1), int(y2)])
        H, W = depth_frame.shape[:2]
        x1 = max(0, min(W - 1, x1))
        x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1))
        y2 = max(0, min(H - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        region = depth_frame[y1 : y2 + 1, x1 : x2 + 1]
        valid = region[(region > 0) & (region < max_mm)]
        if len(valid) < 3:
            return None
        d_mm = float(np.percentile(valid, np.clip(percentile, 5, 95)))
        distance_m = d_mm / 1000.0
        if distance_m < 0.08 or distance_m > 12.0:
            return None
        return distance_m
    
    def get_distance_at_point(self, depth_frame, x, y):
        """
        Obtiene la distancia en metros en un punto específico del frame de profundidad
        Usa una pequeña región alrededor del punto para mayor robustez
        
        Args:
            depth_frame: Frame de profundidad
            x, y: Coordenadas del punto **en píxeles del depth frame**
            
        Returns:
            Distancia en metros, o None si no es válida
        """
        if depth_frame is None:
            return None
        
        # Asegurar que las coordenadas estén dentro de los límites
        x = int(np.clip(x, 0, depth_frame.shape[1] - 1))
        y = int(np.clip(y, 0, depth_frame.shape[0] - 1))
        
        # Usar una pequeña región alrededor del punto para mayor robustez
        # (reduce errores de medición puntual)
        region_size = 3
        x1 = max(0, x - region_size)
        x2 = min(depth_frame.shape[1], x + region_size + 1)
        y1 = max(0, y - region_size)
        y2 = min(depth_frame.shape[0], y + region_size + 1)
        
        # Obtener profundidades válidas en la región
        region = depth_frame[y1:y2, x1:x2]
        valid_depths = region[(region > 0) & (region < 10000)]  # Filtrar 0 y valores > 10m
        
        if len(valid_depths) == 0:
            return None
        
        # Usar mediana para mayor robustez (menos afectada por valores atípicos)
        distance_mm = np.median(valid_depths)
        distance_m = distance_mm / 1000.0
        
        # Validar que la distancia sea razonable (entre 0.1m y 10m)
        # Si es 0 o muy pequeña, probablemente es un error de medición
        if distance_m < 0.1 or distance_m > 10.0:
            return None
        
        return distance_m
    
    def detect_desniveles_suelo(self, depth_frame, umbral_cambio=0.10, umbral_inclinacion=0.08, max_desnivel=0.50):
        """
        Detecta desniveles en el suelo analizando la parte inferior del frame de profundidad
        Optimizado para uso en exteriores (bordillos, escalones, rampas)
        SOLO detecta desniveles menores a 50cm para evitar confundir con objetos grandes
        
        Args:
            depth_frame: Frame de profundidad completo
            umbral_cambio: Cambio mínimo en metros para considerar un desnivel (default: 10cm - bordillos típicos)
            umbral_inclinacion: Diferencia mínima para detectar inclinación (default: 8cm)
            max_desnivel: Máximo desnivel a detectar en metros (default: 50cm) - ignora objetos grandes
            
        Returns:
            dict con información sobre desniveles detectados:
            {
                'hay_desnivel': bool,
                'tipo': 'escalon_arriba' | 'escalon_abajo' | 'inclinacion' | None,
                'altura_cambio': float (metros),
                'distancia': float (metros),
                'mensaje': str
            }
        """
        if depth_frame is None or depth_frame.size == 0:
            return {
                'hay_desnivel': False,
                'tipo': None,
                'altura_cambio': 0.0,
                'distancia': 0.0,
                'mensaje': None
            }
        
        height, width = depth_frame.shape
        # Analizar SOLO la parte más baja del frame (últimos 20% donde está el suelo real)
        # Esto evita confundir objetos grandes (heladeras, muebles) con desniveles
        suelo_inicio = int(height * 0.80)  # Solo últimos 20% del frame
        suelo_fin = height
        
        # Obtener profundidad del suelo en diferentes puntos horizontales
        # Dividir en 7 regiones para mejor resolución (más puntos = mejor detección)
        puntos_muestra = 7
        profundidades = []
        
        for i in range(puntos_muestra):
            x = int(width * (i + 0.5) / puntos_muestra)
            # Tomar promedio de profundidad en una región más grande para exteriores
            # (reduce ruido y mejora estabilidad)
            y_medio = (suelo_inicio + suelo_fin) // 2
            region_size = 15  # Aumentado de 10 a 15 para mejor estabilidad en exteriores
            
            x1 = max(0, x - region_size)
            x2 = min(width, x + region_size)
            y1 = max(suelo_inicio, y_medio - region_size)
            y2 = min(suelo_fin, y_medio + region_size)
            
            # Obtener profundidades válidas (no cero) en la región
            region = depth_frame[y1:y2, x1:x2]
            # Filtrar valores inválidos (0) y valores extremos (ruido)
            valid_depths = region[(region > 0) & (region < 10000)]  # < 10 metros
            
            if len(valid_depths) > 5:  # Necesitamos al menos 5 puntos válidos
                # Convertir de mm a metros y obtener mediana (más robusta que promedio)
                profundidad_mm = np.median(valid_depths)
                profundidad_m = profundidad_mm / 1000.0
                # Rango válido para exteriores: 0.5m a 8m (mejor para uso en calle)
                if 0.5 <= profundidad_m <= 8.0:
                    profundidades.append(profundidad_m)
                else:
                    profundidades.append(None)
            else:
                profundidades.append(None)
        
        # Filtrar valores None
        profundidades_validas = [p for p in profundidades if p is not None]
        
        if len(profundidades_validas) < 4:
            # Necesitamos al menos 4 puntos válidos para detectar desniveles confiablemente
            return {
                'hay_desnivel': False,
                'tipo': None,
                'altura_cambio': 0.0,
                'distancia': np.mean(profundidades_validas) if profundidades_validas else 0.0,
                'mensaje': None
            }
        
        # Calcular estadísticas
        profundidad_promedio = np.mean(profundidades_validas)
        profundidad_min = np.min(profundidades_validas)
        profundidad_max = np.max(profundidades_validas)
        diferencia_max = profundidad_max - profundidad_min
        
        # Detectar escalón (cambio brusco)
        # Comparar regiones adyacentes
        cambios_bruscos = []
        for i in range(len(profundidades) - 1):
            if profundidades[i] is not None and profundidades[i+1] is not None:
                cambio = abs(profundidades[i] - profundidades[i+1])
                if cambio > umbral_cambio:
                    cambios_bruscos.append({
                        'cambio': cambio,
                        'posicion': i,
                        'direccion': 'arriba' if profundidades[i+1] < profundidades[i] else 'abajo'
                    })
        
        # Detectar inclinación (cambio gradual pero significativo)
        inclinacion_detectada = diferencia_max > umbral_inclinacion and len(cambios_bruscos) == 0
        
        resultado = {
            'hay_desnivel': False,
            'tipo': None,
            'altura_cambio': 0.0,
            'distancia': profundidad_promedio,
            'mensaje': None
        }
        
        if cambios_bruscos:
            # Hay un escalón - FILTRAR: solo si es menor a max_desnivel (50cm)
            cambio_max = max(cambios_bruscos, key=lambda x: x['cambio'])
            
            # IMPORTANTE: Solo reportar si el cambio es menor a 50cm
            # Cambios mayores son objetos grandes (heladeras, muebles), no desniveles del suelo
            if cambio_max['cambio'] <= max_desnivel:
                resultado['hay_desnivel'] = True
                resultado['altura_cambio'] = cambio_max['cambio']
                resultado['tipo'] = 'escalon_arriba' if cambio_max['direccion'] == 'arriba' else 'escalon_abajo'
                
                if cambio_max['direccion'] == 'arriba':
                    resultado['mensaje'] = f"Escalón hacia arriba de {cambio_max['cambio']*100:.0f} centímetros"
                else:
                    resultado['mensaje'] = f"Escalón hacia abajo de {cambio_max['cambio']*100:.0f} centímetros"
            else:
                # Cambio muy grande = objeto grande, no desnivel
                resultado['hay_desnivel'] = False
                resultado['mensaje'] = None
        
        elif inclinacion_detectada:
            # Hay inclinación del terreno - FILTRAR: solo si es menor a max_desnivel (50cm)
            if diferencia_max <= max_desnivel:
                resultado['hay_desnivel'] = True
                resultado['tipo'] = 'inclinacion'
                resultado['altura_cambio'] = diferencia_max
                
                # Determinar dirección de la inclinación
                if profundidades[0] is not None and profundidades[-1] is not None:
                    if profundidades[0] < profundidades[-1]:
                        direccion = "hacia la derecha"
                    else:
                        direccion = "hacia la izquierda"
                else:
                    direccion = "del terreno"
                
                resultado['mensaje'] = f"Inclinación {direccion} de {diferencia_max*100:.0f} centímetros"
            else:
                # Inclinación muy grande = objeto grande, no desnivel
                resultado['hay_desnivel'] = False
                resultado['mensaje'] = None
        
        return resultado
    
    @staticmethod
    def _iou_bbox_xyxy(a, b):
        """IoU entre dos cajas [x1,y1,x2,y2]."""
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        iw = max(0, x2 - x1)
        ih = max(0, y2 - y1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _dedupe_objetos_misma_clase(self, raw_objetos, iou_threshold=0.4, centro_px_max=55, dist_m_max=0.35):
        """
        Evita contar dos veces el mismo objeto (p. ej. dos cajas YOLO sobre una persona).
        Por clase: se queda la detección con mayor confianza y descarta duplicados por IoU o
        proximidad de centro + distancia similar.
        """
        if len(raw_objetos) <= 1:
            return raw_objetos
        by_class = {}
        for o in raw_objetos:
            by_class.setdefault(o["nombre"], []).append(o)
        salida = []
        for _cls, grupo in by_class.items():
            grupo = sorted(grupo, key=lambda x: -x["confianza"])
            mantener = []
            for cand in grupo:
                dup = False
                cx, cy = cand["center_x"], cand["center_y"]
                for k in mantener:
                    if self._iou_bbox_xyxy(cand["bbox"], k["bbox"]) >= iou_threshold:
                        dup = True
                        break
                    dpx = ((cx - k["center_x"]) ** 2 + (cy - k["center_y"]) ** 2) ** 0.5
                    if dpx <= centro_px_max and abs(cand["distancia"] - k["distancia"]) <= dist_m_max:
                        dup = True
                        break
                if not dup:
                    mantener.append(cand)
            salida.extend(mantener)
        return salida

    def draw_detections(self, frame, results, depth_frame):
        """
        Dibuja las detecciones y distancias en el frame
        
        Args:
            frame: Frame RGB
            results: Resultados de YOLO
            depth_frame: Frame de profundidad
        """
        raw = []
        rgb_shape = frame.shape
        dshape = depth_frame.shape

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                class_name = self.class_names[cls]

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                dx1, dy1, dx2, dy2 = self._map_rgb_box_to_depth(
                    x1, y1, x2, y2, rgb_shape, dshape
                )
                dcx = (dx1 + dx2) // 2
                dcy = (dy1 + dy2) // 2

                # Persona: muestrear torso (zona central) + percentil cercano; evita leer el fondo.
                if class_name == "person":
                    bw, bh = dx2 - dx1, dy2 - dy1
                    distance = None
                    if bw > 6 and bh > 6:
                        ix1 = int(dx1 + bw * 0.22)
                        ix2 = int(dx2 - bw * 0.22)
                        iy1 = int(dy1 + bh * 0.12)
                        iy2 = int(dy2 - bh * 0.28)
                        distance = self.get_distance_in_bbox(
                            depth_frame, ix1, iy1, ix2, iy2, percentile=28
                        )
                    if distance is None:
                        distance = self.get_distance_in_bbox(
                            depth_frame, dx1, dy1, dx2, dy2, percentile=32
                        )
                    if distance is None:
                        distance = self.get_distance_at_point(depth_frame, dcx, dcy)
                elif class_name in ("chair", "couch", "dining table", "bench"):
                    # Asiento: bbox completo (percentil bajo) evita medir el respaldo o el fondo
                    distance = self.get_distance_in_bbox(
                        depth_frame, dx1, dy1, dx2, dy2, percentile=30
                    )
                    if distance is None:
                        distance = self.get_distance_at_point(depth_frame, dcx, dcy)
                else:
                    distance = self.get_distance_at_point(depth_frame, dcx, dcy)

                if distance is None or distance < 0.1:
                    bottom_dy = min(dy2 - max(1, (dy2 - dy1) // 25), dshape[0] - 1)
                    distance = self.get_distance_at_point(depth_frame, dcx, bottom_dy)

                if distance is not None and distance >= 0.1:
                    raw.append({
                        "nombre": class_name,
                        "distancia": distance,
                        "confianza": conf,
                        "bbox": (x1, y1, x2, y2),
                        "center_x": center_x,
                        "center_y": center_y,
                    })

        dedup = self._dedupe_objetos_misma_clase(raw)

        objetos_detectados = []
        color = (0, 255, 0)
        for o in dedup:
            x1, y1, x2, y2 = o["bbox"]
            conf = o["confianza"]
            class_name = o["nombre"]
            distance = o["distancia"]
            center_x, center_y = o["center_x"], o["center_y"]

            objetos_detectados.append({
                "nombre": class_name,
                "distancia": distance,
                "confianza": conf,
            })

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name} {conf:.2f}"
            distance_text = f"{distance:.2f}m"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - baseline - 20),
                (x1 + text_width + 10, y1),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - baseline - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )
            cv2.putText(
                frame,
                distance_text,
                (x1 + 5, y1 - baseline + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        return objetos_detectados
    
    def speak_object(self, nombre, distancia):
        """
        Habla el objeto detectado usando síntesis de voz
        
        Args:
            nombre: Nombre del objeto
            distancia: Distancia en metros
        """
        if self.tts_engine is None:
            print(f"⚠ TTS no disponible, no puedo hablar: {nombre}")
            return
            
        def _speak():
            try:
                # Traducir nombres comunes al español
                traducciones = {
                    'person': 'persona',
                    'bicycle': 'bicicleta',
                    'car': 'auto',
                    'motorcycle': 'motocicleta',
                    'airplane': 'avión',
                    'bus': 'autobús',
                    'train': 'tren',
                    'truck': 'camión',
                    'boat': 'barco',
                    'traffic light': 'semáforo',
                    'fire hydrant': 'hidrante',
                    'stop sign': 'señal de alto',
                    'parking meter': 'parquímetro',
                    'bench': 'banco',
                    'bird': 'pájaro',
                    'cat': 'gato',
                    'dog': 'perro',
                    'horse': 'caballo',
                    'sheep': 'oveja',
                    'cow': 'vaca',
                    'elephant': 'elefante',
                    'bear': 'oso',
                    'zebra': 'cebra',
                    'giraffe': 'jirafa',
                    'backpack': 'mochila',
                    'umbrella': 'paraguas',
                    'handbag': 'bolso',
                    'tie': 'corbata',
                    'suitcase': 'maleta',
                    'frisbee': 'frisbee',
                    'skis': 'esquís',
                    'snowboard': 'snowboard',
                    'sports ball': 'pelota',
                    'kite': 'cometa',
                    'baseball bat': 'bate de béisbol',
                    'baseball glove': 'guante de béisbol',
                    'skateboard': 'patineta',
                    'surfboard': 'tabla de surf',
                    'tennis racket': 'raqueta de tenis',
                    'bottle': 'botella',
                    'wine glass': 'copa de vino',
                    'cup': 'taza',
                    'fork': 'tenedor',
                    'knife': 'cuchillo',
                    'spoon': 'cuchara',
                    'bowl': 'tazón',
                    'banana': 'plátano',
                    'apple': 'manzana',
                    'sandwich': 'sándwich',
                    'orange': 'naranja',
                    'broccoli': 'brócoli',
                    'carrot': 'zanahoria',
                    'hot dog': 'hot dog',
                    'pizza': 'pizza',
                    'donut': 'donut',
                    'cake': 'pastel',
                    'chair': 'silla',
                    'couch': 'sofá',
                    'potted plant': 'planta en maceta',
                    'bed': 'cama',
                    'dining table': 'mesa de comedor',
                    'toilet': 'inodoro',
                    'tv': 'televisor',
                    'laptop': 'computadora portátil',
                    'mouse': 'ratón',
                    'remote': 'control remoto',
                    'keyboard': 'teclado',
                    'cell phone': 'teléfono celular',
                    'microwave': 'microondas',
                    'oven': 'horno',
                    'toaster': 'tostadora',
                    'sink': 'lavabo',
                    'refrigerator': 'refrigerador',
                    'book': 'libro',
                    'clock': 'reloj',
                    'vase': 'florero',
                    'scissors': 'tijeras',
                    'teddy bear': 'oso de peluche',
                    'hair drier': 'secador de pelo',
                    'toothbrush': 'cepillo de dientes'
                }
                
                nombre_es = traducciones.get(nombre.lower(), nombre)
                
                # Formatear distancia
                if distancia < 1:
                    distancia_texto = f"{int(distancia * 100)} centímetros"
                else:
                    distancia_texto = f"{distancia:.1f} metros"
                
                # Crear mensaje
                mensaje = f"{nombre_es} a {distancia_texto}"
                
                # Asegurar que el auricular esté configurado antes de hablar
                try:
                    import subprocess
                    result = subprocess.run(['pactl', 'list', 'short', 'sinks'], 
                                          capture_output=True, text=True, timeout=0.5)
                    for line in result.stdout.split('\n'):
                        if 'logi' in line.lower() or 'h390' in line.lower():
                            h390_sink = line.split()[1]
                            subprocess.run(['pactl', 'set-default-sink', h390_sink], 
                                         capture_output=True, timeout=0.5)
                            break
                except:
                    pass
                
                with self.tts_lock:
                    self.tts_engine.say(mensaje)
                    self.tts_engine.runAndWait()
            except Exception as e:
                print(f"Error al hablar: {e}")
        
        # Ejecutar en un thread separado para no bloquear
        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()
    
    def should_speak_summary(self):
        """
        Determina si se debe hablar el resumen (cada 10 segundos)
        
        Returns:
            True si se debe hablar, False si no
        """
        current_time = time.time()
        
        # Si nunca se ha hablado, hablar
        if not hasattr(self, 'last_summary_time'):
            self.last_summary_time = current_time
            return True
        
        # Si han pasado más de 10 segundos desde la última vez
        if current_time - self.last_summary_time > 10.0:
            self.last_summary_time = current_time
            return True
        return False
    
    def detect_obstaculos_proximos(self, objetos, distancia_umbral=2.0):
        """
        Detecta si hay obstáculos próximos (a menos de distancia_umbral metros)
        
        Args:
            objetos: Lista de objetos detectados
            distancia_umbral: Distancia en metros para considerar un obstáculo (default: 2.0m)
            
        Returns:
            Lista de obstáculos próximos
        """
        obstaculos = []
        for obj in objetos:
            if obj['distancia'] < distancia_umbral:
                obstaculos.append(obj)
        return obstaculos
    
    def generar_resumen_voz(self, objetos):
        """
        Genera un resumen de los objetos detectados para hablar
        
        Args:
            objetos: Lista de objetos detectados
            
        Returns:
            Mensaje de resumen
        """
        if not objetos:
            return "No estoy viendo ningún objeto en este momento"
        
        # Traducciones
        traducciones = {
            'person': 'persona', 'personas': 'personas',
            'chair': 'silla', 'chairs': 'sillas',
            'car': 'auto', 'cars': 'autos',
            'bicycle': 'bicicleta', 'bicycles': 'bicicletas',
            'dining table': 'mesa', 'tables': 'mesas',
            'couch': 'sofá', 'bed': 'cama',
            'tv': 'televisor', 'laptop': 'computadora',
            'cell phone': 'teléfono', 'bottle': 'botella',
            'cup': 'taza', 'book': 'libro', 'clock': 'reloj',
            'cat': 'gato', 'dog': 'perro'
        }
        
        # Agrupar objetos por tipo
        objetos_agrupados = {}
        for obj in objetos:
            nombre = obj['nombre']
            nombre_es = traducciones.get(nombre.lower(), nombre)
            if nombre_es not in objetos_agrupados:
                objetos_agrupados[nombre_es] = []
            objetos_agrupados[nombre_es].append(obj['distancia'])
        
        # Generar mensaje
        mensajes = []
        for nombre_es, distancias in objetos_agrupados.items():
            distancia_promedio = sum(distancias) / len(distancias)
            dist_texto = self._format_distancia_voz(distancia_promedio) or "distancia desconocida"
            
            if len(distancias) > 1:
                nm = nombre_es
                if nm == "persona":
                    nm = "personas"
                mensajes.append(f"{len(distancias)} {nm} a {dist_texto}")
            else:
                mensajes.append(f"{nombre_es} a {dist_texto}")
        
        resumen = "Estoy viendo: " + ", ".join(mensajes)
        
        # Agregar información de obstáculos
        obstaculos = self.detect_obstaculos_proximos(objetos, distancia_umbral=2.0)
        if obstaculos:
            resumen += f". Atención: hay {len(obstaculos)} obstáculo{'s' if len(obstaculos) > 1 else ''} próximo{'s' if len(obstaculos) > 1 else ''}"
        else:
            resumen += ". No hay obstáculos próximos"
        
        return resumen

    def _ocr_normalize(self, text):
        """Normaliza texto OCR para comparar y para leerlo mejor."""
        if text is None:
            return ""
        t = str(text)
        # Mantener letras, números y puntuación básica; reemplazar separadores por espacios
        t = re.sub(r"[\r\n\t]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _ocr_spanish_like(self, text, conf_max, for_voice_command=False):
        """Heurística simple para aceptar texto probable en español."""
        if not text:
            return False
        t = text.lower()
        # Por voz: el usuario pidió leer; no descartar carteles en inglés, números o sin tildes
        if for_voice_command:
            if len(t.strip()) >= 2 and re.search(
                r"[A-Za-zÁÉÍÓÚÑáéíóúñ0-9]", t, flags=re.IGNORECASE
            ):
                return conf_max >= max(0.2, self.ocr_min_confidence - 0.08)
            return False
        # Si tiene caracteres típicos de español, es buena señal
        if re.search(r"[áéíóúñ]", t, flags=re.IGNORECASE):
            return True
        # Palabras/indicadores comunes de carteles en español
        markers = [
            " el ", " la ", " los ", " las ", " una ", " un ",
            " por ", " para ", " con ", " sin ", " donde ", " donde ",
            " atención", " aviso", " favor", " prohib",
            " entrada", " salida", " horario", " baño", " farmacia",
            " hospital", " restaurante", " dirección", " piso"
        ]
        if any(m in t for m in markers):
            return True
        # Texto latino / números sin marcadores típicos: aceptar si hay confianza suficiente
        if len(text) >= 4 and re.search(r"[a-záéíóúñ0-9]", t, flags=re.IGNORECASE):
            return conf_max >= max(0.32, self.ocr_min_confidence + 0.02)
        return conf_max >= max(0.65, self.ocr_min_confidence + 0.2)

    def _ensure_ocr_reader(self):
        if self.ocr_reader is not None:
            return self.ocr_reader
        import easyocr

        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("MKL_NUM_THREADS", "2")
        with self.ocr_reader_lock:
            if self.ocr_reader is not None:
                return self.ocr_reader
            langs_env = os.environ.get("OCR_LANGS", "es,en").strip()
            langs = [x.strip() for x in langs_env.split(",") if x.strip()]
            if not langs:
                langs = ["es", "en"]
            use_gpu = os.environ.get("OCR_GPU", "").strip().lower() in (
                "1",
                "yes",
                "true",
                "cuda",
            )
            print(f"🧾 OCR: iniciando EasyOCR (idiomas {langs}, gpu={use_gpu})...", flush=True)
            self.ocr_reader = easyocr.Reader(langs, gpu=use_gpu)
            print("🧾 OCR: EasyOCR Reader creado.", flush=True)
            return self.ocr_reader

    def _ocr_preprocess_variants(self, roi_bgr, fast=False):
        """Escala + contraste (CLAHE). fast=True: una sola variante (mucho más rápido por voz)."""
        h, w = roi_bgr.shape[:2]
        min_dim = float(min(h, w))
        try:
            target = float(os.environ.get("OCR_MIN_SIDE", "520"))
        except ValueError:
            target = 520.0
        if fast:
            target = min(target, 420.0)
        target = max(320.0, min(900.0, target))
        scale = 1.0 if min_dim >= target else min(3.0, target / min_dim)
        if scale > 1.02:
            work = cv2.resize(roi_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            work = roi_bgr.copy()
        rgb_base = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
        if fast:
            return [rgb_base]
        lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l_ch)
        lab2 = cv2.merge([l2, a_ch, b_ch])
        bgr_u = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        rgb_clahe = cv2.cvtColor(bgr_u, cv2.COLOR_BGR2RGB)
        return [rgb_base, rgb_clahe]

    def _ocr_readtext_safe(self, reader, rgb, fast=False):
        """readtext con parámetros pensados para texto pequeño; fallback si la API cambia."""
        try:
            if fast:
                mag = float(os.environ.get("OCR_MAG_RATIO_FAST", "1.25"))
                adj = float(os.environ.get("OCR_ADJUST_CONTRAST_FAST", "0.5"))
                tthr = float(os.environ.get("OCR_TEXT_THR_FAST", "0.45"))
                lowt = float(os.environ.get("OCR_LOW_TEXT_FAST", "0.28"))
                lnk = float(os.environ.get("OCR_LINK_THR_FAST", "0.28"))
            else:
                mag = float(os.environ.get("OCR_MAG_RATIO", "2.0"))
                adj = float(os.environ.get("OCR_ADJUST_CONTRAST", "0.7"))
                tthr = float(os.environ.get("OCR_TEXT_THR", "0.52"))
                lowt = float(os.environ.get("OCR_LOW_TEXT", "0.32"))
                lnk = float(os.environ.get("OCR_LINK_THR", "0.32"))
        except ValueError:
            if fast:
                mag, adj, tthr, lowt, lnk = 1.25, 0.5, 0.45, 0.28, 0.28
            else:
                mag, adj, tthr, lowt, lnk = 2.0, 0.7, 0.52, 0.32, 0.32
        try:
            return reader.readtext(
                rgb,
                detail=1,
                paragraph=False,
                mag_ratio=mag,
                adjust_contrast=adj,
                text_threshold=tthr,
                low_text=lowt,
                link_threshold=lnk,
            )
        except TypeError:
            return reader.readtext(rgb, detail=1, paragraph=False, mag_ratio=max(1.2, mag))

    @staticmethod
    def _ocr_bbox_ycenter(bbox):
        try:
            a = np.asarray(bbox, dtype=float).reshape(-1, 2)
            return float(np.mean(a[:, 1]))
        except Exception:
            return 0.0

    @staticmethod
    def _voice_wants_cartel_read(text):
        """
        Pedido de leer (cartel / texto). Incluye 'cartel' y también solo 'leer' en la frase,
        porque Google STT suele cortar o no transcribir bien 'cartel'.
        """
        if not text or not str(text).strip():
            return False
        t = str(text).lower().strip()
        compact = re.sub(r'[\s.,;:!?¿¡"\'-]+', "", t)
        if "carteles" in compact or "cartel" in compact:
            return True
        if re.search(r"\bcarteles?\b", t):
            return True
        # Intención de lectura (lo que pediste: basta con "leer" en la frase)
        if re.search(r"\bleer\b", t) or re.search(r"\bleyendo\b", t):
            return True
        if re.search(r"\bl[eé]eme\b", t) or re.search(r"\bleenos\b", t):
            return True
        if re.search(r"\blee\b", t):
            return True
        return False

    def _ocr_speak_cartel(self, mensaje, from_voice_command):
        """TTS del resultado OCR; no hablar si el watchdog canceló el pedido por voz."""
        if from_voice_command and getattr(self, "_ocr_voice_cancelled", False):
            print("🧾 OCR: resultado listo pero no se anuncia (tiempo agotado o cancelado).", flush=True)
            return
        self.speak_text_sync(mensaje)

    def _ocr_job(self, frame, frame_count, from_voice_command=False, done_event=None):
        """OCR en background. from_voice_command: pedido explícito por voz (no cortar por user_speaking)."""
        started = False
        voice_done = False
        try:
            now = time.time()
            if not from_voice_command:
                with self.user_speaking_lock:
                    if self.user_speaking:
                        return

            roi = frame if frame.flags["C_CONTIGUOUS"] else np.ascontiguousarray(frame)
            h, w = roi.shape[:2]
            # 1080p + EasyOCR es muy lento; por voz usamos lado menor por defecto (más rápido).
            env_ms = "OCR_VOICE_MAX_SIDE" if from_voice_command else "OCR_INPUT_MAX_SIDE"
            default_ms = "800" if from_voice_command else "960"
            try:
                max_side = int(os.environ.get(env_ms, default_ms))
            except ValueError:
                max_side = int(default_ms)
            max_side = max(480, min(1920, max_side))
            if max(h, w) > max_side:
                scale = max_side / float(max(h, w))
                roi = cv2.resize(
                    roi,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
                h, w = roi.shape[:2]
            # Cartel suele estar en el centro; recorte un poco más ajustado opcional
            ym, xm = float(os.environ.get("OCR_CROP_YMARGIN", "0.04")), float(
                os.environ.get("OCR_CROP_XMARGIN", "0.04")
            )
            ym = max(0.0, min(0.2, ym))
            xm = max(0.0, min(0.2, xm))
            y1, y2 = int(h * ym), int(h * (1.0 - ym))
            x1, x2 = int(w * xm), int(w * (1.0 - xm))
            roi = roi[y1:y2, x1:x2]
            if roi.size == 0:
                return

            started = True
            t_ocr0 = time.time()
            print(
                "🧾 OCR: analizando imagen (la primera vez puede tardar al cargar el modelo)...",
                flush=True,
            )

            reader = self._ensure_ocr_reader()
            print("🧾 OCR: escaneando texto en la imagen...", flush=True)
            # Por voz: una variante + readtext más liviano (evita cuelgue larguísimo en CPU).
            ocr_fast = bool(from_voice_command)
            if from_voice_command and os.environ.get("OCR_VOICE_FULL", "").strip().lower() in (
                "1",
                "yes",
                "true",
            ):
                ocr_fast = False
            all_detections = []
            for rgb in self._ocr_preprocess_variants(roi, fast=ocr_fast):
                chunk = self._ocr_readtext_safe(reader, rgb, fast=ocr_fast)
                if chunk:
                    all_detections.extend(chunk)

            if not all_detections:
                return

            # Unificar líneas repetidas entre variantes (quedarse con mayor confianza)
            by_key = {}
            for bbox, text, conf in all_detections:
                t = self._ocr_normalize(text)
                if not t:
                    continue
                cf = float(conf) if conf is not None else 0.0
                key = re.sub(r"\s+", " ", t.lower())
                if key not in by_key or cf > by_key[key][2]:
                    by_key[key] = (bbox, t, cf)

            results = list(by_key.values())
            results.sort(key=lambda r: self._ocr_bbox_ycenter(r[0]))

            raw_texts = []
            for _, text_raw, conf_raw in sorted(
                results, key=lambda x: -x[2]
            )[:8]:
                tn = self._ocr_normalize(text_raw)
                if tn:
                    raw_texts.append((tn, float(conf_raw) if conf_raw is not None else 0.0))
            if raw_texts:
                print(f"🧾 OCR raw (top): {raw_texts}")

            candidates = []
            conf_max = 0.0
            min_conf_ocr = (
                max(0.15, self.ocr_min_confidence - 0.1)
                if from_voice_command
                else self.ocr_min_confidence
            )
            for bbox, text, conf in results:
                t = self._ocr_normalize(text)
                if not t:
                    continue
                if conf is None:
                    continue
                conf_f = float(conf)
                conf_max = max(conf_max, conf_f)
                if conf_f < min_conf_ocr:
                    continue
                if not re.search(r"[A-Za-zÁÉÍÓÚÑáéíóúñ0-9]", t, flags=re.IGNORECASE):
                    continue
                if len(t) < 2:
                    continue
                candidates.append((bbox, t, conf_f))

            if not candidates:
                # Best-effort: si easyocr detecta texto pero lo descartó por confianza,
                # intentamos anunciar el fragmento más legible a partir del raw_texts.
                # (Esto evita quedarnos en silencio cuando el cartel es poco nítido.)
                fallback_words = []
                for tn, _conf in raw_texts:
                    for w_ in tn.split():
                        w_ = w_.strip()
                        if len(w_) < 2:
                            continue
                        if not re.search(r"[A-Za-zÁÉÍÓÚÑáéíóúñ0-9]", w_, flags=re.IGNORECASE):
                            continue
                        if len(w_) >= 3 and not re.search(
                            r"[aeiouáéíóú0-9]", w_, flags=re.IGNORECASE
                        ):
                            continue
                        fallback_words.append(w_)
                        if len(fallback_words) >= self.ocr_max_words:
                            break
                    if len(fallback_words) >= self.ocr_max_words:
                        break

                if not fallback_words:
                    return

                texto = " ".join(fallback_words)[: self.ocr_max_chars].strip()
                texto_key = re.sub(r"[^a-zA-Z0-9ÁÉÍÓÚÑáéíóúñ ]+", "", texto).lower()
                texto_key = re.sub(r"\s+", " ", texto_key).strip()
                if texto_key == self.ocr_last_text_key and not from_voice_command:
                    return
                if texto_key == self.ocr_last_text_key and from_voice_command:
                    print(f"\n🪧 CARTEL (mismo texto, pediste otra vez): {texto}\n")

                mensaje = f"Hay un cartel que dice: {texto}"
                print(f"\n🪧 CARTEL LEÍDO (OCR, best effort): {texto}\n")
                self.ocr_last_text_key = texto_key
                self.ocr_last_announce_time = now
                self._ocr_speak_cartel(mensaje, from_voice_command)
                voice_done = True
                return

            candidates_top = sorted(candidates, key=lambda x: x[2], reverse=True)[:8]
            candidates_top.sort(key=lambda x: self._ocr_bbox_ycenter(x[0]))

            words = []
            for _, t, _ in candidates_top:
                for w_ in t.split():
                    w_ = w_.strip()
                    if not w_:
                        continue
                    if len(w_) == 1 and not re.match(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]", w_, flags=re.IGNORECASE):
                        continue
                    words.append(w_)
                    if len(words) >= self.ocr_max_words:
                        break
                if len(words) >= self.ocr_max_words:
                    break

            if not words:
                return

            texto = " ".join(words)
            texto = texto[: self.ocr_max_chars].strip()
            texto_key = re.sub(r"[^a-zA-Z0-9ÁÉÍÓÚÑáéíóúñ ]+", "", texto).lower()
            texto_key = re.sub(r"\s+", " ", texto_key).strip()

            if not self._ocr_spanish_like(
                texto, conf_max, for_voice_command=from_voice_command
            ):
                return
            if texto_key == self.ocr_last_text_key and not from_voice_command:
                return
            if texto_key == self.ocr_last_text_key and from_voice_command:
                print(f"\n🪧 CARTEL (mismo texto, pediste otra vez): {texto}\n")

            mensaje = f"Hay un cartel que dice: {texto}"
            print(f"\n🪧 CARTEL LEÍDO (OCR): {texto}\n")

            # Actualizar cooldown antes de hablar
            self.ocr_last_text_key = texto_key
            self.ocr_last_announce_time = now

            # Hablar sin bloquear el loop principal (estamos en thread)
            self._ocr_speak_cartel(mensaje, from_voice_command)
            voice_done = True
        except Exception as e:
            print(f"🧾 OCR error: {e}")
        finally:
            try:
                if started:
                    print(
                        f"🧾 OCR: fin análisis en {time.time() - t_ocr0:.1f}s",
                        flush=True,
                    )
            except Exception:
                pass
            with self.ocr_thread_lock:
                self.ocr_in_progress = False
            if done_event is not None:
                done_event.set()
            if (
                from_voice_command
                and started
                and not voice_done
                and not getattr(self, "_ocr_voice_cancelled", False)
            ):
                try:
                    self.speak_text_sync(
                        "No pude leer texto claro en el cartel. "
                        "Acercalo al centro de la imagen, con buena luz, y decí leer otra vez."
                    )
                except Exception:
                    pass
                print("🧾 OCR: sin texto aceptable (fin del análisis).", flush=True)

    def try_auto_read_cartel(self, frame, frame_count):
        """
        Dispara el OCR en background para leer automáticamente un cartel en español.
        """
        try:
            now = time.time()
            with self.user_speaking_lock:
                if self.user_speaking:
                    return
            if frame_count % self.ocr_check_every_n_frames != 0:
                return
            if now - self.ocr_last_announce_time < self.ocr_cooldown_seconds:
                return
            if now - getattr(self, "last_listen_end_time", 0.0) < 8.0:
                return
            if hasattr(self, "last_summary_time") and (now - getattr(self, "last_summary_time", 0)) < 4.0:
                return

            with self.ocr_thread_lock:
                if self.ocr_in_progress:
                    return
                self.ocr_in_progress = True

            frame_copy = frame.copy()
            threading.Thread(
                target=self._ocr_job,
                args=(frame_copy, frame_count, False),
                daemon=True,
            ).start()
        except Exception:
            return
    
    def listen_for_commands(self):
        """Escucha comandos de voz en un thread separado"""
        if self.microphone is None or self.recognizer is None:
            print("⚠ Micrófono no disponible, reconocimiento de voz deshabilitado")
            return
            
        # Suprimir mensajes de ALSA durante la escucha
        import contextlib
        
        @contextlib.contextmanager
        def suppress_stderr():
            with open(os.devnull, 'w') as devnull:
                old_stderr = sys.stderr
                sys.stderr = devnull
                try:
                    yield
                finally:
                    sys.stderr = old_stderr
        
        print("🎤 Escuchando comandos de voz...")
        # Configurar umbral inicial más bajo para mayor sensibilidad
        self.recognizer.energy_threshold = 300  # Más sensible
        print(f"🎤 Umbral de energía inicial: {self.recognizer.energy_threshold:.0f}")
        print("💡 Habla claramente cuando veas '🎤 Escuchando...'\n")
        
        listen_count = 0
        consecutive_failures = 0
        
        while self.running:
            try:
                with self.microphone as source:
                    # Reajustar umbral cada 10 iteraciones (más frecuente)
                    listen_count += 1
                    if listen_count % 10 == 0:
                        try:
                            with suppress_stderr():
                                # Ajustar ruido ambiental con más tiempo
                                self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                            # Forzar umbral más bajo para mayor sensibilidad
                            current_threshold = self.recognizer.energy_threshold
                            # Reducir a máximo 300 (muy sensible)
                            self.recognizer.energy_threshold = min(current_threshold * 0.6, 300)
                            if listen_count % 50 == 0:  # Cada 50 iteraciones, mostrar umbral
                                print(f"🎤 Umbral ajustado: {self.recognizer.energy_threshold:.0f}")
                        except Exception as e:
                            # Si falla el ajuste, forzar umbral bajo
                            self.recognizer.energy_threshold = 300
                    
                    # Escuchar con timeout más largo y phrase_time_limit más largo
                    try:
                        print("🎤 Escuchando... (habla ahora)", flush=True)
                        with suppress_stderr():
                            # Timeout más largo (3 segundos) y phrase_time_limit más largo (8 segundos)
                            audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=8)
                        print("🎤 ✓ Audio detectado!", flush=True)
                        
                        # Activar bandera
                        with self.user_speaking_lock:
                            self.user_speaking = True
                        
                        # Reconocer el audio con reintentos
                        text = None
                        max_retries = 3
                        for retry in range(max_retries):
                            try:
                                text = self.recognizer.recognize_google(audio, language='es-ES')
                                text = text.lower()
                                consecutive_failures = 0  # Resetear contador de fallos
                                break
                            except sr.UnknownValueError:
                                if retry < max_retries - 1:
                                    print(f"🎤 ⚠ Intento {retry + 1}/{max_retries}: No se entendió, reintentando...", flush=True)
                                    time.sleep(0.2)
                                else:
                                    print("🎤 ⚠ No se entendió después de varios intentos, intenta de nuevo\n", flush=True)
                                    with self.user_speaking_lock:
                                        self.user_speaking = False
                                    consecutive_failures += 1
                                    # Si hay muchos fallos, reducir umbral
                                    if consecutive_failures >= 5:
                                        self.recognizer.energy_threshold = max(50, self.recognizer.energy_threshold * 0.5)
                                        print(f"🎤 ⚠ Reduciendo umbral a {self.recognizer.energy_threshold:.0f} por fallos consecutivos")
                                        consecutive_failures = 0
                            except sr.RequestError as e:
                                print(f"❌ Error de conexión: {e}\n", flush=True)
                                with self.user_speaking_lock:
                                    self.user_speaking = False
                                time.sleep(1)  # Esperar antes de reintentar
                                break
                        
                        # Si se reconoció texto, procesarlo
                        if text:
                            # Mostrar en terminal (SIEMPRE)
                            print("\n" + "="*60)
                            print(f"🎤 COMANDO DETECTADO: {text}")
                            print("="*60 + "\n")
                            
                            # Agregar a la cola
                            self.voice_commands.put(text)
                            
                    except sr.WaitTimeoutError:
                        # No se detectó audio, continuar silenciosamente
                        # Reducir umbral si hay muchos timeouts
                        if listen_count % 30 == 0:
                            self.recognizer.energy_threshold = max(50, self.recognizer.energy_threshold * 0.8)
                        continue
            except Exception as e:
                if self.running:
                    # No imprimir errores de ALSA
                    if "ALSA" not in str(e):
                        print(f"Error escuchando: {e}")
                time.sleep(0.1)
    
    def grab_fresh_rgb_for_ocr(self):
        """
        Mientras listen_once() corre, el bucle principal no consume las colas OAK:
        se acumulan frames; al pedir leer, vaciamos y tomamos un par rgb+depth nuevo
        para que la imagen sea la de cuando terminaste de hablar (cartel alineado a la cámara).
        """
        try:
            rq = getattr(self, "q_rgb", None)
            dq = getattr(self, "q_depth", None)
            if rq is None or dq is None:
                return None
            n = 0
            while rq.has():
                rq.tryGet()
                n += 1
            while dq.has():
                dq.tryGet()
                n += 1
            if n:
                print(f"🧾 OCR: descartados {n} frames en cola (acumulados mientras escuchabas).", flush=True)
            t0 = time.time()
            while time.time() - t0 < 1.5:
                if rq.has() and dq.has():
                    in_rgb = rq.tryGet()
                    in_depth = dq.tryGet()
                    if in_rgb is not None and in_depth is not None:
                        bgr = in_rgb.getCvFrame()
                        return np.ascontiguousarray(bgr.copy())
                time.sleep(0.002)
            print("🧾 OCR: no llegó par rgb+depth fresco a tiempo.", flush=True)
            return None
        except Exception as e:
            print(f"🧾 grab_fresh_rgb_for_ocr: {e}", flush=True)
            return None

    def listen_once(self, timeout_seconds=6):
        """Escucha una sola frase (para cuando pregunta '¿Deseas preguntar algo?'). Devuelve texto o None."""
        if self.microphone is None or self.recognizer is None:
            return None
        try:
            import contextlib
            with open(os.devnull, 'w') as devnull:
                old_err = sys.stderr
                sys.stderr = devnull
                try:
                    with self.microphone as source:
                        audio = self.recognizer.listen(
                            source,
                            timeout=timeout_seconds,
                            phrase_time_limit=min(10, timeout_seconds + 2),
                        )
                    text = self.recognizer.recognize_google(audio, language='es-ES')
                    return text.strip().lower() if text else None
                except sr.WaitTimeoutError:
                    return None
                except sr.UnknownValueError:
                    return None
                except sr.RequestError:
                    return None
                finally:
                    sys.stderr = old_err
        except Exception:
            return None
    
    def process_voice_command(self, command, after_listen_prompt=False):
        """
        Procesa un comando de voz y responde.
        after_listen_prompt: True si acaba de sonar '¿Deseas preguntar algo?' (permite
        entender 'estoy' o 'dónde' cuando Google solo transcribe parte de la frase).
        """
        command_lower = command.lower()
        
        # Apagar: solo palabras/frases completas (\b) para no confundir con "cartel", "para leer", etc.
        apagar_patterns = [
            r"\bapagar\b",
            r"\bapaga\b",
            r"\bapagalo\b",
            r"\bapagá\b",
            r"\bapagálo\b",
            r"\bcerrar\b",
            r"\bcierra\b",
            r"\bcerralo\b",
            r"\bcerrá\b",
            r"\bcerrálo\b",
            r"\bsalir\b",
            r"\bsalilo\b",
            r"\bdetener\b",
            r"\bdetén\b",
            r"\bdetenlo\b",
            r"\bdetenélo\b",
            # No \bparar\b ("sin parar"); sí imperativos claros:
            r"\bparalo\b",
            r"\bpará\b",
            r"\bparálo\b",
            # No \btermina\b ("donde termina el cartel"); sí "terminar":
            r"\bterminar\b",
            r"\bterminarlo\b",
            r"\bterminá\b",
            r"\bterminálo\b",
            r"\bapagar\s+el\s+sistema\b",
            r"\bapaga\s+el\s+sistema\b",
            r"\bcierra\s+el\s+sistema\b",
            r"\bcerrar\s+el\s+sistema\b",
        ]

        for pattern in apagar_patterns:
            if re.search(pattern, command_lower):
                print(f"🎤 ✓✓✓ COMANDO DE APAGADO DETECTADO: '{command}' (patrón: {pattern})")
                respuesta = "Apagando el sistema. Hasta luego."
                print(f"📢 Respuesta: {respuesta}")
                
                # Mensaje de despedida (espeak si no hay pyttsx3)
                try:
                    import subprocess
                    if self.tts_engine is not None:
                        with self.tts_lock:
                            self.tts_engine.say(respuesta)
                            self.tts_engine.runAndWait()
                    else:
                        subprocess.run(["espeak", "-v", "es", respuesta], capture_output=True, timeout=5)
                    print("📢 Mensaje de despedida completado")
                except Exception as e:
                    print(f"⚠ Error al hablar despedida: {e}")
                
                # Desactivar bandera
                with self.user_speaking_lock:
                    self.user_speaking = False
                
                # Esperar un momento para asegurar que el audio termine
                time.sleep(0.5)
                
                # Detener el programa
                self.running = False
                return True

        # OCR bajo demanda: "cartel", "puedes leer", "léeme", etc.
        if self._voice_wants_cartel_read(command_lower):
            with self.ocr_thread_lock:
                if self.ocr_in_progress:
                    self.speak_text_sync("Esperá, sigo leyendo el cartel anterior.")
                    with self.user_speaking_lock:
                        self.user_speaking = False
                    return False
                self.ocr_in_progress = True

            # Imagen FRESCA: mientras escuchabas no se leían las colas; last_frame sería viejo.
            frame_copy = self.grab_fresh_rgb_for_ocr()
            if frame_copy is None and self.last_frame_for_ocr is not None:
                try:
                    frame_copy = np.ascontiguousarray(self.last_frame_for_ocr.copy())
                    print("🧾 OCR: usando respaldo (último frame del bucle principal).", flush=True)
                except Exception as e:
                    print(f"🧾 OCR: no se pudo copiar respaldo: {e}")
                    frame_copy = None
            if frame_copy is None:
                respuesta = "No puedo obtener imagen de la cámara ahora mismo."
                print(f"📢 Respuesta: {respuesta}")
                self.speak_text(respuesta, force=True)
                with self.ocr_thread_lock:
                    self.ocr_in_progress = False
                with self.user_speaking_lock:
                    self.user_speaking = False
                return False

            fc = int(self.last_frame_count_for_ocr)

            try:
                self.speak_text_sync("Estoy leyendo el cartel.")
            except Exception:
                pass

            self._ocr_voice_cancelled = False
            ocr_done_evt = threading.Event()

            def _ocr_voice_worker():
                try:
                    self._ocr_job(
                        frame_copy,
                        fc,
                        from_voice_command=True,
                        done_event=ocr_done_evt,
                    )
                except Exception as ex:
                    print(f"🧾 OCR error (voz): {ex}")
                    import traceback

                    traceback.print_exc()
                finally:
                    ocr_done_evt.set()

            threading.Thread(target=_ocr_voice_worker, daemon=True).start()

            def _ocr_voice_watchdog():
                try:
                    sec = float(os.environ.get("OCR_VOICE_TIMEOUT_SEC", "60"))
                except ValueError:
                    sec = 60.0
                sec = max(25.0, min(240.0, sec))
                time.sleep(sec)
                if ocr_done_evt.is_set():
                    return
                self._ocr_voice_cancelled = True
                print(
                    f"🧾 OCR: sin terminar en {sec:.0f}s — se desbloquea el programa. "
                    "(EasyOCR puede seguir en segundo plano; mirá si apareció 'precarga lista'.)",
                    flush=True,
                )
                with self.ocr_thread_lock:
                    self.ocr_in_progress = False
                try:
                    self.speak_text_sync(
                        "Tardó demasiado leer el cartel. "
                        "Esperá a que termine de cargar el reconocedor y decí leer otra vez."
                    )
                except Exception:
                    pass

            threading.Thread(target=_ocr_voice_watchdog, daemon=True).start()

            with self.user_speaking_lock:
                self.user_speaking = False
            return False

        if self._process_nav_voice_command(command, command_lower, after_listen_prompt):
            return True
        
        # Comandos GPS
        gps_patterns = [
            r"dónde estoy",
            r"donde estoy",
            r"cuál es mi ubicación",
            r"cual es mi ubicacion",
            r"qué es mi ubicación",
            r"que es mi ubicacion",
            r"dime dónde estoy",
            r"dime donde estoy",
            r"ubicación",
            r"ubicacion",
            r"coordenadas",
            r"posición",
            r"posicion",
            r"\bgps\b",
            r"localización",
            r"localizacion",
            r"en qué lugar",
            r"en que lugar",
        ]

        def _es_consulta_gps(text):
            if not text or not str(text).strip():
                return False
            t = re.sub(r"\s+", " ", str(text).lower().strip())
            for pattern in gps_patterns:
                if re.search(pattern, t):
                    return True
            if after_listen_prompt:
                # Reconocedor a veces corta "dónde estoy" → solo "estoy" / "donde"
                if t in (
                    "estoy",
                    "donde",
                    "dónde",
                    "ubicame",
                    "ubícame",
                    "localizame",
                    "localízame",
                ):
                    return True
                if t.startswith("donde ") or t.startswith("dónde "):
                    return True
            return False

        if _es_consulta_gps(command_lower):
            voz_txt, extra_consola = self.get_location_text()
            print(f"📍 GPS ({command.strip()}): {voz_txt}")
            if extra_consola:
                print(extra_consola)
            if not self.gps_serial:
                print(
                    "  ℹ Huawei y GPS suelen ser ttyUSB distintos: lsusb + ls -la /dev/ttyUSB*"
                )
            self.speak_text(voz_txt, force=True)
            with self.user_speaking_lock:
                self.user_speaking = False
            return True
        
        # Traducciones de objetos al inglés (para buscar en YOLO)
        traducciones_inv = {
            'silla': 'chair',
            'persona': 'person',
            'personas': 'person',
            'gente': 'person',
            'auto': 'car',
            'coche': 'car',
            'carro': 'car',
            'bicicleta': 'bicycle',
            'bici': 'bicycle',
            'mesa': 'dining table',
            'mesa de comedor': 'dining table',
            'sofá': 'couch',
            'sofa': 'couch',
            'cama': 'bed',
            'televisor': 'tv',
            'tv': 'tv',
            'televisión': 'tv',
            'computadora': 'laptop',
            'laptop': 'laptop',
            'portátil': 'laptop',
            'teléfono': 'cell phone',
            'celular': 'cell phone',
            'móvil': 'cell phone',
            'botella': 'bottle',
            'taza': 'cup',
            'libro': 'book',
            'reloj': 'clock',
            'gato': 'cat',
            'perro': 'dog',
            'perros': 'dog',
            'perra': 'dog',
            'perrito': 'dog'
        }
        
        # Patrones de preguntas (siempre responden con distancia si lo detecta)
        pregunta_patterns = [
            r'estás viendo (?:una |un |el |la )?(\w+)',
            r'estas viendo (?:una |un |el |la )?(\w+)',
            r'ves (?:una |un |el |la |alguna |algún |algun )?(\w+)',
            r'hay (?:una |un |el |la )?(\w+)',
            r'puedes ver (?:una |un |el |la )?(\w+)',
            r'(?:dónde|donde) (?:está |esta |están |estan )?(?:la |el |una |un )?(\w+)',
            r'(?:a cuánto|a cuanto) (?:está |esta )?(?:la |el |una |un )?(\w+)',
            r'(\w+) distancia',
            r'a qué distancia (?:está |están |esta |estan )?(?:una |un |el |la )?(\w+)',
            r'a que distancia (?:está |están |esta |estan )?(?:una |un |el |la )?(\w+)',
            r'cuánto (?:está |están |esta |estan )?(?:una |un |el |la )?(\w+)',
            r'cuanto (?:está |están |esta |estan )?(?:una |un |el |la )?(\w+)',
        ]
        
        objeto_encontrado = None
        
        # Buscar objeto en el comando
        for pattern in pregunta_patterns:
            match = re.search(pattern, command)
            if match:
                objeto_es = match.group(1).lower()
                # Traducir al inglés si es necesario
                objeto_en = traducciones_inv.get(objeto_es, objeto_es)
                objeto_encontrado = objeto_en
                break
        
        # Si no se encontró con patrones, buscar directamente
        if objeto_encontrado is None:
            for obj_es, obj_en in traducciones_inv.items():
                if obj_es in command:
                    objeto_encontrado = obj_en
                    break
        
        # Buscar el objeto en los detectados actualmente (siempre con distancia)
        if objeto_encontrado:
            hallado = self._buscar_objeto_actual(objeto_encontrado)
            if hallado:
                obj_name, obj_info = hallado
                respuesta = self._respuesta_objeto_detectado(
                    obj_name, obj_info.get("distancia")
                )
            else:
                nombre_es = self._nombre_objeto_es(objeto_encontrado)
                respuesta = f"No veo {nombre_es} en este momento."
            print(f"📢 Respuesta: {respuesta}")
            self.speak_text(respuesta, force=True)
            with self.user_speaking_lock:
                self.user_speaking = False
            return True
        
        # Desactivar bandera si no se procesó ningún comando
        with self.user_speaking_lock:
            self.user_speaking = False
        return False
    
    def speak_text(self, text, force=False):
        """
        Habla un texto usando síntesis de voz (espeak si no hay pyttsx3, para SSH/embebido).
        """
        def _speak():
            try:
                import subprocess
                # Sin pyttsx3 (modo SSH/embebido): usar espeak, no aborta
                if self.tts_engine is None:
                    with self.tts_lock:
                        subprocess.run(
                            ["espeak", "-v", "es", "-s", "160", text],
                            capture_output=True,
                            timeout=45,
                        )
                    return
                # Si no es forzado, verificar si el usuario está hablando
                if not force:
                    with self.user_speaking_lock:
                        if self.user_speaking:
                            print("🔇 Pausando síntesis de voz (usuario hablando)")
                            return  # No hablar si el usuario está hablando (solo para resúmenes automáticos)
                
                # Asegurar que el auricular esté configurado antes de hablar
                try:
                    import subprocess
                    result = subprocess.run(['pactl', 'list', 'short', 'sinks'], 
                                          capture_output=True, text=True, timeout=0.5)
                    for line in result.stdout.split('\n'):
                        if 'logi' in line.lower() or 'h390' in line.lower():
                            h390_sink = line.split()[1]
                            subprocess.run(['pactl', 'set-default-sink', h390_sink], 
                                         capture_output=True, timeout=0.5)
                            subprocess.run(['pactl', 'set-sink-volume', h390_sink, '100%'], 
                                         capture_output=True, timeout=0.5)
                            break
                except:
                    pass
                
                # Si no es forzado, verificar nuevamente antes de hablar
                if not force:
                    with self.user_speaking_lock:
                        if self.user_speaking:
                            print("🔇 Pausando síntesis de voz (usuario hablando)")
                            return
                
                with self.tts_lock:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
            except Exception as e:
                print(f"Error al hablar: {e}")
        
        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()
    
    def speak_text_sync(self, text):
        """Habla y espera a que termine (no superpone con lo siguiente). Para el ciclo '¿Deseas preguntar algo?'"""
        try:
            import subprocess
            if self.tts_engine is None:
                with self.tts_lock:
                    subprocess.run(
                        ["espeak", "-v", "es", "-s", "160", text],
                        capture_output=True,
                        timeout=45,
                    )
            else:
                with self.tts_lock:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
        except Exception as e:
            print(f"Error al hablar (sync): {e}")
    
    def run(self):
        """Ejecuta el loop principal de detección"""
        print("\n" + "="*60)
        print("DETECCIÓN DE OBJETOS INICIADA")
        print("="*60)
        print("\n📹 Detección activa (por SSH: ventana en pantalla virtual, no visible)")
        print("🔊 Audio: Resumen por voz cada 10 s (espeak). Conecta auricular/speaker en la Jetson.")
        print("   - Objetos y distancias, obstáculos próximos, desniveles en el suelo")
        print("🎤 Micrófono: Comandos por voz (si está conectado)")
        print("🌐 Si te alejás del WiFi, SSH puede cortarse: usá tmux/screen o datos en la Jetson (modem USB).")
        print("📍 GPS + modem: export GPS_SERIAL_PORT=/dev/ttyUSB1 — identificar con lsusb y ls /dev/ttyUSB*")
        print("🧭 Guía: quiero ir a [calle y número] guíame — o llevame a casa (GPS guardado)")
        print("   Luego: cómo voy / cuánto falta. Cancelar: cancelar ruta")
        print("⌨️  PARA SALIR: Ctrl+C en esta terminal, o di 'terminar'/'apagar' cuando pregunte '¿Deseas preguntar algo?'")
        print("   3 veces dice lo que ve; al 4.º ciclo solo pregunta si deseas preguntar algo.\n")
        print("="*60 + "\n")
        
        frame_count = 0
        start_time = time.time()
        window_created = False
        summary_cycle_count = 0  # Cada 4 ciclos pregunta "¿Deseas preguntar algo?" y activa el micrófono
        gui_enabled = True  # OpenCV puede venir en versión "headless" sin soporte de HighGUI
        
        try:
            while self.running:
                # Verificar tecla PRIMERO (más responsivo)
                if window_created and gui_enabled:
                    try:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == ord('Q') or key == 27:  # 'q', 'Q' o ESC
                            print("\n" + "="*60)
                            print("SALIENDO... (Presionaste 'Q' o ESC)")
                            print("="*60)
                            self.running = False
                            break
                        
                        # Verificar si la ventana fue cerrada
                        if cv2.getWindowProperty("Detección de Objetos - OAK-D Lite", cv2.WND_PROP_VISIBLE) < 1:
                            print("\n" + "="*60)
                            print("SALIENDO... (Ventana cerrada)")
                            print("="*60)
                            self.running = False
                            break
                    except Exception:
                        # Si la ventana no se puede consultar por falta de GUI, pasamos a headless
                        gui_enabled = False
                
                # Verificar si hay frames disponibles antes de obtenerlos
                if not self.q_rgb.has() or not self.q_depth.has():
                    if not self.running:
                        break
                    time.sleep(0.01)  # Pequeña pausa para no saturar CPU
                    continue
                
                # Verificar running nuevamente
                if not self.running:
                    break
                
                # Par rgb+depth más reciente: vaciar la cola evita YOLO sobre frames viejos.
                try:
                    in_rgb, in_depth = None, None
                    while self.q_rgb.has() and self.q_depth.has():
                        r = self.q_rgb.tryGet()
                        d = self.q_depth.tryGet()
                        if r is not None and d is not None:
                            in_rgb, in_depth = r, d
                        else:
                            break
                    if in_rgb is None or in_depth is None:
                        continue
                except Exception as e:
                    if self.running:
                        print(f"Error obteniendo frames: {e}")
                    continue
                
                frame = in_rgb.getCvFrame()
                depth_frame = in_depth.getFrame()
                # Referencia al último frame (sin copia por vuelta: 1080p a 30 fps mataba el FPS).
                # OCR por voz usa grab_fresh_rgb_for_ocr() tras escuchar + copia ahí.
                self.last_frame_for_ocr = frame
                self.last_frame_count_for_ocr = frame_count
                
                # Crear ventana si no existe (solo si aún creemos que hay GUI)
                if not window_created and gui_enabled:
                    try:
                        cv2.namedWindow("Detección de Objetos - OAK-D Lite", cv2.WINDOW_NORMAL)
                        window_created = True
                        print("✓ Ventana de video creada")
                    except Exception as e:
                        gui_enabled = False
                        window_created = False
                        print(f"⚠ OpenCV sin soporte de ventana (headless). Ejecutando sin GUI. Detalle: {e}")
                
                if not self.running:
                    break
                
                # Ejecutar detección YOLO
                results = self.model(frame, verbose=False)
                
                if not self.running:
                    break
                
                # Dibujar detecciones
                objetos = self.draw_detections(frame, results, depth_frame)
                
                # Detectar desniveles en el suelo (cada 5 frames para no saturar)
                desnivel_info = None
                if frame_count % 5 == 0:
                    desnivel_info = self.detect_desniveles_suelo(depth_frame)
                    
                    if desnivel_info['hay_desnivel']:
                        mensaje_actual = desnivel_info['mensaje']
                        if (mensaje_actual != self.ultimo_desnivel_anunciado or 
                            (frame_count - self.frame_desnivel_anterior) > 30):
                            print(f"\n⚠️  DESNIVEL DETECTADO: {mensaje_actual}")
                            self.speak_text(f"Atención: {mensaje_actual}", force=True)
                            self.ultimo_desnivel_anunciado = mensaje_actual
                            self.frame_desnivel_anterior = frame_count
                            cv2.putText(frame, "DESNIVEL!", (10, 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                            cv2.putText(frame, desnivel_info['mensaje'], (10, 90), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Actualizar objetos actuales (por clase: guardar el más cercano)
                with self.objects_lock:
                    self.current_objects = {}
                    for obj in objetos:
                        nom = obj["nombre"]
                        prev = self.current_objects.get(nom)
                        if prev is None or obj["distancia"] < prev["distancia"]:
                            self.current_objects[nom] = {
                                "distancia": obj["distancia"],
                                "confianza": obj["confianza"],
                            }
                
                if frame_count % 10 == 0:
                    if objetos:
                        print(f"\n--- Frame {frame_count} ---")
                        for obj in objetos:
                            print(f"  • {obj['nombre']}: {obj['distancia']:.2f}m (confianza: {obj['confianza']:.2%})")
                        if desnivel_info and desnivel_info['hay_desnivel']:
                            print(f"  ⚠️  Desnivel: {desnivel_info['mensaje']}")
                    else:
                        print(f"\n--- Frame {frame_count} --- Sin objetos detectados")
                        if desnivel_info and desnivel_info['hay_desnivel']:
                            print(f"  ⚠️  Desnivel: {desnivel_info['mensaje']}")
                
                # Hablar resumen cada 10 s: 3 ciclos lo que ve, 4.º solo pregunta
                if objetos:
                    if self.should_speak_summary():
                        summary_cycle_count += 1
                        resumen = self.generar_resumen_voz(objetos)
                        if self._voice_summary_and_maybe_listen(
                            summary_cycle_count, resumen
                        ):
                            break
                else:
                    if self.should_speak_summary():
                        summary_cycle_count += 1
                        ocr_in_prog = False
                        try:
                            with self.ocr_thread_lock:
                                ocr_in_prog = bool(
                                    getattr(self, "ocr_in_progress", False)
                                )
                        except Exception:
                            ocr_in_prog = bool(
                                getattr(self, "ocr_in_progress", False)
                            )
                        if ocr_in_prog:
                            resumen = "Estoy leyendo un cartel."
                        else:
                            resumen = (
                                "No estoy viendo ningún objeto en este momento. "
                                "No hay obstáculos próximos"
                            )
                        if self._voice_summary_and_maybe_listen(
                            summary_cycle_count, resumen
                        ):
                            break
                
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = 30 / elapsed
                    print(f"FPS: {fps:.2f}")
                    start_time = time.time()

                # OCR automático desactivado: ahora solo se lee cuando el usuario lo pide (ej. dice "cartel")
                
                h, w = frame.shape[:2]
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (350, 80), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                cv2.putText(frame, "Presiona 'Q' o ESC para SALIR", (20, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "O di 'APAGAR' por voz", (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                if not self.running:
                    break
                
                if gui_enabled:
                    try:
                        cv2.imshow("Detección de Objetos - OAK-D Lite", frame)
                    except Exception:
                        # Si imshow falla (por headless), desactivamos GUI
                        gui_enabled = False
                    
        except KeyboardInterrupt:
            print("\n" + "="*60)
            print("SALIENDO... (Interrupción detectada)")
            print("="*60)
            self.running = False
        except Exception as e:
            print(f"\nError durante la ejecución: {e}")
            self.running = False
        finally:
            print("\nCerrando cámara y ventanas...")
            self.running = False
            
            # Cerrar todas las ventanas de OpenCV
            try:
                cv2.destroyAllWindows()
            except:
                pass
            
            # Detener el motor de voz si está hablando
            try:
                if self.tts_engine is not None:
                    self.tts_engine.stop()
            except:
                pass
            
            # Cerrar el dispositivo
            try:
                if hasattr(self, 'device'):
                    del self.device
            except:
                pass
            
            print("✓ Programa finalizado correctamente")

def main():
    """Función principal"""
    print("=" * 60)
    print("DETECTOR DE OBJETOS CON YOLO Y OAK-D LITE")
    print("=" * 60)
    
    # Puedes cambiar el modelo aquí:
    # 'yolov8n.pt' - Nano (más rápido, menos preciso)
    # 'yolov8s.pt' - Small (balanceado)
    # 'yolov8m.pt' - Medium (más preciso, más lento)
    # 'yolov8l.pt' - Large (muy preciso, lento)
    # 'yolov8x.pt' - XLarge (máxima precisión, muy lento)
    
    detector = OAKDObjectDetector(model_name='yolov8n.pt')
    detector.run()

if __name__ == "__main__":
    main()

