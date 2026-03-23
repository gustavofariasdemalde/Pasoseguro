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
import os
import sys
import signal
import serial
import serial.tools.list_ports
from geopy.geocoders import Nominatim

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
        
        # Obtener colas de salida (con timeout para evitar bloqueos)
        self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.q_depth = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        
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
        self.ocr_min_confidence = 0.45
        self.ocr_max_chars = 90
        self.ocr_max_words = 12
        self.last_listen_end_time = 0.0  # Evita que el OCR hable justo después de escuchar al usuario
        self.ocr_in_progress = False
        self.ocr_thread_lock = threading.Lock()
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
        
        # Inicializar geocodificador para obtener direcciones
        try:
            self.geolocator = Nominatim(user_agent="oakd_detector")
            print("  ✓ Geocodificador inicializado (para obtener direcciones)")
        except Exception as e:
            print(f"  ⚠ Error inicializando geocodificador: {e}")
            self.geolocator = None
        
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
                # Intentar múltiples baudrates comunes para GPS
                baudrates = [4800, 9600, 38400, 115200]
                gps_connected = False
                for baudrate in baudrates:
                    try:
                        self.gps_serial = serial.Serial(gps_port, baudrate=baudrate, timeout=1)
                        print(f"  ✓ GPS conectado en: {gps_port} (baudrate: {baudrate})")
                        gps_connected = True
                        break
                    except serial.SerialException as e:
                        if baudrate == baudrates[-1]:  # Último intento
                            print(f"  ✗ Error conectando GPS en {gps_port}: {e}")
                            print(f"     Probados baudrates: {baudrates}")
                        continue
                
                if gps_connected:
                    # Iniciar thread para leer GPS
                    self.gps_thread = threading.Thread(target=self.read_gps_data, daemon=True)
                    self.gps_thread.start()
                    print("  ✓ GPS iniciado, leyendo datos...")
                    print("  ℹ El GPS necesita estar al aire libre para recibir señal de satélites")
                else:
                    self.gps_serial = None
                    print("  ⚠ No se pudo conectar al GPS, funcionalidad deshabilitada")
            else:
                print("  ⚠ GPS no encontrado, funcionalidad deshabilitada")
                print("  Conecta el módulo GPS por USB y ejecuta: ./activar_gps.sh")
        except Exception as e:
            print(f"  ⚠ Error inicializando GPS: {e}")
            import traceback
            traceback.print_exc()
            print("  La funcionalidad GPS estará deshabilitada")
            self.gps_serial = None
    
    def find_gps_port(self):
        """Busca el puerto serial del GPS"""
        print("  Buscando puerto GPS...")
        import glob
        
        # Intentar múltiples veces (el puerto puede tardar en aparecer)
        for intento in range(3):
            # Primero, buscar directamente en /dev/ttyUSB* (más confiable)
            usb_ports = glob.glob('/dev/ttyUSB*')
            if usb_ports:
                if intento == 0:
                    print(f"  Encontrados puertos USB: {usb_ports}")
            for port in sorted(usb_ports):
                if os.path.exists(port):
                    try:
                        # Verificar permisos
                        if not os.access(port, os.R_OK | os.W_OK):
                            if intento == 0:
                                print(f"  ⚠ Puerto {port} existe pero no tiene permisos (ejecuta: sudo chmod 666 {port})")
                            continue
                        # Intentar abrir para verificar que es un puerto serial válido
                        test_serial = serial.Serial(port, baudrate=4800, timeout=0.5)
                        test_serial.close()
                        print(f"  ✓ Puerto {port} encontrado y accesible")
                        return port
                    except serial.SerialException as e:
                        if intento == 0:
                            print(f"  ⚠ Puerto {port} no se puede abrir: {e}")
                        continue
                    except (OSError, PermissionError) as e:
                        if intento == 0:
                            print(f"  ⚠ Error de permisos en {port}: {e}")
                            print(f"     Ejecuta: sudo chmod 666 {port}")
                        continue
            
            # Si no se encontró, esperar un poco y reintentar
            if intento < 2:
                time.sleep(1)
        
        # Si no se encuentra en /dev/ttyUSB*, buscar en puertos ACM
        acm_ports = glob.glob('/dev/ttyACM*')
        if acm_ports:
            print(f"  Encontrados puertos ACM: {acm_ports}")
        for port in sorted(acm_ports):
            if os.path.exists(port):
                try:
                    if not os.access(port, os.R_OK | os.W_OK):
                        continue
                    test_serial = serial.Serial(port, baudrate=4800, timeout=0.5)
                    test_serial.close()
                    print(f"  ✓ Puerto {port} encontrado y accesible")
                    return port
                except (serial.SerialException, OSError, PermissionError):
                    continue
        
        # Como último recurso, buscar en la lista de puertos de Python
        print("  Buscando en lista de puertos del sistema...")
        ports = serial.tools.list_ports.comports()
        for port in ports:
            # Buscar por descripción o nombre
            desc = port.description.lower()
            print(f"  Puerto encontrado: {port.device} - {port.description}")
            if 'gps' in desc or 'usb' in desc or 'serial' in desc or 'ch340' in desc or 'ch34' in desc:
                try:
                    if not os.access(port.device, os.R_OK | os.W_OK):
                        print(f"  ⚠ Puerto {port.device} no tiene permisos")
                        continue
                    test_serial = serial.Serial(port.device, baudrate=4800, timeout=0.5)
                    test_serial.close()
                    print(f"  ✓ Puerto {port.device} encontrado y accesible")
                    return port.device
                except (serial.SerialException, OSError, PermissionError) as e:
                    print(f"  ⚠ Puerto {port.device} no se puede abrir: {e}")
                    continue
        
        # Si no se encuentra, probar puertos comunes
        print("  Probando puertos comunes...")
        common_ports = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0', '/dev/ttyACM1']
        for port in common_ports:
            if os.path.exists(port):
                try:
                    if not os.access(port, os.R_OK | os.W_OK):
                        continue
                    test_serial = serial.Serial(port, baudrate=4800, timeout=0.5)
                    test_serial.close()
                    print(f"  ✓ Puerto {port} encontrado y accesible")
                    return port
                except:
                    continue
        
        print("  ✗ No se encontró ningún puerto GPS accesible")
        print("  Sugerencias:")
        print("    1. Ejecuta: ./activar_gps.sh")
        print("    2. Verifica que el GPS esté conectado: lsusb | grep CH340")
        print("    3. Verifica permisos: ls -la /dev/ttyUSB*")
        return None
    
    def parse_nmea(self, nmea_sentence):
        """Parsea una oración NMEA para extraer coordenadas GPS"""
        try:
            if nmea_sentence.startswith('$GPGGA'):
                parts = nmea_sentence.split(',')
                if len(parts) >= 10 and parts[2] and parts[4]:  # Tiene latitud y longitud
                    # Latitud (formato: DDMM.MMMM)
                    lat_str = parts[2]
                    if len(lat_str) >= 4:
                        lat_deg = float(lat_str[:2])
                        lat_min = float(lat_str[2:])
                        latitude = lat_deg + lat_min / 60.0
                        # Corregir: permitir tanto 'N' como 'S'
                        if parts[3] == 'S':
                            latitude = -latitude
                        elif parts[3] != 'N':
                            return None  # Solo si no es N ni S
                    else:
                        return None
                    
                    # Longitud (formato: DDDMM.MMMM)
                    lon_str = parts[4]
                    if len(lon_str) >= 5:
                        lon_deg = float(lon_str[:3])
                        lon_min = float(lon_str[3:])
                        longitude = lon_deg + lon_min / 60.0
                        # Corregir: permitir tanto 'E' como 'W'
                        if parts[5] == 'W':
                            longitude = -longitude
                        elif parts[5] != 'E':
                            return None  # Solo si no es E ni W
                    else:
                        return None
                    
                    # Calidad de señal
                    quality = int(parts[6]) if parts[6] else 0
                    
                    if quality > 0:  # Solo si hay señal GPS válida
                        return {
                            'latitude': latitude,
                            'longitude': longitude,
                            'quality': quality,
                            'timestamp': time.time()
                        }
        except Exception as e:
            pass
        return None
    
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
                        if line.startswith('$GP'):
                            nmea_count += 1
                            # Mostrar primera línea NMEA recibida como confirmación
                            if nmea_count == 1:
                                print(f"  ✓ GPS: Recibiendo datos NMEA (primera línea: {line[:50]}...)")
                            
                            location = self.parse_nmea(line)
                            if location:
                                with self.gps_lock:
                                    self.gps_location = location
                                    print(f"📍 GPS: Lat {location['latitude']:.6f}, Lon {location['longitude']:.6f} (calidad: {location['quality']})")
                            # Mostrar estado cada 30 segundos si no hay señal
                            elif time.time() - last_status_print > 30:
                                print("📡 GPS recibiendo datos pero sin señal de satélites (necesita estar al aire libre)")
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
        """Obtiene la ubicación actual en formato texto con dirección"""
        # Verificar si el GPS está conectado
        if not self.gps_serial:
            return "El módulo GPS no está conectado. Conecta el GPS por USB y reinicia el programa"
        
        with self.gps_lock:
            if self.gps_location:
                lat = self.gps_location['latitude']
                lon = self.gps_location['longitude']
                quality = self.gps_location['quality']
                
                # Intentar obtener dirección (geocodificación inversa)
                direccion = None
                if self.geolocator:
                    try:
                        location_info = self.geolocator.reverse((lat, lon), timeout=5, language='es')
                        if location_info:
                            address = location_info.raw.get('address', {})
                            
                            # Construir dirección legible
                            partes_direccion = []
                            
                            # Calle y número
                            if 'road' in address or 'street' in address:
                                calle = address.get('road') or address.get('street', '')
                                numero = address.get('house_number', '')
                                if numero:
                                    partes_direccion.append(f"{calle} {numero}")
                                elif calle:
                                    partes_direccion.append(calle)
                            
                            # Barrio o suburbio
                            if 'suburb' in address:
                                partes_direccion.append(address['suburb'])
                            elif 'neighbourhood' in address:
                                partes_direccion.append(address['neighbourhood'])
                            
                            # Ciudad
                            if 'city' in address:
                                partes_direccion.append(address['city'])
                            elif 'town' in address:
                                partes_direccion.append(address['town'])
                            elif 'village' in address:
                                partes_direccion.append(address['village'])
                            
                            # Provincia/Estado
                            if 'state' in address:
                                partes_direccion.append(address['state'])
                            
                            if partes_direccion:
                                direccion = ", ".join(partes_direccion)
                    except Exception as e:
                        # Si falla la geocodificación, continuar sin dirección
                        pass
                
                # Formatear respuesta
                if direccion:
                    # Formatear coordenadas también
                    lat_dir = "Norte" if lat >= 0 else "Sur"
                    lon_dir = "Este" if lon >= 0 else "Oeste"
                    lat_abs = abs(lat)
                    lon_abs = abs(lon)
                    
                    return f"Estoy en {direccion}. Coordenadas: {lat_abs:.4f} grados {lat_dir}, {lon_abs:.4f} grados {lon_dir}"
                else:
                    # Si no se pudo obtener dirección, solo coordenadas
                    lat_dir = "Norte" if lat >= 0 else "Sur"
                    lon_dir = "Este" if lon >= 0 else "Oeste"
                    lat_abs = abs(lat)
                    lon_abs = abs(lon)
                    
                    return f"Mi ubicación es: {lat_abs:.4f} grados {lat_dir}, {lon_abs:.4f} grados {lon_dir}"
            else:
                # GPS conectado pero sin señal
                return "El GPS está conectado pero no tiene señal de satélites. Necesito estar al aire libre para recibir señal GPS. El WiFi no es suficiente, necesito ver los satélites"
    
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
    
    def get_distance_at_point(self, depth_frame, x, y):
        """
        Obtiene la distancia en metros en un punto específico del frame de profundidad
        Usa una pequeña región alrededor del punto para mayor robustez
        
        Args:
            depth_frame: Frame de profundidad
            x, y: Coordenadas del punto
            
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
    
    def draw_detections(self, frame, results, depth_frame):
        """
        Dibuja las detecciones y distancias en el frame
        
        Args:
            frame: Frame RGB
            results: Resultados de YOLO
            depth_frame: Frame de profundidad
        """
        objetos_detectados = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Obtener coordenadas del bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Obtener clase y confianza
                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                class_name = self.class_names[cls]
                
                # Calcular el centro del objeto
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Obtener distancia en el centro del objeto
                # Si falla, intentar en la parte inferior del objeto (más cerca del suelo)
                distance = self.get_distance_at_point(depth_frame, center_x, center_y)
                
                # Si la distancia es inválida o 0, intentar en la parte inferior del bounding box
                if distance is None or distance < 0.1:
                    # Probar en la parte inferior del objeto (más cerca del suelo)
                    bottom_y = min(y2 - 5, depth_frame.shape[0] - 1)  # 5 píxeles desde el borde inferior
                    distance = self.get_distance_at_point(depth_frame, center_x, bottom_y)
                
                # Almacenar información del objeto solo si la distancia es válida
                if distance is not None and distance >= 0.1:
                    objetos_detectados.append({
                        'nombre': class_name,
                        'distancia': distance,
                        'confianza': conf
                    })
                    
                    # Dibujar bounding box
                    color = (0, 255, 0)  # Verde
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Preparar texto con información
                    label = f"{class_name} {conf:.2f}"
                    distance_text = f"{distance:.2f}m"
                    
                    # Obtener tamaño del texto
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    
                    # Dibujar fondo para el texto
                    cv2.rectangle(
                        frame,
                        (x1, y1 - text_height - baseline - 20),
                        (x1 + text_width + 10, y1),
                        color,
                        -1
                    )
                    
                    # Dibujar texto de clase y confianza
                    cv2.putText(
                        frame,
                        label,
                        (x1 + 5, y1 - baseline - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1
                    )
                    
                    # Dibujar texto de distancia
                    cv2.putText(
                        frame,
                        distance_text,
                        (x1 + 5, y1 - baseline + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1
                    )
                    
                    # Dibujar punto central
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
            if distancia_promedio < 1:
                dist_texto = f"{int(distancia_promedio * 100)} centímetros"
            else:
                dist_texto = f"{distancia_promedio:.1f} metros"
            
            if len(distancias) > 1:
                mensajes.append(f"{len(distancias)} {nombre_es} a {dist_texto}")
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

    def _ocr_spanish_like(self, text, conf_max):
        """Heurística simple para aceptar texto probable en español."""
        if not text:
            return False
        t = text.lower()
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
        # Si tiene caracteres típicos o marcadores ya devolvimos True.
        # Si no, aceptamos si la confianza es razonablemente alta.
        # (Más permisivo para no dejar pasar carteles reales.)
        return conf_max >= max(0.75, self.ocr_min_confidence + 0.25)

    def _ensure_ocr_reader(self):
        if self.ocr_reader is not None:
            return self.ocr_reader
        # EasyOCR carga un modelo; evitar hacerlo en cada frame
        import easyocr
        # gpu=False: más seguro en Orin Nano si no queremos forzar CUDA
        self.ocr_reader = easyocr.Reader(["es"], gpu=False)
        return self.ocr_reader

    def _ocr_job(self, frame, frame_count):
        """OCR en background para no bloquear el loop principal."""
        try:
            now = time.time()
            # No hablar/leer si estamos esperando al usuario
            with self.user_speaking_lock:
                if self.user_speaking:
                    return

            roi = frame
            h, w = roi.shape[:2]
            # Recortar a zona central amplia: reduce ruido pero no pierde carteles
            y1, y2 = int(h * 0.05), int(h * 0.95)
            x1, x2 = int(w * 0.05), int(w * 0.95)
            roi = roi[y1:y2, x1:x2]
            if roi.size == 0:
                return

            # Asegurar tamaño mínimo para OCR
            min_dim = min(roi.shape[:2])
            if min_dim < 250:
                scale = 2.0
                roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            reader = self._ensure_ocr_reader()
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            results = reader.readtext(rgb, detail=1, paragraph=False)
            if not results:
                return

            raw_texts = []
            for _, text_raw, conf_raw in results[:5]:
                tn = self._ocr_normalize(text_raw)
                if tn:
                    raw_texts.append((tn, float(conf_raw) if conf_raw is not None else 0.0))
            if raw_texts:
                print(f"🧾 OCR raw (top): {raw_texts}")

            candidates = []
            conf_max = 0.0
            for bbox, text, conf in results:
                t = self._ocr_normalize(text)
                if not t:
                    continue
                if conf is None:
                    continue
                conf_f = float(conf)
                conf_max = max(conf_max, conf_f)
                if conf_f < self.ocr_min_confidence:
                    continue
                if not re.search(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]", t, flags=re.IGNORECASE):
                    continue
                if len(t) < 3:
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
                        if len(w_) < 3:
                            continue
                        if not re.search(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]", w_, flags=re.IGNORECASE):
                            continue
                        if not re.search(r"[aeiouáéíóú]", w_, flags=re.IGNORECASE):
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
                if texto_key == self.ocr_last_text_key:
                    return

                mensaje = f"Hay un cartel que dice: {texto}"
                print(f"\n🪧 CARTEL LEÍDO (OCR, best effort): {texto}\n")
                self.ocr_last_text_key = texto_key
                self.ocr_last_announce_time = now
                self.speak_text_sync(mensaje)
                return

            candidates_top = sorted(candidates, key=lambda x: x[2], reverse=True)[:6]
            candidates_top.sort(key=lambda x: (x[0][0][1] + x[0][2][1]) / 2.0)

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

            if not self._ocr_spanish_like(texto, conf_max):
                return
            if texto_key == self.ocr_last_text_key:
                return

            mensaje = f"Hay un cartel que dice: {texto}"
            print(f"\n🪧 CARTEL LEÍDO (OCR): {texto}\n")

            # Actualizar cooldown antes de hablar
            self.ocr_last_text_key = texto_key
            self.ocr_last_announce_time = now

            # Hablar sin bloquear el loop principal (estamos en thread)
            self.speak_text_sync(mensaje)
        except Exception as e:
            print(f"🧾 OCR error: {e}")
        finally:
            with self.ocr_thread_lock:
                self.ocr_in_progress = False

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
            threading.Thread(target=self._ocr_job, args=(frame_copy, frame_count), daemon=True).start()
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
                        audio = self.recognizer.listen(source, timeout=timeout_seconds, phrase_time_limit=5)
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
    
    def process_voice_command(self, command):
        """Procesa un comando de voz y responde"""
        # El usuario está hablando, mantener bandera activa durante el procesamiento
        command_lower = command.lower()
        
        # Comando para apagar el programa - PATRONES MÁS FLEXIBLES
        apagar_patterns = [
            r'apagar',
            r'apaga',
            r'apagalo',
            r'apagalo',
            r'apagá',
            r'apagálo',
            r'cerrar',
            r'cierra',
            r'cerralo',
            r'cerrá',
            r'cerrálo',
            r'salir',
            r'sale',
            r'salilo',
            r'detener',
            r'detén',
            r'detenlo',
            r'detenélo',
            r'parar',
            r'para',
            r'paralo',
            r'pará',
            r'parálo',
            r'terminar',
            r'termina',
            r'terminarlo',
            r'terminá',
            r'terminálo',
            r'apaga sistema',
            r'apagar sistema',
            r'cierra sistema',
            r'cerrar sistema'
        ]
        
        # Verificar si es comando de apagado - BÚSQUEDA MÁS FLEXIBLE
        command_words = command_lower.split()
        for pattern in apagar_patterns:
            # Buscar patrón completo o como palabra individual
            if re.search(pattern, command_lower) or any(re.search(pattern, word) for word in command_words):
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

        # OCR bajo demanda: si el usuario dice "cartel", leer el texto del último frame
        if re.search(r"\bcartel\b", command_lower):
            # Aviso inmediato para que el usuario entienda que ahora está leyendo
            try:
                self.speak_text_sync("Estoy leyendo el cartel.")
            except:
                pass
            # Pequeña espera: ayuda a que el frame del cartel sea más estable
            time.sleep(1.0)
            if self.last_frame_for_ocr is None:
                respuesta = "No puedo ver un cartel ahora mismo."
                print(f"📢 Respuesta: {respuesta}")
                self.speak_text(respuesta, force=True)
                with self.user_speaking_lock:
                    self.user_speaking = False
                return False

            try:
                # Evitar que OCR se ejecute a la vez (por si acaso)
                with self.ocr_thread_lock:
                    if self.ocr_in_progress:
                        return False
                    self.ocr_in_progress = True
                self._ocr_job(self.last_frame_for_ocr, self.last_frame_count_for_ocr)
            finally:
                with self.ocr_thread_lock:
                    self.ocr_in_progress = False

            with self.user_speaking_lock:
                self.user_speaking = False
            return False
        
        # Comandos GPS
        gps_patterns = [
            r'dónde estoy',
            r'donde estoy',
            r'cuál es mi ubicación',
            r'cual es mi ubicacion',
            r'qué es mi ubicación',
            r'que es mi ubicacion',
            r'dime dónde estoy',
            r'dime donde estoy',
            r'ubicación',
            r'ubicacion',
            r'coordenadas',
            r'posición',
            r'posicion',
            r'gps'
        ]
        
        # Verificar si es una pregunta GPS
        for pattern in gps_patterns:
            if re.search(pattern, command_lower):
                location_text = self.get_location_text()
                print(f"🎤 Pregunta GPS: {command}")
                print(f"📍 Respuesta: {location_text}")
                self.speak_text(location_text, force=True)  # Forzar para responder al usuario
                # Desactivar bandera después de responder
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
        
        # Patrones de preguntas
        pregunta_patterns = [
            r'estás viendo (?:una |un |el |la )?(\w+)',
            r'ves (?:una |un |el |la )?(\w+)',
            r'hay (?:una |un |el |la )?(\w+)',
            r'puedes ver (?:una |un |el |la )?(\w+)',
            r'(\w+) distancia',
            r'a qué distancia (?:está |están )?(?:una |un |el |la )?(\w+)',
            r'cuánto (?:está |están )?(?:una |un |el |la )?(\w+)',
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
        
        # Buscar el objeto en los detectados actualmente
        if objeto_encontrado:
            with self.objects_lock:
                if objeto_encontrado in self.current_objects:
                    obj_info = self.current_objects[objeto_encontrado]
                    distancia = obj_info['distancia']
                    
                    # Traducir nombre al español para la respuesta
                    traducciones = {
                        'chair': 'silla',
                        'person': 'persona',
                        'car': 'auto',
                        'bicycle': 'bicicleta',
                        'dining table': 'mesa',
                        'couch': 'sofá',
                        'bed': 'cama',
                        'tv': 'televisor',
                        'laptop': 'computadora',
                        'cell phone': 'teléfono',
                        'bottle': 'botella',
                        'cup': 'taza',
                        'book': 'libro',
                        'clock': 'reloj',
                        'cat': 'gato',
                        'dog': 'perro'
                    }
                    
                    nombre_es = traducciones.get(objeto_encontrado, objeto_encontrado)
                    
                    if distancia < 1:
                        respuesta = f"Sí, estoy viendo {nombre_es} a {int(distancia * 100)} centímetros"
                    else:
                        respuesta = f"Sí, estoy viendo {nombre_es} a {distancia:.1f} metros"
                    
                    print(f"📢 Respuesta: {respuesta}")
                    self.speak_text(respuesta, force=True)  # Forzar para responder al usuario
                    # Desactivar bandera después de responder
                    with self.user_speaking_lock:
                        self.user_speaking = False
                    return True
                else:
                    # Buscar variaciones del nombre
                    for obj_name in self.current_objects.keys():
                        if objeto_encontrado in obj_name.lower() or obj_name.lower() in objeto_encontrado:
                            obj_info = self.current_objects[obj_name]
                            distancia = obj_info['distancia']
                            
                            traducciones = {
                                'chair': 'silla',
                                'person': 'persona',
                                'car': 'auto',
                                'bicycle': 'bicicleta',
                                'dining table': 'mesa',
                                'couch': 'sofá',
                                'bed': 'cama',
                                'tv': 'televisor',
                                'laptop': 'computadora',
                                'cell phone': 'teléfono',
                                'bottle': 'botella',
                                'cup': 'taza',
                                'book': 'libro',
                                'clock': 'reloj',
                                'cat': 'gato',
                                'dog': 'perro'
                            }
                            
                            nombre_es = traducciones.get(obj_name, obj_name)
                            
                            if distancia < 1:
                                respuesta = f"Sí, estoy viendo {nombre_es} a {int(distancia * 100)} centímetros"
                            else:
                                respuesta = f"Sí, estoy viendo {nombre_es} a {distancia:.1f} metros"
                            
                            print(f"📢 Respuesta: {respuesta}")
                            self.speak_text(respuesta, force=True)  # Forzar para responder al usuario
                            # Desactivar bandera después de responder
                            with self.user_speaking_lock:
                                self.user_speaking = False
                            return True
                    
                    # No se encontró el objeto
                    respuesta = f"No, no estoy viendo {objeto_encontrado} en este momento"
                    print(f"📢 Respuesta: {respuesta}")
                    self.speak_text(respuesta, force=True)  # Forzar para responder al usuario
                    # Desactivar bandera después de responder
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
                        subprocess.run(["espeak", "-v", "es", text], capture_output=True, timeout=15)
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
                    subprocess.run(["espeak", "-v", "es", text], capture_output=True, timeout=30)
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
        print("⌨️  PARA SALIR: Ctrl+C en esta terminal, o di 'terminar'/'apagar' cuando pregunte '¿Deseas preguntar algo?'")
        print("   Cada 4 ciclos de lo que ve, te preguntará si quieres preguntar algo y ahí podrás hablar.\n")
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
                
                # Obtener frames (sin bloqueo)
                try:
                    in_rgb = self.q_rgb.tryGet()
                    in_depth = self.q_depth.tryGet()
                    
                    if in_rgb is None or in_depth is None:
                        continue
                except Exception as e:
                    if self.running:
                        print(f"Error obteniendo frames: {e}")
                    continue
                
                frame = in_rgb.getCvFrame()
                depth_frame = in_depth.getFrame()
                # Guardar último frame para OCR bajo demanda (cuando el usuario diga "cartel")
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
                
                # Actualizar objetos actuales para responder preguntas
                with self.objects_lock:
                    self.current_objects = {}
                    for obj in objetos:
                        self.current_objects[obj['nombre']] = {
                            'distancia': obj['distancia'],
                            'confianza': obj['confianza']
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
                
                # Hablar resumen cada 10 segundos (lo que ve)
                if objetos:
                    if self.should_speak_summary():
                        summary_cycle_count += 1
                        resumen = self.generar_resumen_voz(objetos)
                        print(f"\n🔊 RESUMEN: {resumen}")
                        if summary_cycle_count % 4 == 0:
                            # Ciclo 4: hablar y esperar a que termine, luego preguntar y escuchar (sin superponer)
                            self.speak_text_sync(resumen)
                            # Siempre preguntar en ciclo 4.
                            # El lock de audio evita que el OCR y la pregunta se solapen.
                            self.speak_text_sync("¿Deseas preguntar algo?")
                            print("🎤 Escuchando... (puedes preguntar o decir 'terminar'/'apagar')", flush=True)
                            with self.user_speaking_lock:
                                self.user_speaking = True
                            user_text = self.listen_once(timeout_seconds=6)
                            with self.user_speaking_lock:
                                self.user_speaking = False
                            self.last_listen_end_time = time.time()
                            if user_text:
                                print(f"\n🎤 Dijiste: {user_text}\n")
                                self.process_voice_command(user_text)
                                if not self.running:
                                    break
                        else:
                            self.speak_text(resumen)
                else:
                    if self.should_speak_summary():
                        summary_cycle_count += 1
                        # Si acabamos de leer un cartel (o el OCR aún está en progreso),
                        # evita repetir "no veo nada" justo después.
                        ocr_in_prog = False
                        try:
                            with self.ocr_thread_lock:
                                ocr_in_prog = bool(getattr(self, "ocr_in_progress", False))
                        except:
                            ocr_in_prog = bool(getattr(self, "ocr_in_progress", False))
                        if ocr_in_prog or (time.time() - getattr(self, "ocr_last_announce_time", 0.0)) < 15.0:
                            resumen = "Estoy leyendo un cartel."
                        else:
                            resumen = "No estoy viendo ningún objeto en este momento. No hay obstáculos próximos"
                        print(f"\n🔊 RESUMEN: {resumen}")
                        if summary_cycle_count % 4 == 0:
                            self.speak_text_sync(resumen)
                            # Siempre preguntar en ciclo 4.
                            self.speak_text_sync("¿Deseas preguntar algo?")
                            print("🎤 Escuchando... (puedes preguntar o decir 'terminar'/'apagar')", flush=True)
                            with self.user_speaking_lock:
                                self.user_speaking = True
                            user_text = self.listen_once(timeout_seconds=6)
                            with self.user_speaking_lock:
                                self.user_speaking = False
                            self.last_listen_end_time = time.time()
                            if user_text:
                                print(f"\n🎤 Dijiste: {user_text}\n")
                                self.process_voice_command(user_text)
                                if not self.running:
                                    break
                        else:
                            self.speak_text(resumen)
                
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

