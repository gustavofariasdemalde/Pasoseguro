#!/usr/bin/env python3
"""
Programa de detección de objetos con YOLO y cámara OAK-D Lite
Detecta objetos y calcula distancias usando la información de profundidad
"""

import cv2
import numpy as np
import depthai as dai
from ultralytics import YOLO
import time
import pyttsx3
import threading
import speech_recognition as sr
import queue
import re
import os
import sys
import signal

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
        self.device = dai.Device(self.pipeline)
        print("Cámara OAK-D Lite conectada")
        
        # Obtener colas de salida (con timeout para evitar bloqueos)
        self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.q_depth = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        
        # Flag para controlar el loop
        self.running = True
        
        # Configurar manejo de señales para detener correctamente
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Obtener nombres de clases de YOLO
        self.class_names = self.model.names
        
        # Inicializar reconocimiento de voz (inicializar como None primero)
        self.recognizer = None
        self.microphone = None
        
        print("Inicializando reconocimiento de voz...")
        # Suprimir mensajes de ALSA durante la inicialización
        import contextlib
        
        # Suprimir ALSA completamente durante la inicialización
        old_stderr = sys.stderr
        try:
            sys.stderr = open(os.devnull, 'w')
            try:
                self.recognizer = sr.Recognizer()
                
                # Buscar el micrófono H390 específicamente
                mic_list = sr.Microphone.list_microphone_names()
                h390_index = None
                for i, mic_name in enumerate(mic_list):
                    if 'logi' in mic_name.lower() or 'h390' in mic_name.lower() or 'usb headset' in mic_name.lower():
                        h390_index = i
                        print(f"  Micrófono H390 encontrado: [{i}] {mic_name}")
                        break
                
                if h390_index is not None:
                    self.microphone = sr.Microphone(device_index=h390_index)
                    print(f"  ✓ Usando micrófono H390 (índice {h390_index})")
                else:
                    # Si no se encuentra, usar índice 0 que suele ser el H390
                    self.microphone = sr.Microphone(device_index=0)
                    print(f"  ⚠ H390 no encontrado por nombre, usando índice 0 (puede ser el H390)")
                
                # Ajustar para ruido ambiente
                print("Ajustando micrófono para ruido ambiente (2 segundos)...")
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=2)
                print("✓ Reconocimiento de voz inicializado")
            except Exception as e:
                print(f"⚠ Advertencia: Error inicializando micrófono: {e}")
                print("   El reconocimiento de voz puede no funcionar correctamente")
                self.recognizer = None
                self.microphone = None
                print("✓ Reconocimiento de voz inicializado (deshabilitado)")
        finally:
            sys.stderr.close()
            sys.stderr = old_stderr
        
        # Cola para comandos de voz
        self.voice_commands = queue.Queue()
        
        # Objetos detectados actualmente (para responder preguntas)
        self.current_objects = {}
        self.objects_lock = threading.Lock()
        
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
        
        # Inicializar síntesis de voz
        print("Inicializando síntesis de voz...")
        try:
            self.tts_engine = pyttsx3.init()
            
            # Configurar propiedades de voz
            voices = self.tts_engine.getProperty('voices')
            # Intentar usar voz en español si está disponible
            voz_encontrada = False
            for voice in voices:
                if 'spanish' in voice.name.lower() or 'español' in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    voz_encontrada = True
                    print(f"  Usando voz: {voice.name}")
                    break
            
            if not voz_encontrada and len(voices) > 0:
                print(f"  Usando voz por defecto: {voices[0].name}")
            
            # Configurar velocidad y volumen
            self.tts_engine.setProperty('rate', 150)  # Velocidad de habla
            self.tts_engine.setProperty('volume', 1.0)  # Volumen máximo (0.0 a 1.0)
            
            # Asegurar que el auricular esté configurado antes de probar
            try:
                import subprocess
                result = subprocess.run(['pactl', 'list', 'short', 'sinks'], 
                                      capture_output=True, text=True, timeout=1)
                for line in result.stdout.split('\n'):
                    if 'logi' in line.lower() or 'h390' in line.lower():
                        h390_sink = line.split()[1]
                        subprocess.run(['pactl', 'set-default-sink', h390_sink], 
                                     capture_output=True, timeout=1)
                        subprocess.run(['pactl', 'set-sink-volume', h390_sink, '100%'], 
                                     capture_output=True, timeout=1)
                        break
            except:
                pass
            
            # Probar que funciona
            print("  Probando síntesis de voz...")
            print("  (Deberías escuchar: 'Síntesis de voz funcionando')")
            self.tts_engine.say("Síntesis de voz funcionando")
            self.tts_engine.runAndWait()
            print("  ✓ Prueba completada")
            
            # Thread para hablar (evita bloquear la detección)
            self.tts_lock = threading.Lock()
            self.last_spoken = {}  # Para evitar repetir demasiado
            self.last_spoken_time = {}  # Timestamp de última vez que se habló
            
            print("✓ Síntesis de voz inicializada y funcionando")
        except Exception as e:
            print(f"✗ Error inicializando síntesis de voz: {e}")
            print("  El programa continuará pero no hablará los objetos")
            self.tts_engine = None
            self.tts_lock = threading.Lock()
            self.last_spoken = {}
            self.last_spoken_time = {}
    
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
        
        Args:
            depth_frame: Frame de profundidad
            x, y: Coordenadas del punto
            
        Returns:
            Distancia en metros
        """
        if depth_frame is None:
            return None
        
        # Asegurar que las coordenadas estén dentro de los límites
        x = int(np.clip(x, 0, depth_frame.shape[1] - 1))
        y = int(np.clip(y, 0, depth_frame.shape[0] - 1))
        
        # Obtener la distancia en milímetros y convertir a metros
        distance_mm = depth_frame[y, x]
        distance_m = distance_mm / 1000.0
        
        return distance_m
    
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
                distance = self.get_distance_at_point(depth_frame, center_x, center_y)
                
                # Almacenar información del objeto
                if distance is not None:
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
        
        print("🎤 Escuchando comandos de voz... (habla cuando veas el indicador)")
        while self.running:
            try:
                with suppress_stderr():
                    with self.microphone as source:
                        # Escuchar con timeout
                        try:
                            # Mostrar que está escuchando
                            print("🎤 Escuchando... (habla ahora)", end='\r', flush=True)
                            audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                            print("🎤 Procesando audio...                    ", end='\r', flush=True)
                        except sr.WaitTimeoutError:
                            continue
                        
                        # Reconocer el audio
                        try:
                            text = self.recognizer.recognize_google(audio, language='es-ES')
                            text = text.lower()
                            print(f"\n" + "="*60)
                            print(f"🎤 COMANDO DE VOZ DETECTADO: {text}")
                            print("="*60)
                            self.voice_commands.put(text)
                        except sr.UnknownValueError:
                            # Mostrar que está escuchando pero no entendió
                            print("🎤 Escuchando... (no se entendió)")
                        except sr.RequestError as e:
                            print(f"❌ Error en reconocimiento de voz: {e}")
                            print("   ¿Tienes conexión a internet? (Google Speech Recognition requiere internet)")
            except Exception as e:
                if self.running:
                    # No imprimir errores de ALSA
                    if "ALSA" not in str(e):
                        print(f"Error escuchando: {e}")
                time.sleep(0.1)
    
    def process_voice_command(self, command):
        """Procesa un comando de voz y responde"""
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
                    self.speak_text(respuesta)
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
                            self.speak_text(respuesta)
                            return True
                    
                    # No se encontró el objeto
                    respuesta = f"No, no estoy viendo {objeto_encontrado} en este momento"
                    print(f"📢 Respuesta: {respuesta}")
                    self.speak_text(respuesta)
                    return True
        
        return False
    
    def speak_text(self, text):
        """Habla un texto usando síntesis de voz"""
        def _speak():
            try:
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
                
                with self.tts_lock:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
            except Exception as e:
                print(f"Error al hablar: {e}")
        
        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()
    
    def run(self):
        """Ejecuta el loop principal de detección"""
        print("\n" + "="*60)
        print("DETECCIÓN DE OBJETOS INICIADA")
        print("="*60)
        print("\n📹 Ventana de video abierta")
        print("🔊 Audio: El programa hablará un resumen cada 10 segundos")
        print("   - Dice qué objetos ve y sus distancias")
        print("   - Indica si hay obstáculos próximos (menos de 2 metros)")
        print("   - Descansa 10 segundos entre cada mensaje")
        print("🎤 Micrófono: Puedes hacer preguntas por voz")
        print("   Ejemplos: '¿Estás viendo una silla?' o '¿A qué distancia está la persona?'")
        print("⌨️  CONTROLES:")
        print("   • Presiona 'Q', 'q' o ESC en la ventana de video para SALIR")
        print("   • O presiona Ctrl+C en esta terminal para SALIR")
        print("   • O cierra la ventana de video para SALIR")
        print("\n" + "="*60 + "\n")
        
        # Iniciar thread para escuchar comandos de voz (solo si el micrófono está disponible)
        if self.microphone is not None and self.recognizer is not None:
            voice_thread = threading.Thread(target=self.listen_for_commands, daemon=True)
            voice_thread.start()
            print("✓ Micrófono activado, puedes hacer preguntas ahora")
            print("  Ejemplos: '¿Estás viendo una silla?' o '¿A qué distancia está la persona?'")
            print("  Habla claramente cerca del micrófono\n")
        else:
            print("⚠ Micrófono no disponible, reconocimiento de voz deshabilitado")
            print("  El programa seguirá funcionando pero no podrás hacer preguntas por voz\n")
        
        frame_count = 0
        start_time = time.time()
        window_created = False
        
        try:
            while self.running:
                # Verificar tecla PRIMERO (más responsivo)
                if window_created:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q') or key == 27:  # 'q', 'Q' o ESC
                        print("\n" + "="*60)
                        print("SALIENDO... (Presionaste 'Q' o ESC)")
                        print("="*60)
                        self.running = False
                        break
                    
                    # Verificar si la ventana fue cerrada
                    try:
                        if cv2.getWindowProperty("Detección de Objetos - OAK-D Lite", cv2.WND_PROP_VISIBLE) < 1:
                            print("\n" + "="*60)
                            print("SALIENDO... (Ventana cerrada)")
                            print("="*60)
                            self.running = False
                            break
                    except:
                        self.running = False
                        break
                
                # Verificar si hay frames disponibles antes de obtenerlos
                if not self.q_rgb.has() or not self.q_depth.has():
                    # Verificar running antes de sleep
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
                    
                    # Si no hay frames disponibles, continuar
                    if in_rgb is None or in_depth is None:
                        continue
                except Exception as e:
                    if self.running:
                        print(f"Error obteniendo frames: {e}")
                    continue
                
                # Convertir a numpy arrays
                frame = in_rgb.getCvFrame()
                depth_frame = in_depth.getFrame()
                
                # Crear ventana si no existe
                if not window_created:
                    cv2.namedWindow("Detección de Objetos - OAK-D Lite", cv2.WINDOW_NORMAL)
                    window_created = True
                    print("✓ Ventana de video creada")
                
                # Verificar running antes de procesar
                if not self.running:
                    break
                
                # Ejecutar detección YOLO
                results = self.model(frame, verbose=False)
                
                # Verificar running después de YOLO (puede tardar)
                if not self.running:
                    break
                
                # Dibujar detecciones
                objetos = self.draw_detections(frame, results, depth_frame)
                
                # Actualizar objetos actuales para responder preguntas
                with self.objects_lock:
                    self.current_objects = {}
                    for obj in objetos:
                        self.current_objects[obj['nombre']] = {
                            'distancia': obj['distancia'],
                            'confianza': obj['confianza']
                        }
                
                # Mostrar información de objetos detectados
                if objetos:
                    print(f"\n--- Frame {frame_count} ---")
                    for obj in objetos:
                        print(f"  • {obj['nombre']}: {obj['distancia']:.2f}m (confianza: {obj['confianza']:.2%})")
                    
                    # Hablar resumen cada 10 segundos (con descanso)
                    if self.should_speak_summary():
                        resumen = self.generar_resumen_voz(objetos)
                        print(f"\n🔊 RESUMEN (cada 10 segundos): {resumen}")
                        self.speak_text(resumen)
                else:
                    # Si no hay objetos, también hablar cada 10 segundos
                    if self.should_speak_summary():
                        resumen = "No estoy viendo ningún objeto en este momento. No hay obstáculos próximos"
                        print(f"\n🔊 RESUMEN: {resumen}")
                        self.speak_text(resumen)
                
                # Procesar comandos de voz pendientes
                try:
                    while not self.voice_commands.empty():
                        command = self.voice_commands.get_nowait()
                        self.process_voice_command(command)
                except queue.Empty:
                    pass
                
                # Calcular y mostrar FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = 30 / elapsed
                    print(f"FPS: {fps:.2f}")
                    start_time = time.time()
                
                # Agregar instrucciones visibles en el frame
                h, w = frame.shape[:2]
                # Fondo semi-transparente para el texto
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (350, 80), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                # Texto de instrucciones
                cv2.putText(frame, "Presiona 'Q' o ESC para SALIR", (20, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "O Ctrl+C en terminal", (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                # Verificar running antes de mostrar
                if not self.running:
                    break
                
                # Mostrar frame
                cv2.imshow("Detección de Objetos - OAK-D Lite", frame)
                
                # waitKey ya se ejecutó al inicio del loop, no es necesario aquí
                    
        except KeyboardInterrupt:
            print("\n" + "="*60)
            print("SALIENDO... (Presionaste Ctrl+C)")
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

