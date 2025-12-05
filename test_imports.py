#!/usr/bin/env python3
"""
Script de prueba para verificar que todas las dependencias están instaladas correctamente
"""

print("="*60)
print("VERIFICANDO DEPENDENCIAS...")
print("="*60)

errores = []

# Verificar OpenCV
try:
    import cv2
    print("✓ OpenCV:", cv2.__version__)
except ImportError as e:
    print("✗ OpenCV NO encontrado:", e)
    errores.append("opencv-python")

# Verificar NumPy
try:
    import numpy as np
    print("✓ NumPy:", np.__version__)
except ImportError as e:
    print("✗ NumPy NO encontrado:", e)
    errores.append("numpy")

# Verificar DepthAI
try:
    import depthai as dai
    print("✓ DepthAI:", dai.__version__)
except ImportError as e:
    print("✗ DepthAI NO encontrado:", e)
    errores.append("depthai")

# Verificar YOLO
try:
    from ultralytics import YOLO
    print("✓ Ultralytics YOLO: OK")
except ImportError as e:
    print("✗ Ultralytics YOLO NO encontrado:", e)
    errores.append("ultralytics")

# Verificar síntesis de voz
try:
    import pyttsx3
    engine = pyttsx3.init()
    print("✓ pyttsx3: OK")
except ImportError as e:
    print("✗ pyttsx3 NO encontrado:", e)
    errores.append("pyttsx3")

# Verificar reconocimiento de voz
try:
    import speech_recognition as sr
    r = sr.Recognizer()
    print("✓ SpeechRecognition: OK")
except ImportError as e:
    print("✗ SpeechRecognition NO encontrado:", e)
    errores.append("SpeechRecognition")

# Verificar PyAudio
try:
    import pyaudio
    print("✓ PyAudio: OK")
except ImportError as e:
    print("✗ PyAudio NO encontrado:", e)
    errores.append("pyaudio")

print("\n" + "="*60)
if errores:
    print("❌ ERRORES ENCONTRADOS:")
    print("   Faltan las siguientes dependencias:", ", ".join(errores))
    print("\n   Instala con: pip install " + " ".join(errores))
else:
    print("✅ TODAS LAS DEPENDENCIAS ESTÁN INSTALADAS CORRECTAMENTE")
    print("\n   El programa debería funcionar cuando ejecutes:")
    print("   python3 detector_objetos_oakd.py")
print("="*60)

