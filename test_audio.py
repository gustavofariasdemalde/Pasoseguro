#!/usr/bin/env python3
"""
Script para probar que el audio funcione con el auricular H390
"""

import pyttsx3
import subprocess
import sys
import os

print("="*60)
print("PRUEBA DE AUDIO CON AURICULAR H390")
print("="*60)
print()

# Configurar auricular H390 como dispositivo por defecto
print("1. Configurando auricular H390...")
try:
    result = subprocess.run(['pactl', 'list', 'short', 'sinks'], 
                          capture_output=True, text=True, timeout=2)
    h390_sink = None
    for line in result.stdout.split('\n'):
        if 'logi' in line.lower() or 'h390' in line.lower():
            h390_sink = line.split()[1]
            break
    
    if h390_sink:
        subprocess.run(['pactl', 'set-default-sink', h390_sink], 
                     capture_output=True, timeout=2)
        subprocess.run(['pactl', 'set-sink-volume', h390_sink, '100%'], 
                     capture_output=True, timeout=2)
        subprocess.run(['pactl', 'set-sink-mute', h390_sink, '0'], 
                     capture_output=True, timeout=2)
        print(f"   ✓ Auricular H390 configurado: {h390_sink}")
    else:
        print("   ⚠ Auricular H390 no encontrado")
except Exception as e:
    print(f"   ⚠ Error: {e}")

print()
print("2. Inicializando síntesis de voz...")
try:
    engine = pyttsx3.init()
    print("   ✓ Motor de síntesis de voz inicializado")
    
    # Configurar propiedades
    voices = engine.getProperty('voices')
    if len(voices) > 0:
        print(f"   Voz disponible: {voices[0].name}")
    
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)  # Volumen máximo
    
    print()
    print("3. PROBANDO AUDIO...")
    print("   Deberías escuchar: 'Prueba de audio uno dos tres'")
    print()
    
    engine.say("Prueba de audio uno dos tres")
    engine.runAndWait()
    
    print()
    print("   ¿Escuchaste el audio por el auricular?")
    print("   Si NO escuchaste, el problema puede ser:")
    print("   - El auricular no está configurado como dispositivo por defecto")
    print("   - El volumen del sistema está muy bajo")
    print("   - El auricular está silenciado")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*60)
print("PRUEBA COMPLETADA")
print("="*60)

