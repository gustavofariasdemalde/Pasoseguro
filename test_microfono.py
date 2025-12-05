#!/usr/bin/env python3
"""
Script para probar el micrófono del auricular H390
"""

import speech_recognition as sr
import sys
import os

# Suprimir mensajes de ALSA
old_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

print("="*60)
print("PRUEBA DE MICRÓFONO H390")
print("="*60)
print()

try:
    # Inicializar reconocimiento de voz
    print("1. Inicializando reconocimiento de voz...")
    recognizer = sr.Recognizer()
    
    # Buscar micrófono H390 específicamente
    mic_list = sr.Microphone.list_microphone_names()
    h390_index = None
    for i, mic_name in enumerate(mic_list):
        if 'logi' in mic_name.lower() or 'h390' in mic_name.lower() or 'usb headset' in mic_name.lower():
            h390_index = i
            break
    
    if h390_index is not None:
        microphone = sr.Microphone(device_index=h390_index)
        print(f"   ✓ Usando micrófono H390 (índice {h390_index})")
    else:
        microphone = sr.Microphone(device_index=0)
        print(f"   ⚠ Usando índice 0 (puede ser el H390)")
    
    print("   ✓ Reconocimiento inicializado")
    
    # Listar micrófonos disponibles
    print("\n2. Micrófonos disponibles:")
    mic_list = sr.Microphone.list_microphone_names()
    for i, mic_name in enumerate(mic_list):
        marker = " ← ACTUAL" if i == microphone.device_index else ""
        print(f"   [{i}] {mic_name}{marker}")
    
    # Ajustar para ruido ambiente
    print("\n3. Ajustando micrófono para ruido ambiente (3 segundos)...")
    print("   (Mantén silencio durante este tiempo)")
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=3)
    print("   ✓ Ajuste completado")
    
    # Probar reconocimiento
    print("\n4. PRUEBA DE RECONOCIMIENTO")
    print("   Habla algo ahora (tienes 5 segundos)...")
    print("   (Por ejemplo: 'Hola, esto es una prueba')")
    print()
    
    with microphone as source:
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            print("   ✓ Audio capturado, procesando...")
            
            # Intentar reconocer
            try:
                text = recognizer.recognize_google(audio, language='es-ES')
                print(f"\n   🎤 TEXTO RECONOCIDO: '{text}'")
                print("\n   ✅ ¡MICRÓFONO FUNCIONANDO CORRECTAMENTE!")
            except sr.UnknownValueError:
                print("\n   ⚠ No se pudo entender el audio")
                print("   Posibles causas:")
                print("   - Hablaste muy bajo")
                print("   - Había mucho ruido de fondo")
                print("   - El micrófono no captó bien el audio")
            except sr.RequestError as e:
                print(f"\n   ❌ Error en reconocimiento: {e}")
                print("   ¿Tienes conexión a internet?")
        except sr.WaitTimeoutError:
            print("\n   ⚠ No se detectó audio en 5 segundos")
            print("   Posibles causas:")
            print("   - El micrófono no está captando audio")
            print("   - El micrófono no está seleccionado como dispositivo por defecto")
            print("   - El volumen del micrófono está muy bajo")
    
    # Probar nivel de audio
    print("\n5. PRUEBA DE NIVEL DE AUDIO")
    print("   Habla ahora (se medirá el nivel de audio)...")
    with microphone as source:
        try:
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=3)
            # Calcular energía del audio
            import audioop
            energy = audioop.rms(audio.get_raw_data(), 2)
            print(f"   Nivel de audio detectado: {energy}")
            if energy > 1000:
                print("   ✅ Nivel de audio BUENO")
            elif energy > 500:
                print("   ⚠ Nivel de audio BAJO (habla más fuerte)")
            else:
                print("   ❌ Nivel de audio MUY BAJO (micrófono puede no estar funcionando)")
        except sr.WaitTimeoutError:
            print("   ❌ No se detectó audio")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

finally:
    sys.stderr.close()
    sys.stderr = old_stderr

print("\n" + "="*60)
print("PRUEBA COMPLETADA")
print("="*60)

