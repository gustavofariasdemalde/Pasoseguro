#!/bin/bash
# Script para configurar el micrófono H390

echo "============================================================"
echo "CONFIGURACIÓN DEL MICRÓFONO H390"
echo "============================================================"
echo

# Verificar si está conectado
echo "1. Verificando dispositivos de audio USB..."
lsusb | grep -i logitech || echo "   ⚠ No se encontró dispositivo Logitech en USB"

echo
echo "2. Dispositivos de audio disponibles:"
arecord -l 2>/dev/null | grep -i logi || echo "   ⚠ No se encontró micrófono Logitech"

echo
echo "3. Configuración de PulseAudio:"
pactl list short sources | grep -i logi || echo "   ⚠ No se encontró fuente de audio Logitech"

echo
echo "4. Configurando micrófono H390 como dispositivo por defecto..."
# Intentar establecer el micrófono H390 como por defecto
pactl set-default-source $(pactl list short sources | grep -i logi | head -1 | cut -f2) 2>/dev/null && echo "   ✓ Micrófono H390 configurado como por defecto" || echo "   ⚠ No se pudo configurar automáticamente"

echo
echo "5. Aumentando volumen del micrófono..."
# Aumentar volumen al 100%
pactl set-source-volume $(pactl list short sources | grep -i logi | head -1 | cut -f2) 100% 2>/dev/null && echo "   ✓ Volumen configurado al 100%" || echo "   ⚠ No se pudo ajustar el volumen"

echo
echo "6. Desactivando silencio del micrófono..."
# Quitar silencio
pactl set-source-mute $(pactl list short sources | grep -i logi | head -1 | cut -f2) 0 2>/dev/null && echo "   ✓ Silencio desactivado" || echo "   ⚠ No se pudo desactivar silencio"

echo
echo "============================================================"
echo "CONFIGURACIÓN COMPLETADA"
echo "============================================================"
echo
echo "Ahora ejecuta: python3 test_microfono.py"
echo "para probar si el micrófono funciona"

