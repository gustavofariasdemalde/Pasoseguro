#!/bin/bash
# Script para configurar el auricular H390 como dispositivo de salida por defecto

echo "============================================================"
echo "CONFIGURACIÓN DEL AURICULAR H390 PARA AUDIO"
echo "============================================================"
echo

# Buscar el auricular H390
H390_SINK=$(pactl list short sinks | grep -i logi | head -1 | cut -f2)

if [ -z "$H390_SINK" ]; then
    echo "❌ No se encontró el auricular H390"
    echo "   Dispositivos de salida disponibles:"
    pactl list short sinks
    exit 1
fi

echo "✓ Auricular H390 encontrado: $H390_SINK"
echo

# Configurar como dispositivo por defecto
echo "1. Configurando auricular H390 como dispositivo de salida por defecto..."
pactl set-default-sink "$H390_SINK" && echo "   ✓ Configurado como dispositivo por defecto" || echo "   ✗ Error al configurar"

echo
echo "2. Aumentando volumen al 100%..."
pactl set-sink-volume "$H390_SINK" 100% && echo "   ✓ Volumen configurado al 100%" || echo "   ✗ Error al configurar volumen"

echo
echo "3. Desactivando silencio..."
pactl set-sink-mute "$H390_SINK" 0 && echo "   ✓ Silencio desactivado" || echo "   ✗ Error al desactivar silencio"

echo
echo "4. Verificando configuración actual..."
CURRENT_SINK=$(pactl info | grep "Default Sink" | cut -d: -f2 | xargs)
echo "   Dispositivo de salida actual: $CURRENT_SINK"

if [[ "$CURRENT_SINK" == *"Logi"* ]] || [[ "$CURRENT_SINK" == *"H390"* ]]; then
    echo "   ✅ Auricular H390 está configurado correctamente"
else
    echo "   ⚠ El auricular H390 NO está configurado como dispositivo por defecto"
fi

echo
echo "============================================================"
echo "CONFIGURACIÓN COMPLETADA"
echo "============================================================"
echo
echo "Ahora el programa debería usar el auricular H390 para el audio"

