#!/bin/bash
# Script simple para activar GPS (sin sudo después de configuración inicial)

echo "Activando GPS..."

# Recargar módulos (puede fallar sin sudo, pero lo intentamos)
modprobe -r ch34x 2>/dev/null
sleep 1
modprobe ch34x 2>/dev/null
sleep 1

# Intentar forzar vinculación (puede fallar sin sudo)
echo "1a86 7523" > /sys/bus/usb-serial/drivers/ch34x/new_id 2>/dev/null || {
    echo "⚠ Necesitas ejecutar la configuración inicial primero:"
    echo "   sudo ./configurar_gps_permanente.sh"
    exit 1
}

# Forzar detección
udevadm trigger 2>/dev/null
sleep 2

# Verificar puerto
PORTS=$(ls -1 /dev/ttyUSB* 2>/dev/null)
if [ -n "$PORTS" ]; then
    echo "✓ GPS activado: $PORTS"
    exit 0
else
    echo "⚠ GPS no detectado."
    echo "   Desconecta y reconecta el GPS, o ejecuta:"
    echo "   sudo ./configurar_gps_permanente.sh"
    exit 1
fi

