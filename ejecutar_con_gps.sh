#!/bin/bash
# Script que activa el GPS y ejecuta el programa principal

echo "============================================================"
echo "ACTIVANDO GPS Y EJECUTANDO PROGRAMA"
echo "============================================================"

# Activar GPS
echo "1. Activando GPS..."
if command -v activar_gps &> /dev/null; then
    activar_gps
else
    echo "   Usando script local..."
    ./activar_gps_simple.sh || {
        echo "   ⚠ Activación automática falló, intentando con sudo..."
        echo "gustavo" | sudo -S bash -c 'pkill -9 brltty; modprobe -r ch34x; sleep 1; modprobe ch34x; echo "1a86 7523" > /sys/bus/usb-serial/drivers/ch34x/new_id; udevadm trigger; sleep 2'
    }
fi

# Esperar a que aparezca el puerto
echo "2. Verificando puerto GPS..."
sleep 2
if ls -1 /dev/ttyUSB* 2>/dev/null | grep -q .; then
    echo "   ✓ GPS activado: $(ls -1 /dev/ttyUSB* 2>/dev/null)"
else
    echo "   ⚠ GPS no detectado, pero continuando..."
fi

# Ejecutar programa
echo "3. Ejecutando programa principal..."
echo ""
python3 detector_objetos_oakd.py

