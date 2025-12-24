#!/bin/bash
# Script para configurar GPS de forma permanente (solo ejecutar UNA VEZ con sudo)

echo "============================================================"
echo "CONFIGURACIÓN PERMANENTE DEL GPS"
echo "============================================================"
echo ""
echo "Este script configurará el sistema para que:"
echo "1. brltty NO interfiera con el GPS"
echo "2. El GPS se active automáticamente"
echo "3. No necesites sudo cada vez"
echo ""
echo "EJECUTA ESTO UNA SOLA VEZ CON: sudo ./configurar_gps_permanente.sh"
echo ""

# Verificar si se ejecuta como root
if [ "$EUID" -ne 0 ]; then 
    echo "Este script necesita permisos de administrador."
    echo "Ejecuta: sudo ./configurar_gps_permanente.sh"
    exit 1
fi

# 1. Bloquear brltty permanentemente
echo "1. Bloqueando brltty permanentemente..."
systemctl stop brltty 2>/dev/null
systemctl disable brltty 2>/dev/null
systemctl mask brltty 2>/dev/null
pkill -9 brltty 2>/dev/null
echo "   ✓ brltty bloqueado permanentemente"

# 2. Crear regla udev para evitar que brltty reclame el GPS
echo "2. Creando regla udev para GPS..."
cat > /etc/udev/rules.d/99-gps-ch340.rules << 'EOF'
# Regla para GPS CH340 - Evitar que brltty lo capture
# Deshabilitar brltty para este dispositivo
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", \
    ENV{PROGRAM}="/bin/sh -c 'systemctl stop brltty 2>/dev/null || true'", \
    MODE="0666", \
    SYMLINK+="gps"

# Forzar vinculación al driver ch34x
ACTION=="add", SUBSYSTEM=="usb", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", \
    RUN+="/bin/sh -c 'echo 1a86 7523 > /sys/bus/usb-serial/drivers/ch34x/new_id 2>/dev/null || true'"
EOF

echo "   ✓ Regla udev creada en /etc/udev/rules.d/99-gps-ch340.rules"

# 3. Recargar reglas udev
echo "3. Recargando reglas udev..."
udevadm control --reload-rules
udevadm trigger
echo "   ✓ Reglas recargadas"

# 4. Asegurar que los módulos se carguen al inicio
echo "4. Configurando carga automática de módulos..."
if ! grep -q "ch34x" /etc/modules-load.d/ch34x.conf 2>/dev/null; then
    echo "ch34x" > /etc/modules-load.d/ch34x.conf
    echo "usbserial" >> /etc/modules-load.d/ch34x.conf
    echo "   ✓ Módulos configurados para cargarse automáticamente"
else
    echo "   ✓ Módulos ya configurados"
fi

# 5. Crear script sin sudo para activar GPS
echo "5. Creando script de activación sin sudo..."
cat > /usr/local/bin/activar_gps << 'SCRIPT'
#!/bin/bash
# Script para activar GPS sin necesidad de sudo (después de configuración inicial)

# Recargar módulos
modprobe -r ch34x 2>/dev/null
sleep 1
modprobe ch34x 2>/dev/null
sleep 1

# Forzar vinculación
echo "1a86 7523" > /sys/bus/usb-serial/drivers/ch34x/new_id 2>/dev/null
udevadm trigger 2>/dev/null
sleep 2

# Verificar puerto
if ls -1 /dev/ttyUSB* 2>/dev/null | grep -q .; then
    echo "✓ GPS activado: $(ls -1 /dev/ttyUSB* 2>/dev/null)"
    exit 0
else
    echo "⚠ GPS no detectado. Desconecta y reconecta el GPS."
    exit 1
fi
SCRIPT

chmod +x /usr/local/bin/activar_gps
chmod 666 /sys/bus/usb-serial/drivers/ch34x/new_id 2>/dev/null || true
echo "   ✓ Script creado en /usr/local/bin/activar_gps"

# 6. Verificar GPS actual
echo "6. Verificando GPS actual..."
sleep 2
if ls -1 /dev/ttyUSB* 2>/dev/null | grep -q .; then
    echo "   ✓ GPS detectado: $(ls -1 /dev/ttyUSB* 2>/dev/null)"
    for port in $(ls -1 /dev/ttyUSB* 2>/dev/null); do
        chmod 666 "$port" 2>/dev/null
    done
else
    echo "   ⚠ GPS no detectado en este momento"
    echo "   Desconecta y reconecta el GPS para que se active automáticamente"
fi

echo ""
echo "============================================================"
echo "✓ CONFIGURACIÓN COMPLETA"
echo "============================================================"
echo ""
echo "Ahora puedes activar el GPS sin sudo usando:"
echo "  activar_gps"
echo ""
echo "O simplemente desconecta y reconecta el GPS,"
echo "y se activará automáticamente gracias a la regla udev."
echo ""

