#!/usr/bin/env bash
# Quita brltty (suele desconectar el CH340 del GPS en Jetson/Ubuntu) y recarga udev.
# Uso: ./fix_gps_brltty.sh

set -e
cd "$(dirname "$0")"

echo "== 1) Quitar paquete brltty (si está instalado) =="
sudo apt-get -y purge brltty brltty-x11 2>/dev/null || true

echo "== 2) Matar proceso brltty si sigue vivo =="
sudo killall brltty 2>/dev/null || true

echo "== 3) Regla udev extra para CH340 (opcional) =="
RULE_SRC="udev/99-brltty-no-ch340.rules"
if [[ -f "$RULE_SRC" ]]; then
  sudo cp -v "$RULE_SRC" /etc/udev/rules.d/99-brltty-no-ch340.rules
else
  echo "    (omitido: no está $RULE_SRC)"
fi

echo "== 4) Recargar udev =="
sudo udevadm control --reload-rules
sudo udevadm trigger

echo ""
echo "Listo. Ahora:"
echo "  - Desenchufá el USB del GPS, esperá 5 s, enchufalo."
echo "  - Comprobá: ls -la /dev/ttyUSB0"
echo "  - En dmesg NO debe aparecer 'brltty' tras 'attached to ttyUSB0':"
echo "      sudo dmesg | tail -15"
echo ""
