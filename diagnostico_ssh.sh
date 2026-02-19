#!/bin/bash
# Script de diagnóstico SSH

echo "============================================================"
echo "DIAGNÓSTICO DE CONEXIÓN SSH"
echo "============================================================"
echo ""

echo "1. IPs de la Jetson:"
hostname -I
echo ""

echo "2. Estado de SSH:"
systemctl is-active ssh && echo "✓ SSH está ACTIVO" || echo "✗ SSH está INACTIVO"
echo ""

echo "3. SSH escuchando en puerto 22:"
ss -tlnp | grep :22 && echo "✓ SSH escuchando correctamente" || echo "✗ SSH NO está escuchando"
echo ""

echo "4. Interfaces de red activas:"
ip -4 addr show | grep -E "inet |^[0-9]+:" | grep -B1 "inet " | grep -E "^[0-9]+:|inet "
echo ""

echo "5. Gateway (router):"
ip route | grep default | head -1
echo ""

echo "6. Probando conectividad local:"
ping -c 1 127.0.0.1 > /dev/null 2>&1 && echo "✓ Localhost responde" || echo "✗ Problema local"
echo ""

echo "============================================================"
echo "INFORMACIÓN PARA CONEXIÓN:"
echo "============================================================"
echo ""
echo "IP WiFi:  192.168.0.150 (red: FT FARIAS)"
echo "IP Cable: 192.168.0.100"
echo ""
echo "Para conectarte desde tu laptop:"
echo "  ssh gustavo@192.168.0.150  (si ambos están en WiFi FT FARIAS)"
echo "  ssh gustavo@192.168.0.100  (si ambos están por cable)"
echo ""
echo "IMPORTANTE: Asegúrate de que tu laptop esté en la MISMA RED:"
echo "  - Si Jetson está en WiFi 'FT FARIAS', laptop debe estar en la misma WiFi"
echo "  - Si Jetson está en cable, laptop debe estar en el mismo router/switche"
echo ""

