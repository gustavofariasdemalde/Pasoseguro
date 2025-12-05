#!/bin/bash
# Script de emergencia para detener el programa si se queda bloqueado

echo "Deteniendo detector_objetos_oakd.py..."

# Buscar y matar el proceso
pkill -f "detector_objetos_oakd.py"

# Esperar un momento
sleep 1

# Si aún está corriendo, forzar kill
pkill -9 -f "detector_objetos_oakd.py"

# Cerrar todas las ventanas de OpenCV que puedan quedar abiertas
pkill -f "opencv"

echo "✓ Programa detenido"


