#!/bin/bash
# Ejecutar detector por SSH (sin monitor). Usa pantalla virtual Xvfb si no hay DISPLAY.

# Si no hay pantalla (conexión SSH), arrancar pantalla virtual para que OpenCV no aborte
if [ -z "$DISPLAY" ]; then
    if ! command -v Xvfb > /dev/null 2>&1; then
        echo "Para ejecutar por SSH sin monitor hace falta Xvfb."
        echo "En la Jetson ejecuta: sudo apt install xvfb"
        exit 1
    fi
    echo "Sin monitor: usando pantalla virtual (Xvfb) para ejecutar por SSH."
    if ! pgrep -x Xvfb > /dev/null 2>&1; then
        Xvfb :99 -screen 0 1280x720x24 &
        sleep 2
    fi
    export DISPLAY=:99
fi

# Redirigir mensajes de ALSA para no llenar la terminal (opcional)
exec 2> >(grep -v -E "(ALSA|pcm|snd_|Unknown PCM|Unable to find)" >&2)

# Ejecutar el programa
python3 detector_objetos_oakd.py 2>/dev/null
