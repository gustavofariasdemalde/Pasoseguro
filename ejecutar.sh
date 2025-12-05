#!/bin/bash
# Script para ejecutar el detector de objetos suprimiendo mensajes de ALSA

# Redirigir TODOS los mensajes de ALSA a /dev/null
exec 2> >(grep -v -E "(ALSA|pcm|snd_|Unknown PCM|Unable to find)" >&2)

# Ejecutar el programa
python3 detector_objetos_oakd.py 2>/dev/null

