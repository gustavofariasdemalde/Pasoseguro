# Detector de Objetos con YOLO y OAK-D Lite

Este programa detecta objetos en tiempo real usando YOLO (Ultralytics) y calcula sus distancias usando la cámara OAK-D Lite.

## Características

- ✅ Detección de objetos en tiempo real con YOLO v8
- ✅ Cálculo de distancias usando la información de profundidad de OAK-D Lite
- ✅ Visualización de objetos detectados con bounding boxes
- ✅ Información de distancia y confianza para cada objeto
- ✅ **Síntesis de voz**: El programa habla los objetos detectados por el auricular USB
- ✅ **Reconocimiento de voz**: Puedes hacer preguntas por voz usando el micrófono del auricular
- ✅ **Interacción por voz**: Pregunta "¿Estás viendo una silla?" y el programa responderá con la distancia
- ✅ **GPS**: Lee ubicación GPS desde módulo USB y responde cuando preguntas "¿Dónde estoy?"
- ✅ Soporte para diferentes modelos YOLO (nano, small, medium, large, xlarge)

## Requisitos

- Python 3.8 o superior
- Cámara OAK-D Lite conectada al sistema
- Sistema operativo Linux (probado en Ubuntu)

## Instalación

1. Instala las dependencias de Python:

```bash
pip install -r requirements.txt
```

2. Instala el motor de síntesis de voz (espeak-ng):

```bash
sudo apt-get update
sudo apt-get install -y espeak-ng espeak-ng-data
```

3. Asegúrate de que la cámara OAK-D Lite esté conectada y reconocida por el sistema.

4. Conecta tu auricular USB con micrófono (el programa usará el dispositivo de audio por defecto del sistema).

5. (Opcional) Conecta un módulo GPS por USB para obtener tu ubicación. Si usas GPS, ejecuta `./activar_gps.sh` antes de iniciar el programa.

## Uso

Ejecuta el programa:

```bash
python detector_objetos_oakd.py
```

### Controles

- **'Q' o 'q' o ESC**: Salir del programa (en la ventana de video)
- **Comando de voz "apagar"**: Di "apagar", "cerrar", "salir", "detener", "parar" o "terminar" para apagar el programa
- **Tecla 'Q' o ESC**: Salir del programa (en la ventana de video)
- El programa mostrará en la consola los objetos detectados con sus distancias
- **El programa hablará automáticamente** los objetos detectados por el auricular USB

### Interacción por Voz

Puedes hacer preguntas por voz usando el micrófono del auricular. El programa reconocerá tus preguntas y responderá con la distancia del objeto.

**Ejemplos de preguntas que puedes hacer:**

**Sobre objetos detectados:**
- "¿Estás viendo una silla?"
- "¿A qué distancia está la persona?"
- "¿Ves una mesa?"
- "¿Hay un auto?"
- "¿A qué distancia está el perro?"

**Sobre ubicación GPS:**
- "¿Dónde estoy?"
- "¿Cuál es mi ubicación?"
- "¿Qué es mi ubicación?"
- "Dime dónde estoy"
- "Coordenadas"
- "GPS"

**Para apagar el programa:**
- "Apagar"
- "Cerrar"
- "Salir"
- "Detener"
- "Parar"
- "Terminar"

El programa responderá con frases como:
- "Sí, estoy viendo silla a 2.5 metros"
- "Sí, estoy viendo persona a 1.2 metros"
- "No, no estoy viendo [objeto] en este momento"
- "Estoy en Avenida Principal 123, Barrio Centro, Ciudad. Coordenadas: 34.0522 grados Norte, 118.2437 grados Oeste"
- "Mi ubicación es: 34.0522 grados Norte, 118.2437 grados Oeste" (si no se puede obtener la dirección)
- "No tengo señal GPS en este momento. Asegúrate de estar al aire libre"

## Modelos YOLO Disponibles

Puedes cambiar el modelo en el código modificando la línea:

```python
detector = OAKDObjectDetector(model_name='yolov8n.pt')
```

Modelos disponibles:
- `yolov8n.pt` - Nano (rápido, menos preciso) - **Recomendado para inicio**
- `yolov8s.pt` - Small (balanceado)
- `yolov8m.pt` - Medium (más preciso)
- `yolov8l.pt` - Large (muy preciso)
- `yolov8x.pt` - XLarge (máxima precisión)

## Salida

El programa muestra:
- Ventana con video en tiempo real y bounding boxes de objetos detectados
- En la consola: lista de objetos detectados con sus distancias y niveles de confianza
- **Por el auricular USB**: El programa habla los objetos detectados (ej: "persona a 2.5 metros")
- FPS aproximado cada 30 frames

**Nota sobre la síntesis de voz:**
- El programa evita repetir el mismo objeto muy seguido
- Solo habla cuando detecta un objeto nuevo, cambia la distancia significativamente, o pasan más de 3 segundos
- Los nombres de objetos se traducen al español automáticamente

## Notas

- La primera ejecución descargará el modelo YOLO automáticamente
- El cálculo de distancia se realiza en el centro del bounding box del objeto
- La distancia se muestra en metros

## Solución de Problemas

### La cámara no se detecta
- Verifica que la cámara OAK-D Lite esté conectada correctamente
- Asegúrate de tener permisos para acceder a dispositivos USB
- Prueba ejecutar con `sudo` si es necesario

### Bajo rendimiento (FPS bajo)
- Usa el modelo `yolov8n.pt` (nano) para mejor rendimiento
- Reduce la resolución en el código si es necesario

### Errores de importación
- Asegúrate de tener todas las dependencias instaladas: `pip install -r requirements.txt`
- Verifica que estés usando Python 3.8 o superior

### La síntesis de voz no funciona
- Asegúrate de tener espeak-ng instalado: `sudo apt-get install -y espeak-ng espeak-ng-data`
- Verifica que tu auricular USB esté conectado y configurado como dispositivo de audio por defecto
- El programa usará el dispositivo de audio por defecto del sistema

### El reconocimiento de voz no funciona
- Asegúrate de tener conexión a internet (el reconocimiento usa Google Speech Recognition)
- Verifica que el micrófono del auricular esté funcionando
- Habla claramente y cerca del micrófono
- El programa ajusta automáticamente el micrófono para el ruido ambiente al iniciar

### El GPS no funciona
- **Opción 1 (Recomendada)**: Ejecuta `./ejecutar_con_gps.sh` - este script activa el GPS y ejecuta el programa automáticamente
- **Opción 2**: Ejecuta `activar_gps` (sin sudo) antes de iniciar el programa
- **Opción 3**: Si es la primera vez, ejecuta `sudo ./configurar_gps_permanente.sh` (solo una vez)
- Asegúrate de estar al aire libre para recibir señal de satélites
- Si el GPS no aparece, desconéctalo y vuelve a conectarlo, luego ejecuta `activar_gps` de nuevo
- El GPS necesita internet para obtener direcciones completas (calle y número), pero las coordenadas funcionan sin internet

