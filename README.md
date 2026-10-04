# Cámara Belleza

Cámara virtual con alisado de piel y labios pintados (Windows + OBS Virtual Camera).

## Uso
1. Abre OBS una vez, pulsa "Iniciar cámara virtual" y ciérralo (solo hace falta la primera vez para registrar el driver).
2. Doble clic en `iniciar.bat`. Se abre un panel con la vista previa y los controles.
3. En Zoom/Meet/Teams elige **OBS Virtual Camera** como cámara.

## Panel de control
- **Cámara**: desplegable con los nombres de tus cámaras (botón "Actualizar lista" si conectas una nueva).
- **Resolución**: 1280×720, 960×540 (recomendada) o 640×480. Cambiarla reinicia la cámara virtual un instante.
- **Filtros de belleza activados**: interruptor de alisado y labial (apagado = sin esos filtros).
- **Alisado de piel** e **Intensidad del labial**: deslizadores (labial a 0 = sin labial).
- **Color del labial**: botones de colores o "Otro…" para elegir cualquiera.
- **Intensidad del rubor** y **Color del rubor**: dos manchas suaves en las mejillas, que siguen el giro
  y la inclinación de la cabeza y no pisan ojos ni boca. Arranca en 0 (sin rubor). El color se mezcla por
  multiplicación con tu piel, así que conserva la textura y el resultado depende de tu tono de piel;
  sube la intensidad poco a poco, a partir de ~30-40 ya se nota. Botones de colores o "Otro…".
- **Imagen** (bajo la vista previa), de -100 a +100 cada uno (0 = sin cambio):
  - **Exposición**: ±2 pasos de luz (EV), calculada en luz lineal como en una cámara.
  - **Sombras**: aclara (+) u oscurece (-) solo los tonos oscuros; el negro puro no se mueve.
  - **Luces**: aclara (+) o recupera (-) solo los tonos claros; el blanco puro no se mueve.
  - **Brillo**, **Contraste** y **Saturación**.

  Funcionan aunque los filtros de belleza estén apagados. "Restablecer" los devuelve a 0.
- **Espejar vista previa**: solo afecta a esta ventana, no a lo que ven en la reunión.

Los ajustes se guardan en `config.json` y se recuerdan la próxima vez.
`iniciar.bat --list` muestra las cámaras detectadas.

## Si va a pocos fps
La fluidez depende sobre todo de la webcam. Prueba otra resolución, más luz en la habitación
(con poca luz muchas webcams bajan los fps) o conectarla directamente a un puerto USB del equipo, sin hub.

## Archivos
- `camara_belleza.py`: programa principal
- `face_landmarker.task`: modelo de MediaPipe (de storage.googleapis.com/mediapipe-models)
- `requirements.txt`: dependencias de Python
