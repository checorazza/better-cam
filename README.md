# Better Cam

Cámara virtual con filtros de belleza: alisado de piel, labial, rubor, ajustes de imagen y bucle de
video (Windows + OBS Virtual Camera).

## Uso
1. Abre OBS una vez, pulsa "Iniciar cámara virtual" y ciérralo (solo hace falta la primera vez para registrar el driver).
2. Doble clic en `iniciar.bat`. Se abre un panel con la vista previa y los controles.
3. En Zoom/Meet/Teams elige **OBS Virtual Camera** como cámara.

## Panel de control
Arriba a la derecha, **Filtros de belleza activados**: interruptor de todo lo de las pestañas Piel y
Maquillaje (apagado = sin esos filtros). Los controles están en cuatro pestañas:

### Presets (arriba del panel)
Guardan tu *look* con un nombre para volver a él cuando quieras.
- Un preset incluye: alisado, labial (intensidad y color), rubor (intensidad, posición y color) y los 7
  ajustes de imagen (exposición, temperatura, sombras, luces, brillo, contraste y saturación).
  **No** incluye la cámara, la resolución, el espejo de la vista previa, el interruptor general ni el bucle.
- **Guardar como…**: guarda el look actual con el nombre que escribas (hasta 40 letras). Si ya existe uno
  con ese nombre (sin distinguir mayúsculas), pregunta antes de reemplazarlo.
- Elegir un preset de la lista lo aplica al instante, moviendo todos los controles.
- Si tocas algo después de aplicarlo, aparece **● Modificado** y se activa **Actualizar** para guardar
  los cambios en ese mismo preset. Si no, simplemente elige otro: los cambios sin guardar se pierden.
- **Borrar** quita el preset de la lista; tus ajustes actuales no cambian.
- Se guardan en `presets.json` (en la carpeta del programa, ignorado por git). Si el archivo se
  corrompe, se aparta como `presets.json.bak` y se empieza vacío, sin perder el original.
- El programa recuerda qué preset estaba elegido al volver a abrirlo.

### Pestaña Cámara
- **Cámara**: desplegable con los nombres de tus cámaras (botón "Actualizar lista" si conectas una nueva).
- **Resolución**: 1280×720, 960×540 (recomendada) o 640×480. Cambiarla reinicia la cámara virtual un instante.
- **Espejar vista previa**: solo afecta a esta ventana, no a lo que ven en la reunión.

### Pestaña Piel
- **Alisado de piel**: deslizador. Cubre la cara y la frente hasta donde nace el pelo; el pelo, los
  ojos, las cejas y la boca no se alisan. Si tu frente queda corta o se pasa, ajusta `FOREHEAD_EXTRA`
  en `camara_belleza.py` (0,30 = 30 % de la altura de la cara por encima del óvalo de MediaPipe; con
  pelo muy claro, parecido a la piel, bájalo).

### Pestaña Maquillaje
- **Intensidad del labial** y **Color del labial**: botones de colores u "Otro…" (labial a 0 = sin labial).
- **Intensidad del rubor**, **Posición del rubor** y **Color del rubor**: manchas suaves en las mejillas,
  que siguen el giro y la inclinación de la cabeza y no pisan ojos ni boca. Arranca en 0 (sin rubor).
  El color se mezcla por multiplicación con tu piel, así que conserva la textura y el resultado depende
  de tu tono de piel; sube la intensidad poco a poco, a partir de ~30-40 ya se nota.
  Posiciones (según la guía de colocación por tipo de cara):
  1. Diagonal clásico · cara ovalada (por defecto)
  2. Sobre las manzanas · cara redonda
  3. Horizontal · cara alargada
  4. Alto a las sienes · cara corazón
  5. Redondo suave · cara cuadrada
  6. Manzanas hacia fuera · diamante
  7. Mejillas internas · cara ancha (efecto lifting)
  8. Efecto sol · mejillas y nariz
  9. Muñeca · manzanas altas (juvenil)
  10. Elegante · levantado a las sienes
  11. Natural · mínimo sobre manzanas (más suave que el resto)

  Para añadir o retocar posiciones, edita `BLUSH_STYLES` en `camara_belleza.py` (cada una son 5 números).

### Pestaña Bucle
Graba unos segundos con la webcam y emítelos en bucle por la cámara virtual en vez de la imagen en vivo.
- **Grabar** (5, 10, 20 o 30 s): mientras graba sigue saliendo la cámara en vivo. Se guarda la imagen
  *sin filtros*; los filtros de belleza y de imagen se aplican al reproducir, así que puedes ajustarlos
  después (y cambiarlos mientras suena).
- **Cargar video…**: usa un video que ya tengas (mp4, avi, mov, mkv, webm; hasta 60 s, a ~30 fps).
- **Ida y vuelta**: reproduce hacia delante y luego hacia atrás, para que no haya un salto al repetir.
- **Reproducir el bucle en lugar de la cámara en vivo**: el interruptor. Mientras está activo aparece un
  **aviso rojo** sobre las pestañas. Desmárcalo para volver a la cámara. No se puede grabar ni cargar
  mientras suena.
- **Borrar bucle**: lo quita de la memoria y borra `bucle.avi`.
- El bucle se guarda en `bucle.avi` (en la carpeta del programa; está en `.gitignore`) y se recarga al
  abrir, pero **nunca se reproduce solo**: siempre hay que activarlo a mano.
- No incluye audio: el micrófono sigue siendo el de siempre, así que si hablas, la boca del bucle no
  coincidirá. Sirve para una presencia tranquila, no para hablar.
- Si cambias de resolución o de cámara, el bucle se recorta para llenar el cuadro, sin estirarlo.

### Imagen (bajo la vista previa)
De -100 a +100 cada uno (0 = sin cambio):
- **Exposición**: ±2 pasos de luz (EV), calculada en luz lineal como en una cámara.
- **Temperatura**: balance de blancos de la imagen. Positivo = más cálida (tirando a naranja),
  negativo = más fría (tirando a azul). Mantiene el brillo: cambia el tono, no la luminosidad.
- **Sombras**: aclara (+) u oscurece (-) solo los tonos oscuros; el negro puro no se mueve.
- **Luces**: aclara (+) o recupera (-) solo los tonos claros; el blanco puro no se mueve.
- **Brillo**, **Contraste** y **Saturación**.

Funcionan aunque los filtros de belleza estén apagados. "Restablecer" los devuelve a 0.

Los ajustes se guardan en `config.json` y se recuerdan la próxima vez.
`iniciar.bat --list` muestra las cámaras detectadas.

## Si va a pocos fps
La fluidez depende sobre todo de la webcam. Prueba otra resolución, más luz en la habitación
(con poca luz muchas webcams bajan los fps) o conectarla directamente a un puerto USB del equipo, sin hub.
Si tu equipo va justo, el alisado es el filtro que más cuesta; baja su
intensidad a 0 cuando no lo uses.

## Archivos
- `camara_belleza.py`: programa principal
- `face_landmarker.task`: modelo de MediaPipe (de storage.googleapis.com/mediapipe-models)
- `requirements.txt`: dependencias de Python
