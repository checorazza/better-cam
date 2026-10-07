"""Cámara virtual con filtros de belleza (alisado de piel + labios pintados).

Abre un panel de control con vista previa. Lo que ves ahí es lo que sale por la
cámara virtual de OBS: en Zoom/Meet/Teams elige "OBS Virtual Camera".
"""
import argparse
import json
import math
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk
from types import SimpleNamespace

import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image, ImageTk

HERE = Path(__file__).parent
MODEL = HERE / "face_landmarker.task"
CONFIG = HERE / "config.json"
BUCLE_PATH = HERE / "bucle.avi"  # el bucle grabado/cargado se guarda aquí y se recarga al abrir
REC_OPTIONS = (5, 10, 20, 30)    # duraciones de grabación (segundos)
LOOP_MAX_SECONDS = 60            # tope al cargar un video ya hecho

# Índices de landmarks de MediaPipe Face Mesh (478 puntos).
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
             400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
             54, 103, 67, 109]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37,
              39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82,
              81, 80, 191]
L_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
R_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
L_BROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
R_BROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]

# Cuánto se extiende el alisado por ENCIMA del óvalo de MediaPipe (que acaba hacia la mitad de
# la frente), en alturas de cara. El pelo se descarta por color, así que pasarse no lo alisa.
FOREHEAD_EXTRA = 0.30

# Colores de rubor (RGB). Se aplican por multiplicación sobre la piel: el color final
# depende de tu tono de piel, y los tonos claros dan un efecto más sutil.
BLUSH_PRESETS = {
    "Rosa": (250, 140, 160),
    "Melocotón": (255, 170, 140),
    "Coral": (255, 125, 105),
    "Rosa palo": (235, 160, 170),
    "Ciruela": (205, 105, 145),
    "Terracota": (230, 135, 105),
}

# Posiciones del rubor (guía de colocación por tipo de cara). Todo es relativo a la cara y se
# mide en "distancias entre ojos": lat = separación lateral desde el centro, v = altura (0 =
# línea de los ojos, 1 = boca), rx/ry = semiejes de cada mancha, tilt = inclinación del eje
# largo hacia la sien (rad), gain = intensidad relativa. `bridge` añade una mancha central
# sobre la nariz: (v, rx, ry).
BLUSH_STYLES = {
    "Diagonal clásico · cara ovalada":
        dict(lat=0.62, v=0.50, rx=0.40, ry=0.27, tilt=0.35),
    "Sobre las manzanas · cara redonda":
        dict(lat=0.60, v=0.52, rx=0.33, ry=0.29, tilt=0.45),
    "Horizontal · cara alargada":
        dict(lat=0.66, v=0.50, rx=0.54, ry=0.19, tilt=0.0),
    "Alto a las sienes · cara corazón":
        dict(lat=0.72, v=0.36, rx=0.42, ry=0.24, tilt=0.55),
    "Redondo suave · cara cuadrada":
        dict(lat=0.62, v=0.50, rx=0.34, ry=0.32, tilt=0.10),
    "Manzanas hacia fuera · diamante":
        dict(lat=0.76, v=0.50, rx=0.38, ry=0.27, tilt=0.30),
    "Mejillas internas · cara ancha":
        dict(lat=0.44, v=0.44, rx=0.30, ry=0.26, tilt=0.15),
    "Efecto sol · mejillas y nariz":
        dict(lat=0.62, v=0.46, rx=0.46, ry=0.19, tilt=0.0, bridge=(0.40, 0.42, 0.15)),
    "Muñeca · manzanas altas":
        dict(lat=0.56, v=0.38, rx=0.27, ry=0.25, tilt=0.15),
    "Elegante · levantado a las sienes":
        dict(lat=0.76, v=0.40, rx=0.44, ry=0.20, tilt=0.50),
    "Natural · mínimo sobre manzanas":
        dict(lat=0.60, v=0.52, rx=0.29, ry=0.25, tilt=0.20, gain=0.65),
}
DEFAULT_BLUSH_STYLE = next(iter(BLUSH_STYLES))

# Colores de labial (RGB) que aparecen como botones en el panel.
LIP_PRESETS = {
    "Rosa": (225, 90, 120),
    "Rojo": (200, 30, 50),
    "Coral": (240, 100, 90),
    "Nude": (190, 120, 110),
    "Vino": (140, 30, 70),
    "Fucsia": (210, 40, 130),
    "Malva": (170, 90, 140),
}


RESOLUTIONS = {
    "HD 1280×720 (más nítida, menos fluida)": (1280, 720),
    "960×540 (recomendada)": (960, 540),
    "640×480 (4:3, la más ligera)": (640, 480),
}

# Ajustes de imagen en el orden en que aparecen en el panel: (atributo, título).
ADJUSTMENTS_UI = (("exposure", "Exposición"), ("temperature", "Temperatura"),
                  ("shadows", "Sombras"), ("highlights", "Luces"),
                  ("brightness", "Brillo"), ("contrast", "Contraste"),
                  ("saturation", "Saturación"))
ADJUSTMENTS = tuple(k for k, _ in ADJUSTMENTS_UI)

# Selectores de color del panel: tipo -> (atributo de State, colores, título del diálogo).
COLOR_KINDS = {
    "lips": ("lip_rgb", LIP_PRESETS, "Color del labial"),
    "blush": ("blush_rgb", BLUSH_PRESETS, "Color del rubor"),
}

# --------------------------------------------------------------------------- estado

class State:
    """Valores compartidos entre el panel (hilo principal) y el procesado (hilo worker)."""

    def __init__(self):
        self.enabled = True
        self.smooth = 0.6          # 0-1
        self.lips = 0.55           # 0-1 (0 = sin labial)
        self.lip_rgb = LIP_PRESETS["Rosa"]
        self.blush = 0.0          # 0-1 (0 = sin rubor)
        self.blush_rgb = BLUSH_PRESETS["Rosa"]
        self.blush_style = DEFAULT_BLUSH_STYLE
        self.mirror_preview = False
        for k in ADJUSTMENTS:      # ajustes de imagen, -100..100 (0 = sin cambio)
            setattr(self, k, 0)
        self.cam_index = None     # índice DirectShow pedido por el panel
        self.res = (960, 540)      # resolución de captura pedida por el panel
        # Bucle de video: el panel pide cosas con *_request y el worker contesta con loop_*.
        self.loop_play = False     # True = la cámara virtual emite el bucle en vez de la cámara
        self.loop_pingpong = True  # ida y vuelta (evita el salto al repetir)
        self.rec_seconds = 10      # duración de la próxima grabación
        self.rec_request = 0       # >0: el panel pide grabar esos segundos
        self.rec_progress = None   # None = no graba; 0-1 mientras graba
        self.load_request = None   # ruta de un video para cargar como bucle
        self.clear_request = False # el panel pide borrar el bucle
        self.loop_ready = False    # hay un bucle disponible
        self.loop_info = ""        # descripción del bucle actual
        self.loop_msg = ""         # último aviso sobre el bucle (grabado, error...)
        self.status = "Iniciando…"
        self.fps = 0.0
        self.error = None
        self.latest = None         # último fotograma procesado (BGR)
        self.stop = threading.Event()

    def load(self):
        try:
            d = json.loads(CONFIG.read_text())
        except Exception:
            return {}
        self.smooth = float(d.get("smooth", self.smooth))
        self.lips = float(d.get("lips", self.lips))
        self.lip_rgb = tuple(d.get("lip_rgb", self.lip_rgb))
        self.blush = max(0.0, min(1.0, float(d.get("blush", self.blush))))
        self.blush_rgb = tuple(d.get("blush_rgb", self.blush_rgb))
        style = d.get("blush_style", self.blush_style)
        self.blush_style = style if style in BLUSH_STYLES else DEFAULT_BLUSH_STYLE
        self.mirror_preview = bool(d.get("mirror_preview", self.mirror_preview))
        self.enabled = bool(d.get("enabled", self.enabled))
        self.res = tuple(d.get("res", self.res))
        self.loop_pingpong = bool(d.get("loop_pingpong", self.loop_pingpong))
        secs = d.get("rec_seconds", self.rec_seconds)
        self.rec_seconds = secs if secs in REC_OPTIONS else 10
        for k in ADJUSTMENTS:
            setattr(self, k, max(-100, min(100, int(d.get(k, 0)))))
        return d  # loop_play nunca se restaura: el bucle jamás arranca solo

    def save(self, camera_name):
        try:
            CONFIG.write_text(json.dumps({
                "camera": camera_name, "smooth": self.smooth, "lips": self.lips,
                "lip_rgb": list(self.lip_rgb), "blush": self.blush,
                "blush_rgb": list(self.blush_rgb), "blush_style": self.blush_style,
                "mirror_preview": self.mirror_preview,
                "enabled": self.enabled, "res": list(self.res),
                "loop_pingpong": self.loop_pingpong, "rec_seconds": self.rec_seconds,
                **{k: getattr(self, k) for k in ADJUSTMENTS},
            }))
        except OSError:
            pass


# --------------------------------------------------------------------------- cámaras

def list_cameras():
    """[(indice, nombre)] de las webcams, sin la propia cámara virtual de OBS.
    Debe llamarse desde el hilo principal (usa COM)."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        names = FilterGraph().get_input_devices()
    except Exception:
        names = []
        for i in range(6):
            c = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if c.isOpened():
                names.append(f"Cámara {i}")
            c.release()
    return [(i, n) for i, n in enumerate(names) if "obs virtual" not in n.lower()]


def open_camera(index, width, height, fps):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # menos retraso (si el driver lo respeta)
    return cap


# --------------------------------------------------------------------------- bucle de video

def fit_frame(frame, w, h):
    """Escala `frame` para llenar w x h recortando lo que sobre (sin estirar la imagen)."""
    fh, fw = frame.shape[:2]
    if (fh, fw) == (h, w):
        return frame
    scale = max(w / fw, h / fh)
    nw, nh = max(w, round(fw * scale)), max(h, round(fh * scale))
    big = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    return big[y0:y0 + h, x0:x0 + w]


class Loop:
    """Un bucle de video en memoria: fotogramas JPEG de calidad alta (unos 20 MB por cada
    10 s a 960x540, en vez de ~450 MB sin comprimir) y los fps a los que se grabaron."""

    QUALITY = 92

    def __init__(self):
        self.frames, self.fps, self.size = [], 30.0, None

    @property
    def ready(self):
        return len(self.frames) > 1

    @staticmethod
    def encode(frame):
        return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, Loop.QUALITY])[1].tobytes()

    def decode(self, i):
        return cv2.imdecode(np.frombuffer(self.frames[i], np.uint8), cv2.IMREAD_COLOR)

    def set_frames(self, frames, fps, size):
        self.frames, self.fps, self.size = frames, float(fps), size

    @property
    def duration(self):
        return len(self.frames) / self.fps if self.fps else 0.0

    def info(self):
        mb = sum(len(f) for f in self.frames) / 1e6
        return (f"Bucle: {self.duration:.0f} s · {self.fps:.0f} fps · {len(self.frames)} "
                f"fotogramas · {mb:.0f} MB en memoria")

    def index_at(self, t, pingpong):
        """Fotograma que toca a los `t` segundos desde que empezó la reproducción. Con
        `pingpong` va hacia delante y luego hacia atrás, así no hay salto al repetir."""
        n = len(self.frames)
        k = int(t * self.fps)
        if pingpong and n > 1:
            cycle = 2 * (n - 1)
            k %= cycle
            return k if k < n else cycle - k
        return k % n

    def save(self, path):
        """Guarda el bucle como video (AVI con MJPG, que no necesita códecs extra)."""
        w, h = self.size
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), self.fps, (w, h))
        if not writer.isOpened():
            raise OSError("no pude crear el archivo de video")
        try:
            for i in range(len(self.frames)):
                writer.write(self.decode(i))
        finally:
            writer.release()

    def load_video(self, path, max_seconds=LOOP_MAX_SECONDS, max_width=960):
        """Carga un video ya hecho (hasta `max_seconds`, a ~30 fps como mucho)."""
        p = str(path)
        tmp = None
        if not p.isascii():  # OpenCV en Windows suele fallar con rutas con tildes o eñes
            import shutil, tempfile
            tmp = Path(tempfile.mkdtemp()) / ("video" + Path(p).suffix)
            shutil.copyfile(p, tmp)
            p = str(tmp)
        cap = cv2.VideoCapture(p)
        try:
            if not cap.isOpened():
                raise ValueError("no pude abrir ese video")
            src_fps = cap.get(cv2.CAP_PROP_FPS)
            if not (1 <= src_fps <= 240):
                src_fps = 30.0
            step = max(1, round(src_fps / 30))
            frames, size, i = [], None, 0
            while i < int(max_seconds * src_fps):
                ok, f = cap.read()
                if not ok:
                    break
                if i % step == 0:
                    if f.shape[1] > max_width:
                        f = cv2.resize(f, (max_width, round(f.shape[0] * max_width / f.shape[1])),
                                       interpolation=cv2.INTER_AREA)
                    size = size or (f.shape[1], f.shape[0])
                    frames.append(self.encode(f if f.shape[:2] == size[::-1] else fit_frame(f, *size)))
                i += 1
        finally:
            cap.release()
            if tmp is not None:
                shutil.rmtree(tmp.parent, ignore_errors=True)
        if len(frames) < 2:
            raise ValueError("el video es demasiado corto")
        self.set_frames(frames, src_fps / step, size)


# --------------------------------------------------------------------------- filtro

class OneEuro:
    """Filtro One Euro vectorizado: suaviza el temblor cuando estás quieta y casi no
    añade retraso cuando te mueves (el corte de frecuencia sube con la velocidad)."""

    def __init__(self, min_cutoff=1.5, beta=0.08, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x = self.dx = self.t = None

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        a_d = self._alpha(self.d_cutoff, dt)
        dx = (x - self.x) / dt
        self.dx = a_d * dx + (1 - a_d) * self.dx
        speed = np.linalg.norm(self.dx, axis=1, keepdims=True)
        a = self._alpha(self.min_cutoff + self.beta * speed, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x


class ImageAdjust:
    """Exposición, temperatura, sombras, luces, brillo, contraste y saturación (-100..100;
    0 = sin cambio).

    Todo lo tonal y la temperatura se funden en una sola curva por canal (una única pasada
    con LUT), así que apenas cuesta tiempo por fotograma."""

    CURVE_AMOUNT = 0.25  # cuánto mueve las sombras/luces el deslizador al máximo
    TEMP_STOPS = 0.5     # temperatura al máximo: el rojo sube y el azul baja ~0,5 pasos de luz

    def __init__(self):
        self._key = self._lut = None

    @classmethod
    def build_curve(cls, exposure, shadows, highlights, brightness, contrast, temperature=0):
        """LUT de 256 entradas x 3 canales (orden BGR) para cv2.LUT."""
        x0 = np.arange(256, dtype=np.float64) / 255.0
        # Temperatura = balance de blancos: ganancia distinta por canal (B, G, R). Positivo =
        # más cálida (naranja: sube R, baja B); negativo = más fría (azulada). Se normaliza
        # para conservar la luminosidad: cambia el tono de la imagen, no su brillo.
        t = temperature / 100.0
        gains = np.array([2.0 ** (-cls.TEMP_STOPS * t), 1.0, 2.0 ** (cls.TEMP_STOPS * t)])
        gains /= 0.0722 * gains[0] + 0.7152 * gains[1] + 0.2126 * gains[2]
        # Exposición y temperatura en luz lineal (+-100 de exposición = +-2 pasos, EV).
        ev = 2.0 ** (exposure / 50)
        if exposure or temperature:
            x = np.clip(x0[None, :] ** 2.2 * ev * gains[:, None], 0, 1) ** (1 / 2.2)
        else:
            x = np.tile(x0, (3, 1))
        # Máscaras estrechas que valen 0 en negro y en blanco puro (no se "lava" el negro):
        # las sombras actúan sobre todo hacia el 20% de la escala; las luces, hacia el 80%.
        a = cls.CURVE_AMOUNT
        norm = 5 ** 5 / 4 ** 4  # deja el pico de cada máscara en 1
        x = x + a * (shadows / 100) * norm * x * (1 - x) ** 4
        x = x + a * (highlights / 100) * norm * x ** 4 * (1 - x)
        x = np.maximum.accumulate(np.clip(x, 0, 1), axis=1)  # nunca invertir tonos
        x = (x - 0.5) * (1 + contrast / 200) + 0.5 + brightness / 255.0  # contraste x0.5..x1.5
        lut = np.clip(np.rint(x * 255), 0, 255).astype(np.uint8)  # (3, 256)
        return np.ascontiguousarray(lut.T).reshape(256, 1, 3)

    def __call__(self, frame, s):
        out = frame
        key = (s.exposure, s.shadows, s.highlights, s.brightness, s.contrast, s.temperature)
        if any(key):
            if key != self._key:
                self._lut, self._key = self.build_curve(*key), key
            out = cv2.LUT(out, self._lut)
        if s.saturation:
            sat = 1 + s.saturation / 100  # 0 = blanco y negro, 2 = el doble de color
            gray = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            out = cv2.addWeighted(out, sat, gray, 1 - sat, 0)
        return out


def poly(img, pts, value=255):
    cv2.fillPoly(img, [pts.astype(np.int32)], value, lineType=cv2.LINE_AA)


class BeautyFilter:
    def __init__(self, state):
        self.s = state
        self.euro = OneEuro()
        self.pts = None
        self._skin_mean = None  # color medio de la piel (LAB), suavizado en el tiempo
        self.landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(MODEL)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
            )
        )
        self._t0 = time.monotonic()

    def _landmarks(self, frame):
        h, w = frame.shape[:2]
        # Detectar en baja resolución: es más rápido y los landmarks son normalizados.
        small = cv2.resize(frame, (640, int(640 * h / w)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        now = time.monotonic() - self._t0
        res = self.landmarker.detect_for_video(image, int(now * 1000))
        if not res.face_landmarks:
            self.euro.reset()
            self._skin_mean = None
            return None
        pts = np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]], dtype=np.float32)
        return self.euro(pts, now)

    @staticmethod
    def _face_box(frame, pts):
        """Rectángulo (x0, y0, x1, y1) que rodea la cara, o None si es demasiado pequeño."""
        h, w = frame.shape[:2]
        x0, y0 = np.floor(pts[FACE_OVAL].min(axis=0)).astype(int)
        x1, y1 = np.ceil(pts[FACE_OVAL].max(axis=0)).astype(int)
        pad = int(0.08 * max(x1 - x0, y1 - y0))
        x0, y0 = max(x0 - pad, 0), max(y0 - pad, 0)
        x1, y1 = min(x1 + pad, w), min(y1 + pad, h)
        return None if (x1 - x0 < 20 or y1 - y0 < 20) else (x0, y0, x1, y1)

    @staticmethod
    def _skin_mask(pts, box):
        """Máscara suave (alto x ancho, 0-1) de la piel: óvalo de la cara menos ojos, cejas
        y boca, y con menos peso cerca del borde (evita halo en pelo/fondo)."""
        x0, y0, x1, y1 = box
        off = np.array([x0, y0], dtype=np.float32)
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        poly(mask, pts[FACE_OVAL] - off)
        for part in (L_EYE, R_EYE, LIPS_OUTER, L_BROW, R_BROW):
            poly(mask, pts[part] - off, 0)
        k = max(3, ((x1 - x0) // 40) | 1)
        mask = cv2.GaussianBlur(mask, (k * 2 + 1, k * 2 + 1), 0).astype(np.float32) / 255.0
        core = cv2.erode((mask > 0.98).astype(np.uint8) * 255, np.ones((k, k), np.uint8))
        core = cv2.GaussianBlur(core, (k * 2 + 1, k * 2 + 1), 0).astype(np.float32) / 255.0
        return np.minimum(mask, core)

    def _skin(self, frame, pts, strength, reg):
        x0, y0, x1, y1 = reg.box
        roi = frame[y0:y1, x0:x1]
        # Peso = zona geométrica (frente + cara) x parecido a piel. Aquí el parecido es laxo:
        # admite granitos y sombras, pero deja fuera el pelo y el fondo.
        mask = (reg.face * self._skin_likeness(reg.lab_s, reg.mean, loose=True))[..., None]

        # Alisado que conserva bordes (en media resolución por rendimiento).
        small = cv2.resize(roi, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        smooth = cv2.bilateralFilter(small, d=9, sigmaColor=40, sigmaSpace=9)
        smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=40, sigmaSpace=9)
        smooth = cv2.resize(smooth, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Devolver algo de textura fina para que no parezca plástico.
        blur = cv2.GaussianBlur(roi, (0, 0), 3)
        detail = roi.astype(np.float32) - blur.astype(np.float32)
        smooth = smooth.astype(np.float32) + 0.25 * detail * (1 - strength)

        a = mask * strength
        out = roi.astype(np.float32) * (1 - a) + smooth * a
        frame[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)
        return frame

    def _skin_region(self, frame, pts):
        """La piel visible de la frente y la cara, para el alisado. Devuelve None si no hay
        suficiente cara, o un objeto con:
          box     (x0, y0, x1, y1) del recorte que contiene todo
          face    máscara 0-1 de frente + cara, sin ojos, cejas ni boca
          lab_s   el recorte en LAB 8 bits (float32), algo difuminado
          mean    color medio de la piel (LAB), suavizado en el tiempo
        La geometría es generosa (frente hasta el nacimiento del pelo, orejas) y se difumina;
        lo que no es piel (pelo, pared, auriculares) se descarta después por su color con
        _skin_likeness, así no hay borde ni halo."""
        h, w = frame.shape[:2]
        oval = pts[FACE_OVAL]
        cx = float(oval[:, 0].mean())
        face_h = float(oval[:, 1].max() - oval[:, 1].min())
        eyes_mid = (pts[R_EYE].mean(axis=0) + pts[L_EYE].mean(axis=0)) / 2
        down = pts[LIPS_OUTER].mean(axis=0) - eyes_mid
        down = down / (float(np.linalg.norm(down)) + 1e-6)  # de los ojos a la boca (unitario)

        # Frente: la parte alta del óvalo llega solo hasta el centro de la frente; el pelo
        # suele nacer ~0,2 de cara más arriba (o más, con la raya o el flequillo apartados).
        top = oval[oval[:, 1] <= oval[:, 1].mean()]
        fore = top - down * FOREHEAD_EXTRA * face_h
        fore[:, 0] = cx + (fore[:, 0] - cx) * 0.92
        fore_hull = cv2.convexHull(np.vstack([top, fore]).astype(np.float32)).reshape(-1, 2)

        allp = np.vstack([oval, fore])
        pad = int(0.2 * face_h)
        x0, y0 = np.maximum(np.floor(allp.min(axis=0)).astype(int) - pad, 0)
        x1, y1 = np.minimum(np.ceil(allp.max(axis=0)).astype(int) + pad, [w, h])
        if x1 - x0 < 20 or y1 - y0 < 20:
            return None
        off = np.array([x0, y0], np.float32)

        face_bin = np.zeros((y1 - y0, x1 - x0), np.uint8)
        poly(face_bin, oval - off)
        poly(face_bin, fore_hull - off)
        e = max(3, int(0.06 * face_h) | 1)
        k = max(3, int(0.08 * face_h) | 1)
        # Ensanchar hacia fuera y difuminar.
        grown = cv2.dilate(face_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e)))
        face = cv2.GaussianBlur(grown, (k, k), 0).astype(np.float32) / 255.0

        feat = np.zeros_like(face_bin)
        for part in (L_EYE, R_EYE, LIPS_OUTER, L_BROW, R_BROW):
            poly(feat, pts[part] - off)
        face *= 1.0 - cv2.GaussianBlur(feat, (5, 5), 0).astype(np.float32) / 255.0

        lab = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32)
        # Los pesos "esto es piel" se calculan sobre un LAB algo difuminado: el ruido de color
        # (sobre todo en zonas oscuras) los volvería a motas.
        lab_s = cv2.GaussianBlur(lab, (0, 0), 1.5)

        # Color actual de tu piel: mediana (robusta ante pelo o fondo colados en la región)
        # afinada con el parecido a piel, a media resolución y suavizada en el tiempo.
        ls, ms = lab[::2, ::2], face[::2, ::2]
        sel = ms > 0.9
        mean = None
        if int(sel.sum()) >= 50:
            mean = np.median(ls[sel], axis=0)
            for _ in range(2):
                wgt = ms * self._skin_likeness(ls, mean)
                if float(wgt.sum()) < 1.0:
                    mean = None
                    break
                mean = (ls * wgt[..., None]).sum(axis=(0, 1)) / float(wgt.sum())
        if mean is not None:
            self._skin_mean = (mean if self._skin_mean is None
                               else 0.85 * self._skin_mean + 0.15 * mean)
        if self._skin_mean is None:
            return None
        return SimpleNamespace(box=(int(x0), int(y0), int(x1), int(y1)), face=face,
                               lab_s=lab_s, mean=self._skin_mean)

    @staticmethod
    def _skin_likeness(lab, mean, loose=False):
        """Peso 0-1 de cuánto se parece cada píxel (LAB 8 bits) al color de piel `mean`:
        1 si el color es casi el mismo, 0 si se aleja. Deja fuera pelo, fondo, ropa y sombras
        muy profundas, que tienen otro color o mucha menos luz. El modo estricto sirve para
        estimar el color medio de la piel; `loose` admite más variación (granitos, rubor,
        sombras) y sirve para alisar, donde un error no cambia colores."""
        cf, cz, lf, lz = (24.0, 44.0, 70.0, 130.0) if loose else (8.0, 22.0, 50.0, 110.0)
        d = lab - mean
        chroma = np.sqrt(d[..., 1] ** 2 + d[..., 2] ** 2)
        return (np.clip((cz - chroma) / (cz - cf), 0, 1)
                * np.clip((lz - np.abs(d[..., 0])) / (lz - lf), 0, 1))

    @staticmethod
    def _stamp(alpha, box, cx, cy, rx, ry, th):
        """Suma (máximo) una mancha gaussiana elíptica, de semiejes rx/ry y eje largo
        girado `th`, en `alpha` (que cubre la caja `box` del fotograma)."""
        x0, y0, x1, y1 = box
        c, s = math.cos(th), math.sin(th)
        half = int(1.8 * max(rx, ry)) + 1      # fuera de esta ventana el aporte es ~0
        wx0, wx1 = max(int(cx) - half, x0), min(int(cx) + half, x1)
        wy0, wy1 = max(int(cy) - half, y0), min(int(cy) + half, y1)
        if wx1 <= wx0 or wy1 <= wy0:
            return
        dx = np.arange(wx0, wx1, dtype=np.float32)[None, :] - cx
        dy = np.arange(wy0, wy1, dtype=np.float32)[:, None] - cy
        d2 = ((dx * c + dy * s) / rx) ** 2 + ((dy * c - dx * s) / ry) ** 2
        win = alpha[wy0 - y0:wy1 - y0, wx0 - x0:wx1 - x0]
        np.maximum(win, np.exp(-2.0 * d2), out=win)

    def _blush(self, frame, pts, strength, rgb, style):
        box = self._face_box(frame, pts)
        if box is None:
            return frame
        x0, y0, x1, y1 = box
        roi = frame[y0:y1, x0:x1]

        # Referencias de la cara, que siguen el giro/inclinación de la cabeza: centro de cada
        # ojo y de la boca. R_EYE queda a la izquierda de la imagen y L_EYE a la derecha.
        eye_a, eye_b = pts[R_EYE].mean(axis=0), pts[L_EYE].mean(axis=0)
        u = eye_b - eye_a                      # línea de los ojos (hacia la derecha de la imagen)
        mid = (eye_a + eye_b) / 2
        down = pts[LIPS_OUTER].mean(axis=0) - mid  # de los ojos a la boca
        scale = float(np.linalg.norm(u))
        if scale < 8:
            return frame
        ang = math.atan2(u[1], u[0])

        # Una mancha gaussiana por mejilla, colocada según el estilo elegido; el eje largo
        # sube hacia la sien en cada lado. Si el estilo lo pide, otra sobre el puente de la nariz.
        alpha = np.zeros(roi.shape[:2], np.float32)
        for side in (-1, 1):
            cx, cy = mid + side * style["lat"] * u + style["v"] * down
            self._stamp(alpha, box, cx, cy, style["rx"] * scale, style["ry"] * scale,
                        ang - side * style["tilt"])
        if "bridge" in style:
            v, brx, bry = style["bridge"]
            cx, cy = mid + v * down
            self._stamp(alpha, box, cx, cy, brx * scale, bry * scale, ang)
        strength = min(1.0, strength * style.get("gain", 1.0))

        # Solo sobre piel (no ojos, boca ni fuera del óvalo de la cara).
        alpha *= self._skin_mask(pts, box)

        # Mezcla por multiplicación: conserva la textura de la piel, no parece pintura.
        r, g, b = rgb
        tint = np.array([b, g, r], np.float32) / 255.0
        a = (alpha * (0.7 * strength))[..., None]
        out = roi.astype(np.float32) * (1 - a * (1 - tint))
        frame[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)
        return frame

    def _lips(self, frame, pts, strength, rgb):
        h, w = frame.shape[:2]
        x0, y0 = np.floor(pts[LIPS_OUTER].min(axis=0)).astype(int) - 8
        x1, y1 = np.ceil(pts[LIPS_OUTER].max(axis=0)).astype(int) + 8
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 - x0 < 6 or y1 - y0 < 6:
            return frame

        roi = frame[y0:y1, x0:x1]
        off = np.array([x0, y0], dtype=np.float32)
        mask = np.zeros(roi.shape[:2], np.uint8)
        poly(mask, pts[LIPS_OUTER] - off)
        poly(mask, pts[LIPS_INNER] - off, 0)  # no pintar dientes/interior
        k = max(1, (roi.shape[1] // 30) | 1)
        mask = cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32)[..., None] / 255.0

        # Cambiar tono y saturación conservando la luminosidad (textura natural).
        r, g, b = rgb
        th, ts, _ = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = th
        hsv[..., 1] = hsv[..., 1] * 0.3 + ts * 0.7
        hsv[..., 2] = hsv[..., 2] * 0.92
        tinted = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

        a = mask * strength
        out = roi.astype(np.float32) * (1 - a) + tinted.astype(np.float32) * a
        frame[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)
        return frame

    def apply(self, frame):
        s = self.s
        if not s.enabled or (s.smooth <= 0 and s.lips <= 0 and s.blush <= 0):
            return frame
        pts = self._landmarks(frame)
        if pts is None:
            return frame
        frame = frame.copy()
        if s.smooth > 0:
            reg = self._skin_region(frame, pts)
            if reg is not None:
                frame = self._skin(frame, pts, s.smooth, reg)
        if s.blush > 0:
            frame = self._blush(frame, pts, s.blush, s.blush_rgb,
                                BLUSH_STYLES.get(s.blush_style, BLUSH_STYLES[DEFAULT_BLUSH_STYLE]))
        if s.lips > 0:
            frame = self._lips(frame, pts, s.lips, s.lip_rgb)
        return frame


# --------------------------------------------------------------------------- worker

def worker(state, fps):
    """Captura -> filtro -> cámara virtual, hasta que state.stop se active."""
    try:
        flt = BeautyFilter(state)
        adjust = ImageAdjust()
        cap, opened = None, None  # opened = (indice, (ancho, alto))
        cam = None
        t_fps, n = time.monotonic(), 0

        loopbox = [Loop()]  # en una lista para poder cambiarlo desde los hilos auxiliares
        rec = None          # grabación en curso: {"t0", "secs", "frames"}
        was_looping, loop_t0, last_idx = False, 0.0, -1

        def sync_info():
            lp = loopbox[0]
            state.loop_ready = lp.ready
            state.loop_info = lp.info() if lp.ready else ""
            if not lp.ready:
                state.loop_play = False

        def save_bucle(lp):  # en segundo plano: escribir el AVI no debe congelar la imagen
            try:
                lp.save(BUCLE_PATH)
            except Exception as e:
                state.loop_msg = f"No pude guardar {BUCLE_PATH.name}: {e}"

        def load_bucle(path):
            try:
                new = Loop()
                new.load_video(path)
                loopbox[0] = new
                state.loop_msg = "Video cargado como bucle."
                if Path(path).resolve() != BUCLE_PATH.resolve():
                    save_bucle(new)
            except Exception as e:
                state.loop_msg = f"No pude cargar el video: {e}"
            sync_info()

        while not state.stop.is_set():
            # Peticiones del panel sobre el bucle: borrar y cargar un video.
            if state.clear_request:
                state.clear_request = False
                loopbox[0] = Loop()
                try:
                    BUCLE_PATH.unlink()
                except FileNotFoundError:
                    pass
                except OSError as e:
                    state.loop_msg = f"No pude borrar {BUCLE_PATH.name}: {e}"
                else:
                    state.loop_msg = "Bucle borrado."
                sync_info()
            if state.load_request:
                path, state.load_request = state.load_request, None
                state.loop_msg = "Cargando video…"
                threading.Thread(target=load_bucle, args=(path,), daemon=True).start()

            # ¿El panel pidió otra cámara u otra resolución?
            want = (state.cam_index, state.res)
            if state.cam_index is not None and want != opened:
                old = opened
                if cap is not None:  # soltar antes de abrir: DirectShow no admite dos aperturas
                    cap.release()
                    cap = None
                new = open_camera(state.cam_index, *state.res, fps)
                if new.isOpened() and new.read()[0]:
                    if cam is not None and old[1] != state.res:
                        cam.close()  # la cámara virtual se recrea con el nuevo tamaño
                        cam = None
                    cap, opened = new, want
                    flt.euro.reset()
                    state.error = None
                else:
                    new.release()
                    state.error = "No pude abrir esa cámara (¿la usa otra app?)."
                    cap, opened = None, None
                    if old:  # volver a la anterior
                        state.cam_index, state.res = old
                        back = open_camera(old[0], *old[1], fps)
                        if back.isOpened():
                            cap, opened = back, old
                        else:
                            back.release()
                    else:
                        state.cam_index = None
            if cap is None:
                time.sleep(0.05)
                continue

            loop = loopbox[0]
            if state.loop_play and not loop.ready:
                state.loop_play = False
            looping = state.loop_play and cam is not None  # con cámara virtual ya creada

            if looping:
                # Reproducir el bucle en lugar de la cámara. El tiempo manda: el fotograma
                # sale de cuánto lleva sonando, así la velocidad es la grabada.
                if not was_looping:
                    loop_t0, last_idx = time.monotonic(), -1
                    flt.euro.reset()
                    rec, state.rec_progress = None, None  # no se graba mientras suena el bucle
                state.rec_request = 0
                idx = loop.index_at(time.monotonic() - loop_t0, state.loop_pingpong)
                if idx < last_idx and not state.loop_pingpong:
                    flt.euro.reset()  # al repetir hay un salto: no suavizar a través de él
                last_idx = idx
                frame = fit_frame(loop.decode(idx), w, h)
            else:
                if was_looping:  # volver a la cámara en vivo: tirar lo que se acumuló
                    for _ in range(3):
                        cap.read()
                    flt.euro.reset()
                ok, frame = cap.read()
                if not ok:
                    continue
                if cam is None:
                    h, w = frame.shape[:2]
                    cam = pyvirtualcam.Camera(w, h, fps, fmt=pyvirtualcam.PixelFormat.BGR,
                                              backend="obs")
                    state.status = f"Cámara virtual activa: {cam.device} ({w}x{h})"
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h))

                # Grabación: se guarda la imagen de la cámara SIN filtros; los filtros se
                # aplican al reproducir, así se pueden ajustar después.
                if state.rec_request and rec is None:
                    rec = {"t0": time.monotonic(), "secs": state.rec_request, "frames": []}
                    state.rec_request, state.rec_progress = 0, 0.0
                    state.loop_msg = ""
                if rec is not None:
                    rec["frames"].append(Loop.encode(frame))
                    elapsed = time.monotonic() - rec["t0"]
                    state.rec_progress = min(elapsed / rec["secs"], 1.0)
                    if elapsed >= rec["secs"]:
                        new = Loop()
                        new.set_frames(rec["frames"], len(rec["frames"]) / elapsed, (w, h))
                        loopbox[0] = new
                        rec, state.rec_progress = None, None
                        state.loop_msg = f"Bucle grabado (se guarda en {BUCLE_PATH.name})."
                        sync_info()
                        threading.Thread(target=save_bucle, args=(new,), daemon=True).start()
            was_looping = looping

            out = adjust(flt.apply(frame), state)
            cam.send(out)
            if looping:
                cam.sleep_until_next_frame()  # la cámara virtual va a su ritmo (fps)
            state.latest = out
            n += 1
            now = time.monotonic()
            if now - t_fps >= 1.0:
                state.fps = n / (now - t_fps)
                t_fps, n = now, 0
        if cap is not None:
            cap.release()
        if cam is not None:
            cam.close()
    except Exception as e:  # mostrar el error en el panel en vez de morir en silencio
        state.error = f"Error: {e}"
        state.status = "Detenido por un error."


# --------------------------------------------------------------------------- panel

def hex_color(rgb):
    return "#%02x%02x%02x" % tuple(int(c) for c in rgb)


class App:
    PREVIEW_W = 800

    def __init__(self, root, state, cams, args):
        self.root, self.s, self.cams, self.args = root, state, cams, args
        root.title("Cámara Belleza")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.resizable(False, False)

        main = ttk.Frame(root, padding=10)
        main.grid()
        left = ttk.Frame(main)  # vista previa + ajustes de imagen
        left.grid(row=0, column=0, sticky="n")
        self.preview = ttk.Label(left)
        self.preview.grid(row=0, column=0, sticky="n")

        panel = ttk.Frame(main, padding=(14, 0, 0, 0))
        panel.grid(row=0, column=1, sticky="n")

        # Interruptor general (siempre visible)
        self.enabled_var = tk.BooleanVar(value=state.enabled)
        ttk.Checkbutton(panel, text="Filtros de belleza activados", variable=self.enabled_var,
                        command=self.on_change).grid(sticky="w")

        # Aviso bien visible mientras la cámara virtual emite el bucle y no tu cámara en vivo.
        self.banner = tk.Label(panel, text="▶ Reproduciendo el bucle: no es la cámara en vivo",
                               bg="#b00020", fg="white", font=("Segoe UI", 9, "bold"),
                               padx=8, pady=4)
        self.banner.grid(row=1, column=0, sticky="we", pady=(6, 0))  # fila fija: si no, al
        self.banner.grid_remove()                                    # ocultarlo Tk la reutiliza

        # Pestañas: así el panel no crece sin límite al añadir opciones.
        self.nb = nb = ttk.Notebook(panel)
        nb.grid(row=2, column=0, sticky="we", pady=(8, 0))
        tab_cam, tab_skin, tab_makeup, tab_loop = (ttk.Frame(nb, padding=10) for _ in range(4))
        nb.add(tab_cam, text="Cámara")
        nb.add(tab_skin, text="Piel")
        nb.add(tab_makeup, text="Maquillaje")
        nb.add(tab_loop, text="Bucle")
        for tab in (tab_cam, tab_skin, tab_makeup, tab_loop):
            tab.columnconfigure(0, weight=1)  # los deslizadores ocupan todo el ancho
        self.swatches, self.color_lbls = {}, {}  # por tipo: "lips" / "blush"

        # -- Cámara
        ttk.Label(tab_cam, text="Cámara", font=("Segoe UI", 10, "bold")).grid(sticky="w")
        self.cam_var = tk.StringVar()
        self.combo = ttk.Combobox(tab_cam, textvariable=self.cam_var, state="readonly", width=34)
        self.combo.grid(sticky="we", pady=(2, 0))
        self.combo.bind("<<ComboboxSelected>>", self.on_camera)
        ttk.Button(tab_cam, text="Actualizar lista", command=self.refresh_cams).grid(
            sticky="w", pady=(4, 8))

        ttk.Label(tab_cam, text="Resolución", font=("Segoe UI", 10, "bold")).grid(sticky="w")
        self.res_var = tk.StringVar(value=self.res_label(state.res))
        self.res_combo = ttk.Combobox(tab_cam, textvariable=self.res_var, state="readonly",
                                      values=list(RESOLUTIONS), width=34)
        self.res_combo.grid(sticky="we", pady=(2, 12))
        self.res_combo.bind("<<ComboboxSelected>>", self.on_resolution)

        self.mirror_var = tk.BooleanVar(value=state.mirror_preview)
        ttk.Checkbutton(tab_cam, text="Espejar vista previa (solo aquí,\nno en la reunión)",
                        variable=self.mirror_var, command=self.on_change).grid(sticky="w")

        # -- Piel
        ttk.Label(tab_skin, text="Alisado de piel", font=("Segoe UI", 10, "bold")).grid(
            sticky="w")
        self.smooth_var = tk.DoubleVar(value=state.smooth * 100)
        ttk.Scale(tab_skin, from_=0, to=100, variable=self.smooth_var,
                  command=lambda _=None: self.on_change()).grid(sticky="we", pady=(2, 14))

        # -- Maquillaje: labial
        ttk.Label(tab_makeup, text="Intensidad del labial", font=("Segoe UI", 10, "bold")).grid(
            sticky="w")
        self.lips_var = tk.DoubleVar(value=state.lips * 100)
        ttk.Scale(tab_makeup, from_=0, to=100, variable=self.lips_var,
                  command=lambda _=None: self.on_change()).grid(sticky="we", pady=(2, 8))
        self._color_picker(tab_makeup, "lips", "Color del labial", LIP_PRESETS, bottom=12)

        # -- Maquillaje: rubor
        ttk.Label(tab_makeup, text="Intensidad del rubor", font=("Segoe UI", 10, "bold")).grid(
            sticky="w")
        self.blush_var = tk.DoubleVar(value=state.blush * 100)
        ttk.Scale(tab_makeup, from_=0, to=100, variable=self.blush_var,
                  command=lambda _=None: self.on_change()).grid(sticky="we", pady=(2, 8))
        ttk.Label(tab_makeup, text="Posición del rubor", font=("Segoe UI", 10, "bold")).grid(
            sticky="w")
        self.blush_style_var = tk.StringVar(value=state.blush_style)
        style_combo = ttk.Combobox(tab_makeup, textvariable=self.blush_style_var,
                                   state="readonly", values=list(BLUSH_STYLES), width=34)
        style_combo.grid(sticky="we", pady=(2, 8))
        style_combo.bind("<<ComboboxSelected>>", lambda _=None: self.on_change())
        self._color_picker(tab_makeup, "blush", "Color del rubor", BLUSH_PRESETS, bottom=4)

        # -- Bucle: grabar unos segundos con la cámara (o cargar un video) y emitirlos en bucle
        ttk.Label(tab_loop, text="Grabar un bucle con la cámara",
                  font=("Segoe UI", 10, "bold")).grid(sticky="w")
        rec_row = ttk.Frame(tab_loop)
        rec_row.grid(sticky="w", pady=(4, 2))
        self.rec_var = tk.StringVar(value=f"{state.rec_seconds} s")
        rec_combo = ttk.Combobox(rec_row, textvariable=self.rec_var, state="readonly", width=6,
                                 values=[f"{s} s" for s in REC_OPTIONS])
        rec_combo.pack(side="left")
        rec_combo.bind("<<ComboboxSelected>>", lambda _=None: self.on_change())
        self.rec_btn = ttk.Button(rec_row, text="● Grabar", command=self.start_record)
        self.rec_btn.pack(side="left", padx=8)
        self.rec_lbl = ttk.Label(tab_loop, text="", foreground="#555")
        self.rec_lbl.grid(sticky="w")
        ttk.Separator(tab_loop).grid(sticky="we", pady=8)
        files_row = ttk.Frame(tab_loop)
        files_row.grid(sticky="w")
        self.load_btn = ttk.Button(files_row, text="Cargar video…", command=self.load_video)
        self.load_btn.pack(side="left")
        self.clear_btn = ttk.Button(files_row, text="Borrar bucle", command=self.clear_loop)
        self.clear_btn.pack(side="left", padx=8)
        self.pingpong_var = tk.BooleanVar(value=state.loop_pingpong)
        ttk.Checkbutton(tab_loop, text="Ida y vuelta (evita el salto al repetir)",
                        variable=self.pingpong_var, command=self.on_change).grid(
            sticky="w", pady=(10, 0))
        self.loop_play_var = tk.BooleanVar(value=False)  # nunca arranca solo
        self.play_chk = ttk.Checkbutton(
            tab_loop, text="Reproducir el bucle en lugar\nde la cámara en vivo",
            variable=self.loop_play_var, command=self.on_change)
        self.play_chk.grid(sticky="w", pady=(8, 0))
        self.loop_lbl = ttk.Label(tab_loop, text="", foreground="#555", wraplength=260,
                                  justify="left")
        self.loop_lbl.grid(sticky="w", pady=(10, 0))
        ttk.Label(tab_loop, foreground="#555", wraplength=260, justify="left", text=(
            "Se graba la imagen de la cámara sin filtros; los filtros de belleza y de imagen se "
            "aplican al reproducir, así que puedes ajustarlos después. El bucle no incluye audio: "
            "el micrófono sigue siendo el de siempre. Se guarda en bucle.avi y se carga al abrir "
            "el programa, pero nunca se reproduce solo.")).grid(sticky="w", pady=(6, 0))

        # Ajustes de imagen (bajo la vista previa)
        img = ttk.LabelFrame(left, text="Imagen", padding=(10, 4, 10, 8))
        img.grid(row=1, column=0, sticky="we", pady=(10, 0))
        self.adj = {}  # nombre -> (variable, etiqueta de valor)
        for k, (key, title) in enumerate(ADJUSTMENTS_UI):  # 4 por fila: título + deslizador
            row, col = (k // 4) * 2, k % 4
            img.columnconfigure(col, weight=1)
            head = ttk.Frame(img)
            head.grid(row=row, column=col, sticky="we", padx=6, pady=(6 if row else 0, 0))
            ttk.Label(head, text=title, font=("Segoe UI", 10, "bold")).pack(side="left")
            val = ttk.Label(head, text="0", width=5, anchor="e")
            val.pack(side="right")
            var = tk.DoubleVar(value=getattr(state, key))
            ttk.Scale(img, from_=-100, to=100, variable=var,
                      command=lambda _=None: self.on_change()).grid(
                row=row + 1, column=col, sticky="we", padx=6)
            self.adj[key] = (var, val)
        # El botón ocupa el hueco que queda libre al final de la segunda fila.
        ttk.Button(img, text="Restablecer", command=self.reset_image).grid(
            row=3, column=3, sticky="e", padx=6)

        # Estado
        self.status_lbl = ttk.Label(main, text="", foreground="#555")
        self.status_lbl.grid(row=1, column=1, sticky="sw", pady=(10, 0))
        self.err_lbl = ttk.Label(main, text="", foreground="#b00020", wraplength=300)
        self.err_lbl.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        self.refresh_cams(initial=True)
        for kind in COLOR_KINDS:
            self.set_color(kind, getattr(state, COLOR_KINDS[kind][0]), save=False)
        self.tick()

    # -- cámara
    def refresh_cams(self, initial=False):
        self.cams = list_cameras()
        names = [n for _, n in self.cams]
        self.combo["values"] = names
        if not names:
            self.err_lbl.config(text="No encontré ninguna webcam.")
            return
        current = self.cam_var.get() if not initial else self.saved_camera
        if current not in names:
            current = names[0]
        self.cam_var.set(current)
        self.on_camera()

    def on_camera(self, _=None):
        name = self.cam_var.get()
        for idx, n in self.cams:
            if n == name:
                self.s.cam_index = idx
        self.on_change()

    @staticmethod
    def res_label(res):
        return next((k for k, v in RESOLUTIONS.items() if v == tuple(res)), list(RESOLUTIONS)[1])

    def on_resolution(self, _=None):
        self.s.res = RESOLUTIONS[self.res_var.get()]
        self.on_change()

    # -- colores (labial y rubor)
    def _color_picker(self, parent, kind, title, presets, bottom):
        """Botones de colores + 'Otro…' + etiqueta con el color elegido."""
        ttk.Label(parent, text=title, font=("Segoe UI", 10, "bold")).grid(sticky="w")
        sw = ttk.Frame(parent)
        sw.grid(sticky="w", pady=(4, 0))
        self.swatches[kind] = {}
        for k, (name, rgb) in enumerate(presets.items()):
            b = tk.Button(sw, bg=hex_color(rgb), width=3, height=1, bd=2, relief="raised",
                          activebackground=hex_color(rgb), cursor="hand2",
                          command=lambda c=rgb: self.set_color(kind, c))
            b.grid(row=k // 4, column=k % 4, padx=2, pady=2)
            self.swatches[kind][name] = (b, rgb)
        n = len(presets)
        ttk.Button(sw, text="Otro…", width=6, command=lambda: self.pick_color(kind)).grid(
            row=n // 4, column=n % 4, padx=2, pady=2)
        self.color_lbls[kind] = ttk.Label(parent, text="")
        self.color_lbls[kind].grid(sticky="w", pady=(2, bottom))

    def set_color(self, kind, rgb, save=True):
        attr = COLOR_KINDS[kind][0]
        setattr(self.s, attr, tuple(int(c) for c in rgb))
        cur = getattr(self.s, attr)
        for b, c in self.swatches[kind].values():
            b.config(relief="sunken" if tuple(c) == cur else "raised",
                     bd=4 if tuple(c) == cur else 2)
        name = next((n for n, (_, c) in self.swatches[kind].items() if tuple(c) == cur),
                    "Personalizado")
        self.color_lbls[kind].config(text=f"Seleccionado: {name}")
        if save:
            self.on_change()

    def pick_color(self, kind):
        attr, _, title = COLOR_KINDS[kind]
        rgb, _ = colorchooser.askcolor(color=hex_color(getattr(self.s, attr)), title=title)
        if rgb:
            self.set_color(kind, rgb)

    # -- común
    def on_change(self):
        s = self.s
        s.enabled = self.enabled_var.get()
        s.smooth = round(self.smooth_var.get() / 100, 3)
        s.lips = round(self.lips_var.get() / 100, 3)
        s.blush = round(self.blush_var.get() / 100, 3)
        s.blush_style = self.blush_style_var.get()
        s.mirror_preview = self.mirror_var.get()
        for key, (var, lbl) in self.adj.items():
            v = int(round(var.get()))
            setattr(s, key, v)
            lbl.config(text=f"{v:+d}" if v else "0")
        s.loop_pingpong = self.pingpong_var.get()
        s.loop_play = self.loop_play_var.get()
        s.rec_seconds = int(self.rec_var.get().split()[0])
        s.save(self.cam_var.get())

    def reset_image(self):
        for var, _ in self.adj.values():
            var.set(0)
        self.on_change()

    # -- bucle
    def start_record(self):
        s = self.s
        if s.loop_play or s.rec_progress is not None:
            return
        if s.loop_ready and not messagebox.askyesno(
                "Grabar un bucle nuevo",
                "Ya hay un bucle. ¿Reemplazarlo (y el archivo bucle.avi) por una grabación nueva?"):
            return
        self.on_change()
        s.rec_request = s.rec_seconds  # el worker empieza en el siguiente fotograma

    def load_video(self):
        s = self.s
        if s.loop_play or s.rec_progress is not None:
            return
        path = filedialog.askopenfilename(
            title="Elegir un video para el bucle",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm *.wmv"), ("Todos", "*.*")])
        if not path:
            return
        if s.loop_ready and not messagebox.askyesno(
                "Cargar un video", "Ya hay un bucle. ¿Reemplazarlo por este video?"):
            return
        s.load_request = path

    def clear_loop(self):
        if self.s.loop_ready and messagebox.askyesno(
                "Borrar el bucle", "¿Borrar el bucle y el archivo bucle.avi?"):
            self.s.loop_play = False
            self.loop_play_var.set(False)
            self.s.clear_request = True

    def tick(self):
        s = self.s
        # Si el worker volvió a la cámara anterior, reflejarlo en el desplegable.
        active = next((n for i, n in self.cams if i == s.cam_index), None)
        if active and active != self.cam_var.get():
            self.cam_var.set(active)
        if self.res_label(s.res) != self.res_var.get():
            self.res_var.set(self.res_label(s.res))
        frame = s.latest
        if frame is not None:
            h, w = frame.shape[:2]
            view = cv2.resize(frame, (self.PREVIEW_W, int(self.PREVIEW_W * h / w)),
                              interpolation=cv2.INTER_AREA)
            if s.mirror_preview:
                view = cv2.flip(view, 1)
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB)))
            self.preview.configure(image=img)
            self.preview.image = img
        self.status_lbl.config(text=f"{s.status}\n{s.fps:.0f} fps")
        self.err_lbl.config(text=s.error or "")

        # Bucle: el worker puede apagarlo solo (p. ej. si se borra); aviso y botones al día.
        if self.loop_play_var.get() != s.loop_play:
            self.loop_play_var.set(s.loop_play)
        playing = s.loop_play and s.loop_ready
        recording = s.rec_progress is not None
        if playing:
            self.banner.grid()
        else:
            self.banner.grid_remove()
        self.rec_lbl.config(
            text=f"Grabando… {int(s.rec_progress * 100)} %  (mientras, sale la cámara en vivo)"
            if recording else "")

        def enable(widget, on):
            widget.state(["!disabled"] if on else ["disabled"])
        enable(self.rec_btn, not playing and not recording)
        enable(self.load_btn, not playing and not recording)
        enable(self.clear_btn, s.loop_ready and not recording)
        enable(self.play_chk, s.loop_ready and not recording)
        text = s.loop_info or "No hay ningún bucle."
        self.loop_lbl.config(text=text + (f"\n{s.loop_msg}" if s.loop_msg else ""))
        self.root.after(33, self.tick)

    def close(self):
        self.s.stop.set()
        self.root.after(50, self.root.destroy)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--list", action="store_true", help="listar cámaras y salir")
    ap.add_argument("--selftest", type=float, metavar="SEG",
                    help="abrir el panel SEG segundos, comprobar que hay imagen y salir")
    args = ap.parse_args()

    if not MODEL.exists():
        raise SystemExit(f"Falta el modelo {MODEL.name}. Ver README.md.")
    cams = list_cameras()
    if args.list or not cams:
        for i, n in cams:
            print(f"{i}: {n}")
        raise SystemExit(0 if cams else "No encontré ninguna webcam.")

    state = State()
    saved = state.load()
    root = tk.Tk()
    App.saved_camera = saved.get("camera")
    app = App(root, state, cams, args)
    if BUCLE_PATH.exists():  # recarga el último bucle, pero NO lo reproduce
        state.load_request = str(BUCLE_PATH)

    t = threading.Thread(target=worker, args=(state, args.fps), daemon=True)
    t.start()

    if args.selftest:
        def check():
            ok = state.latest is not None
            print(f"selftest: imagen={ok} fps={state.fps:.1f} status={state.status!r} "
                  f"error={state.error!r}")
            app.close()
        root.after(int(args.selftest * 1000), check)

    root.mainloop()
    state.stop.set()
    t.join(timeout=3)


if __name__ == "__main__":
    main()
