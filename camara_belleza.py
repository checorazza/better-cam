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
from tkinter import colorchooser, messagebox, ttk

import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image, ImageTk

HERE = Path(__file__).parent
MODEL = HERE / "face_landmarker.task"
CONFIG = HERE / "config.json"

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

# Colores de labial (RGB) que aparecen como botones en el panel.
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
ADJUSTMENTS_UI = (("exposure", "Exposición"), ("shadows", "Sombras"), ("highlights", "Luces"),
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
        self.blush = 0.0           # 0-1 (0 = sin rubor)
        self.blush_rgb = BLUSH_PRESETS["Rosa"]
        self.mirror_preview = False
        for k in ADJUSTMENTS:      # ajustes de imagen, -100..100 (0 = sin cambio)
            setattr(self, k, 0)
        self.cam_index = None     # índice DirectShow pedido por el panel
        self.res = (960, 540)      # resolución de captura pedida por el panel
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
        self.mirror_preview = bool(d.get("mirror_preview", self.mirror_preview))
        self.enabled = bool(d.get("enabled", self.enabled))
        self.res = tuple(d.get("res", self.res))
        for k in ADJUSTMENTS:
            setattr(self, k, max(-100, min(100, int(d.get(k, 0)))))
        return d

    def save(self, camera_name):
        try:
            CONFIG.write_text(json.dumps({
                "camera": camera_name, "smooth": self.smooth, "lips": self.lips,
                "lip_rgb": list(self.lip_rgb), "blush": self.blush,
                "blush_rgb": list(self.blush_rgb), "mirror_preview": self.mirror_preview,
                "enabled": self.enabled, "res": list(self.res),
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
    """Exposición, sombras, luces, brillo, contraste y saturación (-100..100; 0 = sin cambio).

    Todo lo tonal se funde en una sola curva de 256 valores (una única pasada con LUT),
    así que apenas cuesta tiempo por fotograma."""

    CURVE_AMOUNT = 0.25  # cuánto mueve las sombras/luces el deslizador al máximo

    def __init__(self):
        self._key = self._lut = None

    @classmethod
    def build_curve(cls, exposure, shadows, highlights, brightness, contrast):
        x = np.arange(256, dtype=np.float64) / 255.0
        if exposure:  # en luz lineal: +-100 = +-2 pasos (EV)
            x = np.clip(x ** 2.2 * 2.0 ** (exposure / 50), 0, 1) ** (1 / 2.2)
        # Máscaras estrechas que valen 0 en negro y en blanco puro (no se "lava" el negro):
        # las sombras actúan sobre todo hacia el 20% de la escala; las luces, hacia el 80%.
        a = cls.CURVE_AMOUNT
        norm = 5 ** 5 / 4 ** 4  # deja el pico de cada máscara en 1
        x = x + a * (shadows / 100) * norm * x * (1 - x) ** 4
        x = x + a * (highlights / 100) * norm * x ** 4 * (1 - x)
        x = np.maximum.accumulate(np.clip(x, 0, 1))  # nunca invertir tonos, pase lo que pase
        x = (x - 0.5) * (1 + contrast / 200) + 0.5 + brightness / 255.0  # contraste x0.5..x1.5
        return np.clip(np.rint(x * 255), 0, 255).astype(np.uint8)

    def __call__(self, frame, s):
        out = frame
        key = (s.exposure, s.shadows, s.highlights, s.brightness, s.contrast)
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

    def _skin(self, frame, pts, strength):
        box = self._face_box(frame, pts)
        if box is None:
            return frame
        x0, y0, x1, y1 = box
        roi = frame[y0:y1, x0:x1]
        mask = self._skin_mask(pts, box)[..., None]

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

    def _blush(self, frame, pts, strength, rgb):
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
        rx, ry = 0.40 * scale, 0.27 * scale    # semiejes de cada mancha

        # Una mancha gaussiana por mejilla, en la "manzana": algo hacia fuera de la pupila
        # y a media altura entre los ojos y la boca; el eje largo sube hacia la sien.
        alpha = np.zeros(roi.shape[:2], np.float32)
        for side in (-1, 1):
            cx, cy = mid + side * 0.62 * u + 0.5 * down
            th = ang - side * 0.35
            c, s_ = math.cos(th), math.sin(th)
            half = int(1.8 * rx) + 1           # fuera de esta ventana el aporte es ~0
            wx0, wx1 = max(int(cx) - half, x0), min(int(cx) + half, x1)
            wy0, wy1 = max(int(cy) - half, y0), min(int(cy) + half, y1)
            if wx1 <= wx0 or wy1 <= wy0:
                continue
            dx = np.arange(wx0, wx1, dtype=np.float32)[None, :] - cx
            dy = np.arange(wy0, wy1, dtype=np.float32)[:, None] - cy
            d2 = ((dx * c + dy * s_) / rx) ** 2 + ((dy * c - dx * s_) / ry) ** 2
            win = alpha[wy0 - y0:wy1 - y0, wx0 - x0:wx1 - x0]
            np.maximum(win, np.exp(-2.0 * d2), out=win)

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
            frame = self._skin(frame, pts, s.smooth)
        if s.blush > 0:
            frame = self._blush(frame, pts, s.blush, s.blush_rgb)
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
        while not state.stop.is_set():
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

            out = adjust(flt.apply(frame), state)
            cam.send(out)
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

        # Cámara
        ttk.Label(panel, text="Cámara", font=("Segoe UI", 10, "bold")).grid(sticky="w")
        self.cam_var = tk.StringVar()
        self.combo = ttk.Combobox(panel, textvariable=self.cam_var, state="readonly", width=34)
        self.combo.grid(sticky="we", pady=(2, 0))
        self.combo.bind("<<ComboboxSelected>>", self.on_camera)
        ttk.Button(panel, text="Actualizar lista", command=self.refresh_cams).grid(
            sticky="w", pady=(4, 8))

        ttk.Label(panel, text="Resolución", font=("Segoe UI", 10, "bold")).grid(sticky="w")
        self.res_var = tk.StringVar(value=self.res_label(state.res))
        self.res_combo = ttk.Combobox(panel, textvariable=self.res_var, state="readonly",
                                      values=list(RESOLUTIONS), width=34)
        self.res_combo.grid(sticky="we", pady=(2, 12))
        self.res_combo.bind("<<ComboboxSelected>>", self.on_resolution)

        # Interruptor general
        self.enabled_var = tk.BooleanVar(value=state.enabled)
        ttk.Checkbutton(panel, text="Filtros de belleza activados", variable=self.enabled_var,
                        command=self.on_change).grid(sticky="w", pady=(0, 10))

        # Piel
        ttk.Label(panel, text="Alisado de piel", font=("Segoe UI", 10, "bold")).grid(sticky="w")
        self.smooth_var = tk.DoubleVar(value=state.smooth * 100)
        ttk.Scale(panel, from_=0, to=100, variable=self.smooth_var,
                  command=lambda _=None: self.on_change()).grid(sticky="we", pady=(2, 12))

        # Labial
        ttk.Label(panel, text="Intensidad del labial", font=("Segoe UI", 10, "bold")).grid(
            sticky="w")
        self.lips_var = tk.DoubleVar(value=state.lips * 100)
        ttk.Scale(panel, from_=0, to=100, variable=self.lips_var,
                  command=lambda _=None: self.on_change()).grid(sticky="we", pady=(2, 8))

        self.swatches, self.color_lbls = {}, {}  # por tipo: "lips" / "blush"
        self._color_picker(panel, "lips", "Color del labial", LIP_PRESETS, bottom=12)

        # Rubor
        ttk.Label(panel, text="Intensidad del rubor", font=("Segoe UI", 10, "bold")).grid(
            sticky="w")
        self.blush_var = tk.DoubleVar(value=state.blush * 100)
        ttk.Scale(panel, from_=0, to=100, variable=self.blush_var,
                  command=lambda _=None: self.on_change()).grid(sticky="we", pady=(2, 8))
        self._color_picker(panel, "blush", "Color del rubor", BLUSH_PRESETS, bottom=12)

        # Vista previa
        self.mirror_var = tk.BooleanVar(value=state.mirror_preview)
        ttk.Checkbutton(panel, text="Espejar vista previa (solo aquí, no en la reunión)",
                        variable=self.mirror_var, command=self.on_change).grid(sticky="w")

        # Ajustes de imagen (bajo la vista previa)
        img = ttk.LabelFrame(left, text="Imagen", padding=(10, 4, 10, 8))
        img.grid(row=1, column=0, sticky="we", pady=(10, 0))
        self.adj = {}  # nombre -> (variable, etiqueta de valor)
        for k, (key, title) in enumerate(ADJUSTMENTS_UI):  # 3 por fila: título + deslizador
            row, col = (k // 3) * 2, k % 3
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
        ttk.Button(img, text="Restablecer", command=self.reset_image).grid(
            row=4, column=0, columnspan=3, sticky="e", pady=(6, 0))

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
        s.mirror_preview = self.mirror_var.get()
        for key, (var, lbl) in self.adj.items():
            v = int(round(var.get()))
            setattr(s, key, v)
            lbl.config(text=f"{v:+d}" if v else "0")
        s.save(self.cam_var.get())

    def reset_image(self):
        for var, _ in self.adj.values():
            var.set(0)
        self.on_change()

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
