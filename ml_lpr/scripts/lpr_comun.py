"""Funciones comunes del pipeline LPR: validación, cámara y lectura de patentes."""
import configparser
import os
import re
import threading
import time
from pathlib import Path
from urllib.parse import quote
import subprocess
import numpy as np

# RTSP por TCP: imágenes completas, sin paquetes perdidos
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
import cv2

BASE = Path.home() / "seguria-lpr"
CONFIG = BASE / "camara.ini"
MARGEN = 0.05

LETRAS_NUEVAS = "BCDFGHJKLPRSTVWXYZ"
RE_NUEVA = re.compile(rf"^[{LETRAS_NUEVAS}]{{4}}\d{{2}}$")
RE_ANTIGUA = re.compile(r"^[A-Z]{2}\d{4}$")
A_NUMERO = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
            "S": "5", "G": "6", "T": "7", "B": "8"}
A_LETRA = {"0": "D", "1": "L", "2": "Z", "5": "S", "6": "G", "8": "B"}


def limpiar_texto(texto):
    texto = texto.upper().replace("CHILE", "")
    return re.sub(r"[^A-Z0-9]", "", texto)


def corregir_y_validar(texto):
    """Validación v2: respeta el texto del OCR si ya es válido; si no, corrige por posición."""
    if len(texto) != 6:
        return None
    if RE_NUEVA.match(texto) or RE_ANTIGUA.match(texto):
        return texto
    nueva = "".join(A_LETRA.get(c, c) for c in texto[:4]) + \
            "".join(A_NUMERO.get(c, c) for c in texto[4:])
    if RE_NUEVA.match(nueva):
        return nueva
    antigua = "".join(A_LETRA.get(c, c) for c in texto[:2]) + \
              "".join(A_NUMERO.get(c, c) for c in texto[2:])
    if RE_ANTIGUA.match(antigua):
        return antigua
    return None


def url_camara(stream):
    cfg = configparser.ConfigParser()
    cfg.read(CONFIG)
    c = cfg["camara"]
    usuario = quote(c["usuario"], safe="")
    clave = quote(c["clave"], safe="")
    return f"rtsp://{usuario}:{clave}@{c['ip']}:554/{stream}"


class LectorCamara:
    """Lee la cámara en un hilo aparte y guarda solo el cuadro más reciente."""

    def __init__(self, url):
        self.url = url
        self.captura = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cuadro = None
        self.lock = threading.Lock()
        self.activo = self.captura.isOpened()
        threading.Thread(target=self._leer, daemon=True).start()

    def _leer(self):
        fallos = 0
        while self.activo:
            ok, cuadro = self.captura.read()
            if ok:
                fallos = 0
                with self.lock:
                    self.cuadro = cuadro
            else:
                fallos += 1
                if fallos >= 50:  # unos 5 segundos sin imagen: reconectar
                    print("Sin imagen de la cámara, reconectando...")
                    self.captura.release()
                    time.sleep(2)
                    self.captura = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                    fallos = 0
                else:
                    time.sleep(0.1)

    def ultimo(self):
        with self.lock:
            return None if self.cuadro is None else self.cuadro.copy()

    def cerrar(self):
        self.activo = False
        self.captura.release()


def leer_patente(detector, ocr, cuadro, temporal, conf_min=0.5):
    """Detecta, recorta, lee y valida. Devuelve (patente, texto_crudo, caja, confianza_ocr)."""
    resultado = detector(cuadro, imgsz=640, conf=conf_min, verbose=False)[0]
    if len(resultado.boxes) == 0:
        return None, None, None, None
    areas = resultado.boxes.xywh[:, 2] * resultado.boxes.xywh[:, 3]
    i = int(areas.argmax())
    x1, y1, x2, y2 = (int(v) for v in resultado.boxes.xyxy[i].tolist())
    caja = (x1, y1, x2, y2)
    mx, my = (x2 - x1) * MARGEN, (y2 - y1) * MARGEN
    alto, ancho = cuadro.shape[:2]
    rx1, ry1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    rx2, ry2 = min(ancho, int(x2 + mx)), min(alto, int(y2 + my))
    cv2.imwrite(str(temporal), cuadro[ry1:ry2, rx1:rx2])

    prediccion = ocr.run(str(temporal), return_confidence=True)[0]
    texto = prediccion.plate
    crudo = limpiar_texto(texto)

    # Confianza del OCR: la del carácter menos seguro (0 a 100)
    confianza = None
    if prediccion.char_probs is not None and texto:
        probs = np.ravel(prediccion.char_probs)[:len(texto)]
        if len(probs) > 0:
            confianza = round(float(probs.min()) * 100, 1)

    return corregir_y_validar(crudo), crudo, caja, confianza

FPS_LECTURA = 10  # cuadros por segundo que FFmpeg entrega a Python


class LectorCamaraFFmpeg:
    """Lee la cámara con FFmpeg y decodificación por hardware; guarda solo el cuadro más reciente."""

    def __init__(self, url):
        self.url = url
        self.crudo = None
        self.proceso = None
        self.lock = threading.Lock()
        self.codec, self.ancho, self.alto = self._medir()
        self.tamano = self.ancho * self.alto * 3 // 2  # bytes de un cuadro YUV420
        self.activo = self.ancho > 0
        if self.activo:
            hw = "hardware" if self.codec == "h264" else "procesador (códec no compatible con hardware)"
            print(f"Cámara: {self.codec} {self.ancho}x{self.alto} | decodificación por {hw}")
            threading.Thread(target=self._leer, daemon=True).start()

    def _medir(self):
        """Consulta el códec y la resolución del stream con ffprobe."""
        comando = ["ffprobe", "-v", "error", "-rtsp_transport", "tcp",
                   "-select_streams", "v:0",
                   "-show_entries", "stream=codec_name,width,height",
                   "-of", "csv=p=0", self.url]
        try:
            salida = subprocess.run(comando, capture_output=True, text=True, timeout=20).stdout
            codec, ancho, alto = salida.strip().splitlines()[0].split(",")
            return codec, int(ancho), int(alto)
        except Exception:
            return None, 0, 0

    def _iniciar(self):
        decodificador = ["-c:v", "h264_v4l2m2m"] if self.codec == "h264" else []
        comando = ["ffmpeg", "-hide_banner", "-loglevel", "error",
                   "-rtsp_transport", "tcp", "-fflags", "nobuffer", "-flags", "low_delay",
                   *decodificador, "-i", self.url,
                   "-an", "-vf", f"fps={FPS_LECTURA}",
                   "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"]
        self.proceso = subprocess.Popen(comando, stdout=subprocess.PIPE)

    def _leer(self):
        while self.activo:
            self._iniciar()
            while self.activo:
                datos = self.proceso.stdout.read(self.tamano)
                if len(datos) < self.tamano:  # se cortó el video
                    break
                with self.lock:
                    self.crudo = datos
            if self.activo:
                print("Sin imagen de la cámara, reconectando...")
                self.proceso.kill()
                time.sleep(2)

    def ultimo(self):
        with self.lock:
            crudo = self.crudo
        if crudo is None:
            return None
        yuv = np.frombuffer(crudo, np.uint8).reshape(self.alto * 3 // 2, self.ancho)
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    def cerrar(self):
        self.activo = False
        if self.proceso:
            self.proceso.kill()