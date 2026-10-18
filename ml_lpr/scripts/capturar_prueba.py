import configparser
import os
import time
from pathlib import Path
from urllib.parse import quote

# RTSP por TCP: evita imágenes cortadas o con manchas por paquetes perdidos
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
import cv2

CONFIG = Path.home() / "seguria-lpr" / "camara.ini"


def url_camara(stream):
    cfg = configparser.ConfigParser()
    cfg.read(CONFIG)
    c = cfg["camara"]
    usuario = quote(c["usuario"], safe="")
    clave = quote(c["clave"], safe="")
    return f"rtsp://{usuario}:{clave}@{c['ip']}:554/{stream}"


def probar(stream):
    print(f"\n=== {stream} ===")
    inicio = time.perf_counter()
    captura = cv2.VideoCapture(url_camara(stream), cv2.CAP_FFMPEG)
    if not captura.isOpened():
        print("No se pudo abrir el stream. Revisa la IP, el usuario y la clave en camara.ini")
        return
    print(f"Conexión: {(time.perf_counter() - inicio) * 1000:.0f} ms")

    for _ in range(10):  # los primeros cuadros pueden llegar incompletos
        captura.read()
    ok, cuadro = captura.read()
    captura.release()

    if not ok:
        print("Se conectó, pero no llegó ninguna imagen")
        return
    alto, ancho = cuadro.shape[:2]
    salida = f"captura_{stream}.jpg"
    cv2.imwrite(salida, cuadro)
    print(f"Resolución: {ancho}x{alto} | guardada en {salida}")


for stream in ("stream1", "stream2"):
    probar(stream)