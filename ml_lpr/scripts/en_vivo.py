import configparser
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

# RTSP por TCP: imágenes completas, sin paquetes perdidos
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
import cv2
from fast_plate_ocr import LicensePlateRecognizer
from ultralytics import YOLO

# --- Configuración ---
BASE = Path.home() / "seguria-lpr"
CONFIG = BASE / "camara.ini"
MODELO = BASE / "modelos" / "v1_roboflow_n640_ncnn_model"
STREAM = "stream1"
CONF_MIN = 0.5          # confianza mínima de YOLO para considerar una patente
CONFIRMACIONES = 2      # lecturas iguales seguidas para validar una patente
ESPERA_REPETIDA = 30    # segundos antes de volver a informar la misma patente
MARGEN = 0.05
DEPURAR = True          # muestra cada lectura, útil para pruebas

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
    if len(texto) != 6:
        return None
    # 1. Si el OCR ya entregó un formato válido, se respeta tal cual
    if RE_NUEVA.match(texto) or RE_ANTIGUA.match(texto):
        return texto
    # 2. Si no calza con ninguno, se intentan las correcciones por posición
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


def leer_patente(detector, ocr, cuadro, temporal):
    """Detecta, recorta, lee y valida. Devuelve (patente, texto_crudo)."""
    resultado = detector(cuadro, imgsz=640, conf=CONF_MIN, verbose=False)[0]
    if len(resultado.boxes) == 0:
        return None, None
    areas = resultado.boxes.xywh[:, 2] * resultado.boxes.xywh[:, 3]
    i = int(areas.argmax())
    x1, y1, x2, y2 = resultado.boxes.xyxy[i].tolist()
    mx, my = (x2 - x1) * MARGEN, (y2 - y1) * MARGEN
    alto, ancho = cuadro.shape[:2]
    x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    x2, y2 = min(ancho, int(x2 + mx)), min(alto, int(y2 + my))
    cv2.imwrite(str(temporal), cuadro[y1:y2, x1:x2])
    prediccion = ocr.run(str(temporal))[0]
    texto = prediccion.plate if hasattr(prediccion, "plate") else str(prediccion)
    crudo = limpiar_texto(texto)
    return corregir_y_validar(crudo), crudo


def main():
    print("Cargando modelos...")
    detector = YOLO(str(MODELO), task="detect")
    ocr = LicensePlateRecognizer("cct-s-v2-global-model")

    print(f"Conectando a la cámara ({STREAM})...")
    camara = LectorCamara(url_camara(STREAM))
    if not camara.activo:
        print("No se pudo conectar. Revisa camara.ini y que la cámara esté encendida.")
        return

    temporal = BASE / "recorte_tmp.jpg"
    candidata, conteo = None, 0
    ultima_vez = {}
    print("Sistema en marcha. Presiona Ctrl+C para detener.\n")

    try:
        while True:
            cuadro = camara.ultimo()
            if cuadro is None:
                time.sleep(0.1)
                continue

            inicio = time.perf_counter()
            patente, crudo = leer_patente(detector, ocr, cuadro, temporal)
            ms = (time.perf_counter() - inicio) * 1000

            if DEPURAR and crudo is not None:
                print(f"  lectura: {crudo} -> {patente} ({ms:.0f} ms)")

            if patente is None:
                candidata, conteo = None, 0
                continue

            # Confirmación: la misma patente en lecturas seguidas
            ahora = time.time()

            # Si ya se informó y sigue a la vista, solo se actualiza la hora
            if patente in ultima_vez and ahora - ultima_vez[patente] < ESPERA_REPETIDA:
                ultima_vez[patente] = ahora
                continue

            # Confirmación: la misma patente en lecturas seguidas
            if patente == candidata:
                conteo += 1
            else:
                candidata, conteo = patente, 1

            if conteo >= CONFIRMACIONES:
                ultima_vez[patente] = ahora
                candidata, conteo = None, 0
                print(f"[{datetime.now():%H:%M:%S}] PATENTE DETECTADA: {patente}")
    except KeyboardInterrupt:
        print("\nDetenido.")
    finally:
        camara.cerrar()


if __name__ == "__main__":
    main()