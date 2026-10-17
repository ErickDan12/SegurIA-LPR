import argparse
import csv
import re
import time
from pathlib import Path

import cv2
from fast_plate_ocr import LicensePlateRecognizer
from ultralytics import YOLO

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
    if len(texto) != 6:
        return None
    nueva = "".join(A_LETRA.get(c, c) for c in texto[:4]) + \
            "".join(A_NUMERO.get(c, c) for c in texto[4:])
    if RE_NUEVA.match(nueva):
        return nueva
    antigua = "".join(A_LETRA.get(c, c) for c in texto[:2]) + \
              "".join(A_NUMERO.get(c, c) for c in texto[2:])
    if RE_ANTIGUA.match(antigua):
        return antigua
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default="modelos/v1_roboflow_n640_ncnn_model")
    parser.add_argument("--carpeta", default="datos/desarrollo")
    args = parser.parse_args()

    carpeta = Path(args.carpeta)
    temporal = Path("recorte_tmp.jpg")

    print(f"Cargando modelo: {args.modelo}")
    detector = YOLO(args.modelo, task="detect")
    ocr = LicensePlateRecognizer("cct-s-v2-global-model")

    with open(carpeta / "desarrollo_patentes.csv", encoding="utf-8-sig") as f:
        reales = {fila["archivo"]: fila["patente"].strip()
                  for fila in csv.DictReader(f, delimiter=";")}

    t_det, t_ocr, aciertos = [], [], 0
    for n, (nombre, real) in enumerate(reales.items()):
        imagen = cv2.imread(str(carpeta / nombre))

        inicio = time.perf_counter()
        resultado = detector(imagen, imgsz=640, verbose=False)[0]
        fin_det = time.perf_counter()

        patente = None
        if len(resultado.boxes) > 0:
            areas = resultado.boxes.xywh[:, 2] * resultado.boxes.xywh[:, 3]
            i = int(areas.argmax())
            x1, y1, x2, y2 = resultado.boxes.xyxy[i].tolist()
            mx, my = (x2 - x1) * MARGEN, (y2 - y1) * MARGEN
            alto, ancho = imagen.shape[:2]
            x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
            x2, y2 = min(ancho, int(x2 + mx)), min(alto, int(y2 + my))
            cv2.imwrite(str(temporal), imagen[y1:y2, x1:x2])
            prediccion = ocr.run(str(temporal))[0]
            texto = prediccion.plate if hasattr(prediccion, "plate") else str(prediccion)
            patente = corregir_y_validar(limpiar_texto(texto))
        fin_ocr = time.perf_counter()

        ms_det, ms_ocr = (fin_det - inicio) * 1000, (fin_ocr - fin_det) * 1000
        if n > 0:  # la primera imagen incluye la carga inicial y no se cuenta
            t_det.append(ms_det)
            t_ocr.append(ms_ocr)
        ok = patente == real
        aciertos += ok
        print(f"{real:<7} -> {str(patente):<7} {'OK' if ok else '--'}  "
              f"detección {ms_det:6.0f} ms | OCR {ms_ocr:5.0f} ms")

    det, lec = sum(t_det) / len(t_det), sum(t_ocr) / len(t_ocr)
    total = len(reales)
    print(f"\nExactitud: {aciertos}/{total} = {aciertos / total:.0%}")
    print(f"Tiempo promedio por imagen: detección {det:.0f} ms + OCR {lec:.0f} ms = {det + lec:.0f} ms")
    print(f"Equivale a {1000 / (det + lec):.1f} imágenes por segundo")


if __name__ == "__main__":
    main()