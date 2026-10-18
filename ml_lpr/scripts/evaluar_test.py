import csv
import re
import time
from pathlib import Path

import cv2
from fast_plate_ocr import LicensePlateRecognizer
from ultralytics import YOLO

# --- Pipeline fijado antes de evaluar ---
MODELO_YOLO = "modelos/v1_roboflow_n640/v1_roboflow_n640.pt"
MODELO_OCR = "cct-s-v2-global-model"
CARPETA = Path("datos_ppu_chile/test")
CSV_REAL = CARPETA / "test_patentes.csv"
TEMPORAL = Path("runs/evaluacion_test")
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


def leer(ocr, ruta):
    prediccion = ocr.run(str(ruta))[0]
    texto = prediccion.plate if hasattr(prediccion, "plate") else str(prediccion)
    return limpiar_texto(texto)


def main():
    TEMPORAL.mkdir(parents=True, exist_ok=True)
    detector = YOLO(MODELO_YOLO)
    ocr = LicensePlateRecognizer(MODELO_OCR)

    with open(CSV_REAL, encoding="utf-8-sig") as f:
        reales = {fila["archivo"]: fila["patente"].strip()
                  for fila in csv.DictReader(f, delimiter=";")}

    conteo = {"correcta": 0, "sin_deteccion": 0, "formato_invalido": 0, "lectura_incorrecta": 0}
    ocr_solo_ok, ocr_solo_total = 0, 0
    fallos, tiempos = [], []

    for nombre, real in reales.items():
        imagen = cv2.imread(str(CARPETA / nombre))

        # --- Sistema completo: imagen -> patente ---
        inicio = time.perf_counter()
        resultado = detector(imagen, verbose=False)[0]
        if len(resultado.boxes) == 0:
            conteo["sin_deteccion"] += 1
            fallos.append((real, "sin detección", ""))
        else:
            # Caja más grande = vehículo más cercano a la cámara
            areas = resultado.boxes.xywh[:, 2] * resultado.boxes.xywh[:, 3]
            i = int(areas.argmax())
            x1, y1, x2, y2 = resultado.boxes.xyxy[i].tolist()
            mx, my = (x2 - x1) * MARGEN, (y2 - y1) * MARGEN
            alto, ancho = imagen.shape[:2]
            x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
            x2, y2 = min(ancho, int(x2 + mx)), min(alto, int(y2 + my))

            ruta = TEMPORAL / "recorte.jpg"
            cv2.imwrite(str(ruta), imagen[y1:y2, x1:x2])
            bruto = leer(ocr, ruta)
            patente = corregir_y_validar(bruto)

            if patente == real:
                conteo["correcta"] += 1
            elif patente is None:
                conteo["formato_invalido"] += 1
                fallos.append((real, "formato inválido", bruto))
            else:
                conteo["lectura_incorrecta"] += 1
                fallos.append((real, "lectura incorrecta", patente))
        tiempos.append(time.perf_counter() - inicio)

        # --- OCR por separado, sobre el recorte del sistema de referencia ---
        cortes = list(CARPETA.glob(f"{Path(nombre).stem}_cut.*"))
        if cortes:
            ocr_solo_total += 1
            ocr_solo_ok += corregir_y_validar(leer(ocr, cortes[0])) == real

    total = len(reales)
    tiempo_ms = sum(tiempos[1:]) / max(1, len(tiempos) - 1) * 1000  # sin la primera (carga inicial)

    print(f"\n=== Sistema completo (YOLO + fast-plate-ocr + regex) ===")
    print(f"Correctas: {conteo['correcta']}/{total} = {conteo['correcta'] / total:.1%}")
    print(f"  Sin detección:      {conteo['sin_deteccion']}")
    print(f"  Formato inválido:   {conteo['formato_invalido']}")
    print(f"  Lectura incorrecta: {conteo['lectura_incorrecta']}")
    print(f"Tiempo promedio por imagen (PC): {tiempo_ms:.0f} ms")

    print(f"\n=== OCR por separado (recortes del sistema de referencia) ===")
    print(f"Correctas: {ocr_solo_ok}/{ocr_solo_total} = {ocr_solo_ok / max(1, ocr_solo_total):.1%}")

    if fallos:
        print(f"\n=== Fallos del sistema completo ===")
        for real, motivo, leido in fallos:
            print(f"  {real}: {motivo} (leyó: {leido})")


if __name__ == "__main__":
    main()