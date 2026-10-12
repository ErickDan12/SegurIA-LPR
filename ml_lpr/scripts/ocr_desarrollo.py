import csv
import re
from pathlib import Path
from fast_plate_ocr import LicensePlateRecognizer

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

# --- Configuración ---
MODELO = "modelos/v1_roboflow_n640/v1_roboflow_n640.pt"
CARPETA = Path("datos_ppu_chile/desarrollo")
CSV_REAL = CARPETA / "desarrollo_patentes.csv"
SALIDA_RECORTES = Path("runs/ocr_desarrollo")
MARGEN = 0.05  # agranda un 5% el recorte para no cortar caracteres
PERMITIDOS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Formatos de patente chilena
LETRAS_NUEVAS = "BCDFGHJKLPRSTVWXYZ"
RE_NUEVA = re.compile(rf"^[{LETRAS_NUEVAS}]{{4}}\d{{2}}$")  # BBBB12
RE_ANTIGUA = re.compile(r"^[A-Z]{2}\d{4}$")                  # AB1234

# Correcciones según la posición del carácter
A_NUMERO = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
            "S": "5", "G": "6", "T": "7", "B": "8"}
A_LETRA = {"0": "D", "1": "L", "2": "Z", "5": "S", "6": "G", "8": "B"}


def preprocesar(recorte):
    gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    grande = cv2.resize(gris, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(grande)


def limpiar_componentes(gris):
    """Deja solo las figuras con forma de carácter: quita marco, separadores y la palabra CHILE."""
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    alto, ancho = binaria.shape

    # 1. Borrar las líneas del marco (largas y rectas) para despegarlas de los caracteres
    k_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(0.35 * ancho)), 1))
    k_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, int(0.8 * alto))))
    lineas = cv2.bitwise_or(cv2.morphologyEx(binaria, cv2.MORPH_OPEN, k_horizontal),
                            cv2.morphologyEx(binaria, cv2.MORPH_OPEN, k_vertical))
    binaria = cv2.subtract(binaria, lineas)

    # 2. Separar las figuras y quedarse con las que tienen tamaño de carácter
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    candidatos = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        muy_ancho = w > 0.25 * ancho
        if not muy_ancho and h > 0.2 * alto:
            candidatos.append((i, h))

    if not candidatos:
        return gris

    # Altura de referencia: mediana de las 6 figuras más altas (los caracteres)
    alturas = sorted((h for _, h in candidatos), reverse=True)[:6]
    h_ref = float(np.median(alturas))

    mascara = np.zeros_like(binaria)
    conservados = 0
    for i, h in candidatos:
        if 0.7 * h_ref <= h <= 1.3 * h_ref:
            mascara[etiquetas == i] = 255
            conservados += 1

    if conservados < 4:  # la limpieza falló: mejor usar el original
        return gris

    limpio = cv2.bitwise_not(mascara)  # caracteres negros sobre fondo blanco
    return cv2.copyMakeBorder(limpio, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)


VARIANTES = {
    "base": lambda recorte: preprocesar(recorte),
    "limpio": lambda recorte: limpiar_componentes(preprocesar(recorte)),
}


def limpiar_texto(texto):
    texto = texto.upper().replace("CHILE", "")
    return re.sub(r"[^A-Z0-9]", "", texto)


def corregir_y_validar(texto):
    """Devuelve (patente, formato) o (None, None) si no calza."""
    if len(texto) != 6:
        return None, None
    nueva = "".join(A_LETRA.get(c, c) for c in texto[:4]) + \
            "".join(A_NUMERO.get(c, c) for c in texto[4:])
    if RE_NUEVA.match(nueva):
        return nueva, "nueva"
    antigua = "".join(A_LETRA.get(c, c) for c in texto[:2]) + \
              "".join(A_NUMERO.get(c, c) for c in texto[2:])
    if RE_ANTIGUA.match(antigua):
        return antigua, "antigua"
    return None, None


def leer(lector, imagen):
    """OCR + filtro por altura + orden de izquierda a derecha."""
    detecciones = lector.readtext(imagen, detail=1, allowlist=PERMITIDOS)
    segmentos = []
    for caja, texto, _ in detecciones:
        ys = [p[1] for p in caja]
        xs = [p[0] for p in caja]
        segmentos.append({"texto": texto, "alto": max(ys) - min(ys), "x": min(xs)})
    if not segmentos:
        return ""
    alto_max = max(s["alto"] for s in segmentos)
    principales = sorted((s for s in segmentos if s["alto"] >= 0.6 * alto_max),
                         key=lambda s: s["x"])
    return limpiar_texto("".join(s["texto"] for s in principales))


def main():
    SALIDA_RECORTES.mkdir(parents=True, exist_ok=True)
    modelo = YOLO(MODELO)
    lector = easyocr.Reader(["en"], gpu=True)
    lector_placas = LicensePlateRecognizer("cct-s-v2-global-model")

    with open(CSV_REAL, encoding="utf-8-sig") as f:
        reales = {fila["archivo"]: fila["patente"].strip()
                  for fila in csv.DictReader(f, delimiter=";")}

    aciertos = {v: 0 for v in [*VARIANTES, "combinado", "fastplate"]}
    for nombre, real in reales.items():
        imagen = cv2.imread(str(CARPETA / nombre))
        resultado = modelo(imagen, verbose=False)[0]

        if len(resultado.boxes) == 0:
            print(f"{real:<7} | sin detección")
            continue

        # Toma la caja con mayor confianza
        i = int(resultado.boxes.conf.argmax())
        x1, y1, x2, y2 = resultado.boxes.xyxy[i].tolist()
        mx, my = (x2 - x1) * MARGEN, (y2 - y1) * MARGEN
        alto, ancho = imagen.shape[:2]
        x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
        x2, y2 = min(ancho, int(x2 + mx)), min(alto, int(y2 + my))
        recorte = imagen[y1:y2, x1:x2]

        # Recorte sin procesar, para fast-plate-ocr (hace su propio preprocesamiento)
        ruta_crudo = SALIDA_RECORTES / f"crudo_{nombre}"
        cv2.imwrite(str(ruta_crudo), recorte)

        resultados = {}
        for variante, funcion in VARIANTES.items():
            procesada = funcion(recorte)
            cv2.imwrite(str(SALIDA_RECORTES / f"{variante}_{nombre}"), procesada)
            bruto = leer(lector, procesada)
            patente, _ = corregir_y_validar(bruto)
            resultados[variante] = (bruto, patente)

        # Combinado: usa la base si pasa la validación chilena; si no, la limpia
        resultados["combinado"] = resultados["base"] if resultados["base"][1] else resultados["limpio"]

        # OCR especializado en patentes
        prediccion = lector_placas.run(str(ruta_crudo))[0]
        texto = prediccion.plate if hasattr(prediccion, "plate") else str(prediccion)
        bruto_fp = limpiar_texto(texto)
        resultados["fastplate"] = (bruto_fp, corregir_y_validar(bruto_fp)[0])

        partes = [f"{real:<7}"]
        for variante, (bruto, patente) in resultados.items():
            ok = patente == real
            aciertos[variante] += ok
            partes.append(f"{variante}: {bruto:<9} -> {str(patente):<7} {'OK' if ok else '--'}")
        print(" | ".join(partes))

    total = len(reales)
    print()
    for variante, a in aciertos.items():
        print(f"{variante}: {a}/{total} = {a / total:.0%}")

if __name__ == "__main__":
    main()