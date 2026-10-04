import csv
import re
from pathlib import Path

import cv2
import easyocr
from ultralytics import YOLO

# --- Configuración ---
MODELO = "modelos/v1_roboflow_n640/v1_roboflow_n640.pt"
CARPETA = Path("datos_ppu_chile/desarrollo")
CSV_REAL = CARPETA / "desarrollo_patentes.csv"
SALIDA_RECORTES = Path("runs/ocr_desarrollo")
MARGEN = 0.05  # agranda un 5% el recorte para no cortar caracteres

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


def limpiar(texto):
    texto = texto.upper().replace("CHILE", "")
    return re.sub(r"[^A-Z0-9]", "", texto)


def corregir_y_validar(texto):
    """Devuelve (patente, formato) o (None, None) si no calza."""
    if len(texto) != 6:
        return None, None
    # Intento formato nuevo: 4 letras + 2 números
    nueva = "".join(A_LETRA.get(c, c) for c in texto[:4]) + \
            "".join(A_NUMERO.get(c, c) for c in texto[4:])
    if RE_NUEVA.match(nueva):
        return nueva, "nueva"
    # Intento formato antiguo: 2 letras + 4 números
    antigua = "".join(A_LETRA.get(c, c) for c in texto[:2]) + \
              "".join(A_NUMERO.get(c, c) for c in texto[2:])
    if RE_ANTIGUA.match(antigua):
        return antigua, "antigua"
    return None, None


def main():
    SALIDA_RECORTES.mkdir(parents=True, exist_ok=True)
    modelo = YOLO(MODELO)
    lector = easyocr.Reader(["en"], gpu=True)

    with open(CSV_REAL, encoding="utf-8-sig") as f:
        reales = {fila["archivo"]: fila["patente"].strip()
                  for fila in csv.DictReader(f, delimiter=";")}

    aciertos = 0
    for nombre, real in reales.items():
        imagen = cv2.imread(str(CARPETA / nombre))
        resultado = modelo(imagen, verbose=False)[0]

        if len(resultado.boxes) == 0:
            print(f"{nombre}: sin detección | real: {real}")
            continue

        # Toma la caja con mayor confianza
        i = int(resultado.boxes.conf.argmax())
        x1, y1, x2, y2 = resultado.boxes.xyxy[i].tolist()
        mx, my = (x2 - x1) * MARGEN, (y2 - y1) * MARGEN
        alto, ancho = imagen.shape[:2]
        x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
        x2, y2 = min(ancho, int(x2 + mx)), min(alto, int(y2 + my))

        recorte = preprocesar(imagen[y1:y2, x1:x2])
        cv2.imwrite(str(SALIDA_RECORTES / nombre), recorte)

        # Cada detección del OCR: (caja, texto, confianza)
        detecciones = lector.readtext(recorte, detail=1,
                                      allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        segmentos = []
        for caja, texto, _ in detecciones:
            ys = [p[1] for p in caja]
            xs = [p[0] for p in caja]
            segmentos.append({"texto": texto, "alto": max(ys) - min(ys), "x": min(xs)})

        if segmentos:
            alto_max = max(s["alto"] for s in segmentos)
            # Descarta trozos pequeños (palabra CHILE, emblema) y ordena de izquierda a derecha
            principales = [s for s in segmentos if s["alto"] >= 0.6 * alto_max]
            principales.sort(key=lambda s: s["x"])
            bruto = limpiar("".join(s["texto"] for s in principales))
        else:
            bruto = ""

        patente, formato = corregir_y_validar(bruto)
        ok = patente == real
        aciertos += ok

        detalle = ", ".join(f"{s['texto']}(h={int(s['alto'])})" for s in segmentos)
        print(f"{nombre}: OCR='{bruto}' -> {patente} ({formato}) | real: {real} | {'OK' if ok else 'FALLA'}")
        print(f"    trozos: {detalle}")

    print(f"\nExactitud: {aciertos}/{len(reales)} = {aciertos / len(reales):.0%}")


if __name__ == "__main__":
    main()