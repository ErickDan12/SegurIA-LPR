import csv
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

ORIGEN = Path("datos_ppu_chile/1_exportadas")
DESARROLLO = Path("datos_ppu_chile/desarrollo")
TEST = Path("datos_ppu_chile/test")
RESERVA = Path("datos_ppu_chile/2_seleccionadas")
CSV_DESARROLLO = DESARROLLO / "desarrollo_patentes.csv"
CSV_TEST = TEST / "test_patentes.csv"

N_DESARROLLO_NUEVAS = 21   # para completar ~30 junto a las 9 actuales
N_TEST = 200
SEMILLA = 42               # hace que el reparto aleatorio sea repetible

EXTENSIONES = {".jpg", ".jpeg", ".png"}
PATRON_PATENTE = re.compile(r"-([A-Z0-9]+)_CL\d")
FORMATO_CHILENO = re.compile(r"^([BCDFGHJKLPRSTVWXYZ]{4}\d{2}|[A-Z]{2}\d{4})$")


def leer_csv(ruta):
    if not ruta.exists():
        return []
    with open(ruta, encoding="utf-8-sig") as f:
        return [(fila["archivo"].strip(), fila["patente"].strip())
                for fila in csv.DictReader(f, delimiter=";") if fila.get("archivo")]


def escribir_csv(ruta, filas):
    with open(ruta, "w", encoding="utf-8", newline="") as f:
        f.write("archivo;patente\n")
        for archivo, patente in filas:
            f.write(f"{archivo};{patente}\n")


def main():
    # 1. Agrupar cada imagen completa con su recorte _cut
    capturas = defaultdict(dict)
    for ruta in ORIGEN.rglob("*"):
        if ruta.suffix.lower() not in EXTENSIONES:
            continue
        es_cut = ruta.stem.endswith("_cut")
        base = ruta.stem[:-4] if es_cut else ruta.stem
        capturas[base]["cut" if es_cut else "completa"] = ruta

    # 2. Agrupar por patente (una lista de capturas por vehículo)
    por_patente = defaultdict(list)
    sin_patente = []
    for base, partes in capturas.items():
        m = PATRON_PATENTE.search(base)
        if m and "completa" in partes:
            por_patente[m.group(1)].append(partes)
        else:
            sin_patente.append(base)

    # 3. Excluir patentes ya usadas en desarrollo
    filas_dev = leer_csv(CSV_DESARROLLO)
    usadas = {patente for _, patente in filas_dev}

    validas = sorted(p for p in por_patente if p not in usadas and FORMATO_CHILENO.match(p))
    no_validas = sorted(p for p in por_patente if not FORMATO_CHILENO.match(p))

    random.seed(SEMILLA)
    random.shuffle(validas)
    nuevas_dev = validas[:N_DESARROLLO_NUEVAS]
    test = validas[N_DESARROLLO_NUEVAS:N_DESARROLLO_NUEVAS + N_TEST]
    reserva = validas[N_DESARROLLO_NUEVAS + N_TEST:] + no_validas

    # 4. Copiar archivos (una captura por vehículo: la completa y su _cut)
    def copiar(patentes, destino):
        destino.mkdir(parents=True, exist_ok=True)
        filas = []
        for patente in patentes:
            partes = por_patente[patente][0]
            for ruta in partes.values():
                shutil.copy2(ruta, destino / ruta.name)
            filas.append((partes["completa"].name, patente))
        return filas

    filas_dev += copiar(nuevas_dev, DESARROLLO)
    filas_test = copiar(test, TEST)
    copiar(reserva, RESERVA)

    escribir_csv(CSV_DESARROLLO, filas_dev)
    escribir_csv(CSV_TEST, filas_test)

    # 5. Resumen
    print(f"Capturas encontradas: {len(capturas)}")
    print(f"Vehículos distintos (patentes): {len(por_patente)}")
    print(f"Sin patente reconocible en el nombre o sin imagen completa: {len(sin_patente)}")
    print(f"Desarrollo: {len(filas_dev)} (9 anteriores + {len(nuevas_dev)} nuevas)")
    print(f"Test: {len(filas_test)}")
    print(f"Reserva: {len(reserva)}")
    if no_validas:
        print(f"\nPatentes fuera del formato chileno (quedaron en reserva): {', '.join(no_validas)}")


if __name__ == "__main__":
    main()