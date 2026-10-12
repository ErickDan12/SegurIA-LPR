import csv
from itertools import combinations
from pathlib import Path

CSVS = [Path("datos_ppu_chile/desarrollo/desarrollo_patentes.csv"),
        Path("datos_ppu_chile/test/test_patentes.csv")]

filas = []
for ruta in CSVS:
    with open(ruta, encoding="utf-8-sig") as f:
        for fila in csv.DictReader(f, delimiter=";"):
            filas.append((ruta.parent.name, fila["archivo"], fila["patente"].strip()))

def difieren_en_uno(a, b):
    return len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1

encontrados = 0
for (c1, a1, p1), (c2, a2, p2) in combinations(filas, 2):
    if p1 == p2 or difieren_en_uno(p1, p2):
        encontrados += 1
        print(f"{p1} ({c1})  <->  {p2} ({c2})")
        print(f"    {a1}\n    {a2}")

print(f"\nPares sospechosos: {encontrados}")