import csv
import html
from pathlib import Path

TEST = Path("datos_ppu_chile/test")
CSV_TEST = TEST / "test_patentes.csv"
SALIDA = TEST / "revision_test.html"

with open(CSV_TEST, encoding="utf-8-sig") as f:
    filas = [(fila["archivo"], fila["patente"]) for fila in csv.DictReader(f, delimiter=";")]

bloques = []
for n, (archivo, patente) in enumerate(filas, start=1):
    base = Path(archivo).stem
    recortes = list(TEST.glob(f"{base}_cut.*"))
    imagen = recortes[0].name if recortes else archivo
    bloques.append(
        f'<div class="item"><span class="n">{n}</span>'
        f'<img src="{html.escape(imagen)}">'
        f'<span class="pat">{html.escape(patente)}</span></div>'
    )

pagina = f"""<!doctype html><html><head><meta charset="utf-8"><title>Revisión test</title>
<style>
body {{ font-family: sans-serif; background:#222; color:#eee; }}
.item {{ display:flex; align-items:center; gap:16px; padding:6px; border-bottom:1px solid #444; }}
.n {{ width:40px; color:#888; }}
img {{ height:60px; background:#000; }}
.pat {{ font-size:28px; font-family:monospace; letter-spacing:4px; }}
</style></head><body>
<h2>Test: {len(filas)} patentes. Compara la imagen con el texto.</h2>
{''.join(bloques)}
</body></html>"""

SALIDA.write_text(pagina, encoding="utf-8")
print(f"Página creada: {SALIDA}")