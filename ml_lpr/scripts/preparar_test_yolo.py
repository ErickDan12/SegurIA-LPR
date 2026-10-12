import re
import shutil
from pathlib import Path

EXPORT = Path("datos_ppu_chile/export_labelstudio")
IMAGENES = Path("datos_ppu_chile/test_para_etiquetar")
DESTINO = Path("datos_ppu_chile/test_yolo")

# Identificador único de cada captura, ej: 1414127242315-14-000009-119041
ID_CAPTURA = re.compile(r"(\d{10,}-\d+-\d+-\d+)-")


def obtener_id(nombre):
    m = ID_CAPTURA.search(nombre)
    return m.group(1) if m else None


(DESTINO / "images").mkdir(parents=True, exist_ok=True)
(DESTINO / "labels").mkdir(parents=True, exist_ok=True)

originales = {obtener_id(p.name): p for p in IMAGENES.iterdir() if p.is_file()}
etiquetas = list((EXPORT / "labels").glob("*.txt"))

copiadas, cajas, sin_imagen, ids_etiquetados = 0, 0, [], set()
for txt in etiquetas:
    id_captura = obtener_id(txt.name)
    if id_captura not in originales:
        sin_imagen.append(txt.name)
        continue
    imagen = originales[id_captura]
    shutil.copy2(imagen, DESTINO / "images" / imagen.name)
    shutil.copy2(txt, DESTINO / "labels" / f"{imagen.stem}.txt")
    cajas += sum(1 for linea in txt.read_text().splitlines() if linea.strip())
    ids_etiquetados.add(id_captura)
    copiadas += 1

faltan = sorted(originales[i].name for i in set(originales) - ids_etiquetados)

print(f"Imágenes con etiqueta copiadas: {copiadas}")
print(f"Cajas totales: {cajas}")
if sin_imagen:
    print(f"Etiquetas sin imagen correspondiente: {sin_imagen}")
if faltan:
    print(f"Imágenes sin etiqueta exportada ({len(faltan)}): {faltan}")