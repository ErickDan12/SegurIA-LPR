"""Visor web de prueba: video en vivo con las detecciones, patentes confirmadas y envío al backend."""
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
from fast_plate_ocr import LicensePlateRecognizer
from flask import Flask, Response, jsonify
from ultralytics import YOLO

from lpr_comun import BASE, LectorCamara, LectorCamaraFFmpeg, leer_patente, url_camara
import logging
logging.getLogger("werkzeug").setLevel(logging.ERROR)

MODELO = BASE / "modelos" / "v1_roboflow_n640_ncnn_model"
STREAM = "stream2"
USAR_FFMPEG = False      # con stream2 basta el lector de OpenCV
ENVIAR_AL_BACKEND = True
ANCHO_VISOR = 640
FPS_VISOR = 3
TEMPORAL = Path("/dev/shm/recorte_tmp.jpg")  # en RAM: evita desgastar la microSD
CONFIRMACIONES = 2       # lecturas iguales seguidas para validar una patente
ESPERA_REPETIDA = 30     # segundos sin ver la patente antes de volver a informarla
CONF_MIN_OCR = 90        # bajo esto la lectura se marca BAJA y no se envía al backend
COLORES = {"autorizado": "#22c55e", "visita": "#3b82f6", "denegado": "#ef4444"}


class Estado:
    """Datos compartidos entre el análisis y la página."""

    def __init__(self):
        self.lock = threading.Lock()
        self.caja = None
        self.texto_caja = ""
        self.color = (0, 200, 255)
        self.hora_caja = 0.0
        self.lectura = {}
        self.eventos = deque(maxlen=20)


estado = Estado()
camara = None
backend = None

def parecida(a, b, max_diferencias=2):
    """True si dos patentes del mismo largo difieren en pocos caracteres (posible error de lectura)."""
    return len(a) == len(b) and sum(x != y for x, y in zip(a, b)) <= max_diferencias

def procesar():
    """Hilo de análisis: lee patentes del último cuadro y aplica confirmación y espera."""
    detector = YOLO(str(MODELO), task="detect")
    ocr = LicensePlateRecognizer("cct-s-v2-global-model")
    candidata, conteo, ultima_vez = None, 0, {}
    print("Modelos cargados, analizando...")

    while True:
        cuadro = camara.ultimo()
        if cuadro is None:
            time.sleep(0.1)
            continue

        inicio = time.perf_counter()
        patente, crudo, caja, confianza = leer_patente(detector, ocr, cuadro, TEMPORAL)
        ms = (time.perf_counter() - inicio) * 1000
        ahora = time.time()
        confirmada = False

        # Lectura dudosa: se muestra en el visor, pero no cuenta para confirmar
        dudosa = patente is not None and (confianza is None or confianza < CONF_MIN_OCR)

        if patente is None or dudosa:
            candidata, conteo = None, 0
            if caja is None:
                time.sleep(0.5)  # sin patente a la vista: analizar menos seguido
        elif patente in ultima_vez and ahora - ultima_vez[patente] < ESPERA_REPETIDA:
            ultima_vez[patente] = ahora  # ya informada y sigue a la vista
            confirmada = True
        else:
            if patente == candidata:
                conteo += 1
            else:
                candidata, conteo = patente, 1
            if conteo >= CONFIRMACIONES:
                candidata, conteo = None, 0
                confirmada = True
                similar = next((p for p, t in ultima_vez.items()
                                if ahora - t < ESPERA_REPETIDA and parecida(p, patente)), None)
                if similar:
                    # Variante de una patente recién informada: mismo vehículo, no se envía
                    ultima_vez[similar] = ahora
                    print(f"  {patente} descartada: parecida a {similar}, ya informada")
                else:
                    ultima_vez[patente] = ahora
                    resultado = backend.registrar_deteccion(patente, confianza, cuadro, caja) if backend else "sin backend"
                    evento = {"patente": patente, "confianza": confianza, "resultado": resultado,
                              "hora": datetime.now().isoformat(timespec="seconds")}
                    print(f"[{evento['hora']}] PATENTE DETECTADA: {patente} ({confianza}%) -> {resultado}")
                    with estado.lock:
                        estado.eventos.appendleft(evento)

        with estado.lock:
            estado.lectura = {"crudo": crudo or "", "patente": patente or "", "confianza": confianza,
                              "ms": round(ms), "hora": datetime.now().strftime("%H:%M:%S")}
            if caja:
                estado.caja = caja
                estado.texto_caja = f"{patente or crudo} {'OK' if confirmada else ('BAJA' if dudosa else '?')}"
                estado.color = (0, 200, 0) if confirmada else (0, 200, 255)
                estado.hora_caja = ahora

def generar_video():
    """Envía el video en vivo (MJPEG) con la última caja dibujada encima."""
    while True:
        cuadro = camara.ultimo()
        if cuadro is None:
            time.sleep(0.1)
            continue
        cuadro = cuadro.copy()  # dibujar sobre una copia, no sobre el cuadro compartido
        with estado.lock:
            caja, texto, color, hora = estado.caja, estado.texto_caja, estado.color, estado.hora_caja
        if caja and time.time() - hora < 1.5:
            x1, y1, x2, y2 = caja
            cv2.rectangle(cuadro, (x1, y1), (x2, y2), color, 2)
            cv2.putText(cuadro, texto, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        alto, ancho = cuadro.shape[:2]
        if ancho != ANCHO_VISOR:
            cuadro = cv2.resize(cuadro, (ANCHO_VISOR, int(alto * ANCHO_VISOR / ancho)))
        ok, jpg = cv2.imencode(".jpg", cuadro, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
        time.sleep(1 / FPS_VISOR)


PAGINA = """<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<title>SegurIA LPR - Visor de prueba</title>
<style>
  body { font-family: sans-serif; background: #111; color: #eee; margin: 16px; }
  .contenedor { display: flex; gap: 16px; flex-wrap: wrap; }
  img { max-width: 100%; border: 1px solid #444; }
  .panel { min-width: 320px; }
  td, th { padding: 4px 10px; text-align: left; }
  .pat { font-family: monospace; font-size: 22px; letter-spacing: 2px; }
  .res { font-weight: bold; text-transform: uppercase; }
</style></head><body>
<h2>SegurIA LPR · Visor de prueba</h2>
<div class="contenedor">
  <img src="/video" width="640">
  <div class="panel">
    <h3>Última lectura</h3>
    <div id="lectura">—</div>
    <h3>Patentes confirmadas</h3>
    <table id="eventos"></table>
    <p style="color:#888;font-size:13px">Resultado preliminar según la copia local. El oficial lo registra el backend.</p>
  </div>
</div>
<script>
const COLORES = """ + str(COLORES).replace("'", '"') + """;
async function actualizar() {
  try {
    const r = await fetch('/estado');
    const d = await r.json();
    const l = d.lectura;
    document.getElementById('lectura').innerHTML = !l.hora ? '—'
      : l.crudo ? `${l.hora} · OCR: <b>${l.crudo}</b> → <span class="pat">${l.patente || 'inválida'}</span> (${l.confianza ?? '–'}% · ${l.ms} ms)`
      : `${l.hora} · sin patente a la vista (${l.ms} ms)`;
    document.getElementById('eventos').innerHTML = '<tr><th>Hora</th><th>Patente</th><th>Resultado</th></tr>' +
      d.eventos.map(e => `<tr><td>${e.hora.slice(11)}</td><td class="pat">${e.patente}</td>` +
        `<td class="res" style="color:${COLORES[e.resultado] || '#aaa'}">${e.resultado}</td></tr>`).join('');
  } catch (e) {}
}
setInterval(actualizar, 1000);
actualizar();
</script>
</body></html>"""

app = Flask(__name__)


@app.route("/")
def inicio():
    return PAGINA


@app.route("/video")
def video():
    return Response(generar_video(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/estado")
def ver_estado():
    with estado.lock:
        return jsonify(lectura=estado.lectura, eventos=list(estado.eventos))


if __name__ == "__main__":
    if ENVIAR_AL_BACKEND:
        from integracion import ClienteBackend
        backend = ClienteBackend()

    print(f"Conectando a la cámara ({STREAM})...")
    lector = LectorCamaraFFmpeg if USAR_FFMPEG else LectorCamara
    camara = lector(url_camara(STREAM))
    if not camara.activo:
        print("No se pudo conectar. Revisa camara.ini y que la cámara esté encendida.")
        raise SystemExit(1)
    threading.Thread(target=procesar, daemon=True).start()
    print("Abre en tu navegador: http://seguria-pi.local:5000  (Ctrl+C para detener)")
    app.run(host="0.0.0.0", port=5000, threaded=True)