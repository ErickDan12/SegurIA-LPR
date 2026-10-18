"""Integración con el backend de SegurIA LPR: copia local de patentes, envío de detecciones y heartbeat."""
import base64
import configparser
import json
import threading
import time
import uuid
from datetime import datetime

import cv2
import requests
import socketio

from lpr_comun import BASE

CONFIG_SERVIDOR = BASE / "servidor.ini"
ARCHIVO_PATENTES = BASE / "patentes_local.json"
COLA = BASE / "cola"
COLA_ERRORES = COLA / "errores"
HEARTBEAT_S = 60      # cada cuánto avisa que está vivo
REINTENTO_S = 15      # espera antes de reintentar un envío fallido
TIMEOUT_S = 90        # el plan gratuito de Render puede tardar en "despertar"
ESPACIO = "/dispositivos"


def imagen_compuesta(cuadro, caja, calidad=80):
    """Escena completa con la patente marcada y su recorte ampliado en la esquina superior derecha."""
    imagen = cuadro.copy()
    if caja:
        x1, y1, x2, y2 = caja
        recorte = cuadro[y1:y2, x1:x2]
        if recorte.size:
            alto, ancho = imagen.shape[:2]
            ancho_r = int(ancho * 0.35)
            alto_r = int(recorte.shape[0] * ancho_r / recorte.shape[1])
            if alto_r > alto // 2:  # recorte muy alto: limitar por altura
                alto_r = alto // 2
                ancho_r = int(recorte.shape[1] * alto_r / recorte.shape[0])
            ampliado = cv2.resize(recorte, (ancho_r, alto_r), interpolation=cv2.INTER_CUBIC)
            m = 8
            imagen[m:m + alto_r, ancho - ancho_r - m:ancho - m] = ampliado
            cv2.rectangle(imagen, (ancho - ancho_r - m - 2, m - 2), (ancho - m + 1, m + alto_r + 1), (255, 255, 255), 2)
        cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 200, 0), 2)
    ok, jpg = cv2.imencode(".jpg", imagen, [cv2.IMWRITE_JPEG_QUALITY, calidad])
    return "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()


class ClienteBackend:
    def __init__(self):
        cfg = configparser.ConfigParser()
        cfg.read(CONFIG_SERVIDOR)
        s = cfg["servidor"]
        self.url = s["url"].rstrip("/")
        self.headers = {"X-Dispositivo-Id": s["identificador"], "X-API-Key": s["api_key"]}
        self.auth = {"identificador": s["identificador"], "api_key": s["api_key"]}

        self.lock = threading.Lock()
        self.patentes = self._cargar_copia()
        COLA_ERRORES.mkdir(parents=True, exist_ok=True)
        print(f"[backend] Copia local cargada: {len(self.patentes)} patente(s)")

        self.sio = socketio.Client(reconnection=True, reconnection_delay=5, reconnection_delay_max=60)
        self._registrar_eventos()
        for tarea in (self._conectar_socket, self._enviar_cola, self._latido):
            threading.Thread(target=tarea, daemon=True).start()

    # ----- Copia local de patentes autorizadas -----

    def _cargar_copia(self):
        try:
            filas = json.loads(ARCHIVO_PATENTES.read_text(encoding="utf-8"))
            return {f["patente"]: f for f in filas}
        except Exception:
            return {}

    def _guardar_copia(self):
        with self.lock:
            filas = list(self.patentes.values())
        temporal = ARCHIVO_PATENTES.with_suffix(".tmp")
        temporal.write_text(json.dumps(filas, ensure_ascii=False), encoding="utf-8")
        temporal.replace(ARCHIVO_PATENTES)  # reemplazo atómico: nunca queda un archivo a medias

    def _registrar_eventos(self):
        @self.sio.on("connect", namespace=ESPACIO)
        def conectado():
            print("[backend] Conectado en tiempo real")

        @self.sio.on("disconnect", namespace=ESPACIO)
        def desconectado(*args):
            print("[backend] Desconectado del tiempo real; reintentando...")

        @self.sio.on("patentes:completa", namespace=ESPACIO)
        def completa(datos):
            with self.lock:
                self.patentes = {p["patente"]: p for p in datos.get("patentes", [])}
            self._guardar_copia()
            print(f"[backend] Lista completa recibida: {len(self.patentes)} patente(s)")

        @self.sio.on("patentes:cambios", namespace=ESPACIO)
        def cambios(datos):
            with self.lock:
                for p in datos.get("altas", []):
                    self.patentes[p["patente"]] = p
                for patente in datos.get("bajas", []):
                    self.patentes.pop(patente, None)
            self._guardar_copia()
            print(f"[backend] Cambios: +{len(datos.get('altas', []))} / -{len(datos.get('bajas', []))}")

    def _conectar_socket(self):
        while True:
            try:
                self.sio.connect(self.url, namespaces=[ESPACIO], auth=self.auth, wait_timeout=TIMEOUT_S)
                self.sio.wait()  # bloquea mientras exista la conexión (se reconecta sola)
            except Exception as e:
                print(f"[backend] Sin conexión en tiempo real ({e}); reintento en {REINTENTO_S} s")
            time.sleep(REINTENTO_S)

    def resultado_local(self, patente):
        """Resultado preliminar con la copia local. El oficial lo decide el backend."""
        with self.lock:
            fila = self.patentes.get(patente)
        if not fila:
            return "denegado"
        if fila.get("origen") == "vehiculo":
            return "autorizado"
        hasta = fila.get("vigente_hasta")
        if hasta:
            try:
                if datetime.fromisoformat(hasta.replace("Z", "+00:00")) < datetime.now().astimezone():
                    return "denegado"
            except ValueError:
                pass
        return "visita"

    # ----- Envío de detecciones (con cola en disco) -----

    def registrar_deteccion(self, patente, confianza, cuadro, caja):
        """Guarda la detección en la cola y devuelve el resultado preliminar."""
        datos = {
            "patente": patente,
            "confianza_ocr": confianza,
            "imagen_base64": imagen_compuesta(cuadro, caja),
            "fecha_hora": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        nombre = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.json"
        temporal = COLA / (nombre + ".tmp")
        temporal.write_text(json.dumps(datos), encoding="utf-8")
        temporal.replace(COLA / nombre)
        return self.resultado_local(patente)

    def _enviar_cola(self):
        while True:
            pendientes = sorted(COLA.glob("*.json"))
            if not pendientes:
                time.sleep(1)
                continue
            archivo = pendientes[0]
            try:
                datos = json.loads(archivo.read_text(encoding="utf-8"))
                r = requests.post(f"{self.url}/api/dispositivos/equipo/accesos",
                                  json=datos, headers=self.headers, timeout=TIMEOUT_S)
            except Exception as e:
                print(f"[backend] No se pudo enviar ({e}); se reintentará en {REINTENTO_S} s")
                time.sleep(REINTENTO_S)
                continue

            if r.status_code == 201:
                acceso = r.json()
                print(f"[backend] Registrado: {acceso.get('patente_detectada')} -> {acceso.get('resultado')}")
                archivo.unlink()
            elif r.status_code in (400, 409, 413):
                # Datos rechazados o equipo mal configurado: se aparta para revisarlo, no se reintenta en bucle
                print(f"[backend] Rechazado ({r.status_code}): {r.text[:200]} -> movido a cola/errores")
                archivo.replace(COLA_ERRORES / archivo.name)
            else:
                print(f"[backend] Error {r.status_code}: {r.text[:200]}; se reintentará en {REINTENTO_S} s")
                time.sleep(REINTENTO_S)

    # ----- Señal de vida -----

    def _latido(self):
        while True:
            try:
                requests.post(f"{self.url}/api/dispositivos/equipo/heartbeat",
                              json={}, headers=self.headers, timeout=TIMEOUT_S)
            except Exception as e:
                print(f"[backend] Heartbeat falló ({e})")
            time.sleep(HEARTBEAT_S)