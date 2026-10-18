# SegurIA LPR

Proyecto de Título de Ingeniería en Informática (Duoc UC). Sistema de reconocimiento de patentes vehiculares (LPR) para control de acceso en condominios y edificios, con hardware de bajo costo como alternativa a las cámaras LPR comerciales.

## Arquitectura

El sistema se compone de tres partes que se comunican por red:

| Componente | Tecnología | Repositorio |
|---|---|---|
| Reconocimiento de patentes (borde) | Python, YOLOv8, fast-plate-ocr, Raspberry Pi 4 + cámara TP-Link Tapo C310 | Este repositorio (`ml_lpr/`) |
| Backend | Node.js, Express, Socket.io, PostgreSQL | [seguria-lpr-backend](https://github.com/dahyungotica/seguria-lpr-backend) |
| Frontend | HTML, CSS, JavaScript | [seguria-lpr-frontend](https://github.com/dahyungotica/seguria-lpr-frontend) |

```
Cámara (RTSP) → Raspberry Pi: detección + lectura de patentes
                      │
                      ├─ HTTP: registra cada detección (patente, confianza, imagen)
                      └─ Socket.io: recibe la lista de patentes autorizadas del recinto
                      ▼
                   Backend (API REST + tiempo real) ←→ PostgreSQL
                      │
                      ▼
                   Frontend: administrador, propietario y monitor del guardia
```

## Contenido de este repositorio

- `Fase 1/` y `Fase 2/`: documentación del proyecto por etapa.
- `ml_lpr/`: módulo de reconocimiento de patentes.
  - `scripts/`: preparación de datos, evaluación, pipeline para la Raspberry Pi, lectura en vivo y visor web de diagnóstico.
  - `registro_entrenamientos.md`: bitácora con el entorno, cada entrenamiento, las pruebas de OCR y las evaluaciones.
  - `modelos/`: resultados y configuración de cada modelo entrenado. Los pesos del modelo se publican en [Releases](../../releases).

## Pipeline de reconocimiento

1. **Detección (YOLOv8n):** encuentra la patente en la imagen. Entrenado con transfer learning sobre el dataset público *License Plate Recognition* v12 de Roboflow Universe Projects (CC BY 4.0).
2. **Lectura (fast-plate-ocr):** lee el texto del recorte de la patente.
3. **Validación:** verifica el formato chileno (4 letras + 2 números, o 2 letras + 4 números). Si la lectura ya es válida se respeta tal cual; si no, se corrigen confusiones según la posición del carácter.

## Resultados principales

| Evaluación | Resultado |
|---|---|
| Detección en imágenes reales de cámara de acceso (173 imágenes) | mAP50 0,995 · recall 0,982 |
| Sistema completo en imágenes reales (173 imágenes) | 96,5 % de patentes leídas correctamente |
| Raspberry Pi 4 (modelo NCNN) | ~0,5 s por imagen |

El detalle de cada prueba, el análisis de errores y las decisiones están en [`ml_lpr/registro_entrenamientos.md`](ml_lpr/registro_entrenamientos.md).

## Instalación en la Raspberry Pi

Requisitos: Raspberry Pi 4 con Raspberry Pi OS Lite 64-bit, el proyecto en
`~/seguria-lpr` con su entorno virtual, y el modelo exportado a NCNN en
`modelos/`.

### 1. Configuración (archivos fuera del repositorio)

`~/seguria-lpr/camara.ini`: IP, usuario y contraseña de la cámara (cuenta local RTSP).

`~/seguria-lpr/servidor.ini`: credenciales del equipo, creadas por el
administrador del recinto en "Cámaras y equipos":

```ini
[servidor]
url = https://seguria-lpr-backend.onrender.com
identificador = RPI-SEGURIA-01
api_key = <api key entregada por el panel>
```

Ambos archivos deben quedar con `chmod 600`.

### 2. Arranque automático

```bash
sudo cp deploy/seguria-lpr.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now seguria-lpr
```

El servicio arranca al encender el equipo y se reinicia solo si falla.

| Acción | Comando |
|---|---|
| Ver estado | `systemctl status seguria-lpr` |
| Ver registros en vivo | `journalctl -u seguria-lpr -f` |
| Aplicar cambios | `sudo systemctl restart seguria-lpr` |

### 3. Verificación

- Visor de diagnóstico: `http://<ip-raspberry>:5000`.
- En el panel, el equipo aparece en línea y las detecciones llegan al monitor del guardia.

### Funcionamiento sin conexión

La Raspberry Pi mantiene una copia local de las patentes autorizadas y entrega
un resultado preliminar al instante. Si no hay internet, las detecciones quedan
en `cola/` y se envían al reconectar, con su fecha y hora original. El resultado
oficial lo registra el backend.

## Datos y privacidad

- Las imágenes reales usadas para evaluar el sistema se utilizan con autorización, solo con fines de evaluación, y **no se publican** en este repositorio.
- Los datasets, las imágenes, los modelos entrenados y las credenciales de la cámara están excluidos mediante `.gitignore`.

## Autores

- Erick Quinchavil — Machine Learning e integración de hardware.
- Brandon Cayuela — Backend, frontend y modelo de datos.