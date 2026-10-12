## Entorno
- Python 3.12.10
- PyTorch 2.11.0 + CUDA 12.8
- Ultralytics 8.4.172
- GPU: NVIDIA GeForce RTX 5060 (8 GB)
- Dataset: Roboflow Universe Projects, License Plate Recognition v12 (CC BY 4.0)


## Entrenamientos

| Versión | Fecha | Dataset | Modelo | Parámetros | Épocas | Tiempo | P | R | mAP50 | mAP50-95 | Observaciones |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1_roboflow_n640 | 03/10/2026 | Roboflow v12 | yolov8n | imgsz=640, batch=16, patience=20 | 91 (mejor: 71) | 1,27 h | 0,987 | 0,950 | 0,972 | 0,692 | Métricas en validación Roboflow. Early stopping en época 91. |

| v1_roboflow_n640 (test) | 03/10/2026 | Roboflow v12 – test | yolov8n | – | – | – | 0,994 | 0,946 | 0,963 | 0,697 | Evaluación sobre test Roboflow (1.020 imágenes no vistas). Consistente con validación. |

## OCR

| Versión | Fecha | Conjunto | OCR | Preprocesamiento | Exactitud | Observaciones |
|---|---|---|---|---|---|---|
| ocr_v1 | 04/10/2026 | desarrollo (9 imgs) | EasyOCR (en) | gris, x3, CLAHE, margen 5% | 5/9 (56%) | Orden por posición y filtro por altura corrigen "CHILE" y el orden. Fallos restantes: caracteres extra pegados (bordes o emblema separador). |


## Conjuntos de datos reales

- Origen: 250 capturas (imagen completa + recorte) de cámara de acceso, patente según nombre del archivo (lectura del sistema LPR de referencia).
- 207 vehículos distintos; 6 motos excluidas (fuera de alcance).
- Desarrollo: 30 vehículos (OCR). Test: 174 vehículos, revisados a mano.
- Revisión del test: 1 descartada por ilegible; 5 duplicados del mismo vehículo con lecturas distintas del sistema de referencia (confusiones S/G, R/P, 2/7), detectados por patentes que difieren en un carácter.
- La marca, el color y la pista registrados por el sistema de referencia no son confiables (el mismo vehículo aparece con marcas distintas).

| v1_roboflow_n640 (test real) | 11/10/2026 | Imágenes reales cámara de acceso (173 imgs, 182 patentes) | yolov8n | – | – | – | 1,000 | 0,982 | 0,995 | 0,598 | Primera evaluación con imágenes reales etiquetadas a mano. Detección resuelta sin reentrenar; mAP50-95 menor por diferencias en el ajuste de bordes. |  