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