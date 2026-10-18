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
-------------------------------------------------------------------------------------------------------------

## Conjuntos de datos reales

- Origen: 250 capturas (imagen completa + recorte) de cámara de acceso, patente según nombre del archivo (lectura del sistema LPR de referencia).
- 207 vehículos distintos; 6 motos excluidas (fuera de alcance).
- Desarrollo: 30 vehículos (OCR). Test: 174 vehículos, revisados a mano.
- Revisión del test: 1 descartada por ilegible; 5 duplicados del mismo vehículo con lecturas distintas del sistema de referencia (confusiones S/G, R/P, 2/7), detectados por patentes que difieren en un carácter.
- La marca, el color y la pista registrados por el sistema de referencia no son confiables (el mismo vehículo aparece con marcas distintas).
-------------------------------------------------------------------------------------------------------------
| v1_roboflow_n640 (test real) | 11/10/2026 | Imágenes reales cámara de acceso (173 imgs, 182 patentes) | yolov8n | – | – | – | 1,000 | 0,982 | 0,995 | 0,598 | Primera evaluación con imágenes reales etiquetadas a mano. Detección resuelta sin reentrenar; mAP50-95 menor por diferencias en el ajuste de bordes. |  
-------------------------------------------------------------------------------------------------------------
| ocr_v1 | 11/10/2026 | desarrollo (30 imgs) | EasyOCR (en) | gris, x3, CLAHE, margen 5% | 11/30 (37%) | Fallos por bordes, separadores y letra W. |
| ocr_v2 | 11/10/2026 | desarrollo (30 imgs) | EasyOCR (en) | + limpieza por componentes (v2) | 6/30 (20%) | Corrige separadores, pero agrega restos del marco en los extremos. |
| ocr_v3 | 11/10/2026 | desarrollo (30 imgs) | EasyOCR (en) | combinado (base; limpio si base no valida) | 14/30 (47%) | Mejor resultado con EasyOCR. |
| ocr_v4 | 11/10/2026 | desarrollo (30 imgs) | fast-plate-ocr cct-s-v2-global (ONNX, CPU) | recorte crudo, margen 5% | 30/30 (100%) | OCR especializado en patentes. Elegido para el sistema. |
------------------------------------------------------------------------------------
## Evaluación final del sistema completo (test real)

- Pipeline: YOLOv8n v1 (caja más grande) + recorte con margen 5% + fast-plate-ocr cct-s-v2-global (ONNX, CPU) + regex chileno.
- Resultado: 165/173 (95,4%). OCR por separado (recortes del sistema de referencia): 162/173 (93,6%). Tiempo promedio en PC: 29 ms por imagen.
- Resultado en escenas aplicables al proyecto (excluye SFXR46, cuyo objetivo no era el vehículo más cercano): 165/172 (95,9%).
- La lectura casi no pierde exactitud por la detección: el margen de mejora está en la selección de la patente, el OCR y la validación.
- Fallos por categoría (8): selección de vehículo 2, formato antiguo 2, confusión de carácter 3, detección 1.

### Análisis de fallos

| Patente real | Resultado | Categoría | Observación |
|---|---|---|---|
| CZZD27 | GZZD27 | Confusión de carácter (C→G) | La forma de la C se parece a una G en esta tipografía. Para una persona se entiende como C. |
| DRCC90 | DRCC90 (correcta) | Error de datos corregido | En la primera evaluación la imagen no se evaluó porque el nombre del archivo quedó alterado en el CSV. Corregido y reevaluado: el sistema la lee bien, aunque el OCR sobre el recorte del sistema de referencia falla (R→H por el sol). |
| SFXR46 | KWKF82 | Selección de vehículo (escena no aplicable) | Cámara de dos pistas: la regla de la caja más grande eligió la furgoneta de la otra pista, cuya patente se leyó correctamente. El sistema está diseñado para un acceso de una sola pista, donde solo importa el vehículo más cercano; esta escena no representa el escenario del proyecto. |
| CPZT69 | WT5479 | Selección de vehículo | YOLO detectó ambas patentes, pero generó dos cajas superpuestas sobre la del vehículo de atrás, y una de ellas tenía más área que la caja ajustada de la patente principal. La regla de la caja más grande eligió mal. |
| XB8835 | XBBB35 | Formato antiguo | Patente antigua (2 letras + 4 números). El OCR leyó 8 como B y el texto calzó con el formato nuevo, que la validación prioriza. |
| CRLV45 | GRLV45 | Confusión de carácter (C→G) | La placa tiene un defecto físico que hace que la C parezca una G. |
| PLDW22 | Sin detección | Detección | Placa de un tipo distinto, con parte de los bordes no visible. |
| YL2154 | YLZL54 | Formato antiguo | Mismo caso que XB8835 (2→Z, 1→L). |
| FKHZ86 | FKHZ88 | Confusión de carácter (6→8) | Iluminación y ángulo desfavorables para la lectura. |

### Pendientes identificados

- **Patentes de formato antiguo:** cuando el texto calza con ambos formatos, la validación elige el nuevo. Hay que definir una estrategia y probarla en el conjunto de desarrollo, agregándole más patentes antiguas. No se debe ajustar mirando el test.
- **Selección de la patente:** el área de la caja no basta cuando hay detecciones duplicadas u holgadas. Alternativas a probar en desarrollo: elegir la caja más baja en la imagen (en una cámara elevada, el vehículo más cercano aparece más abajo) o combinar área y confianza.
- **Confusión C/G:** se repite en 2 casos, y también aparece en el sistema de referencia (S/G).

## Despliegue en Raspberry Pi 4 (4 GB)

- Entorno: Raspberry Pi OS Lite 64-bit (Debian 13), Python 3.13, PyTorch versión CPU, Ultralytics, fast-plate-ocr (ONNX, CPU).
- Modelo exportado a NCNN (imgsz=640) desde el `.pt` v1.
- Prueba con el conjunto de desarrollo (30 imágenes), pipeline completo con la regla de la caja más grande:

| Formato | Detección | OCR | Total por imagen | Imágenes/s | Exactitud |
|---|---|---|---|---|---|
| NCNN | 376 ms | 146 ms | 522 ms | 1,9 | 29/30 |
| PyTorch (.pt) | 929 ms | 149 ms | 1.078 ms | 0,9 | 29/30 |

- NCNN es 2,5 veces más rápido en la detección, sin pérdida de exactitud: es el formato elegido para el despliegue.
- La carga inicial del modelo toma alrededor de 4 s; en operación ocurre una sola vez, al iniciar el servicio.
- Fallo SBWC79 → TTDY78: con la regla de la caja más grande se seleccionó otra patente (a confirmar visualmente). En el PC, con la regla de mayor confianza, se leía bien.

## Lectura en vivo desde la cámara (Tapo C310 por RTSP)

| Configuración | Resolución | Atraso aproximado | Lectura de patentes |
|---|---|---|---|
| stream1 + OpenCV (decodificación por procesador) | 1920x1080 | Creciente: el procesador no alcanza a decodificar | Correcta, pero sobre cuadros atrasados |
| stream1 + FFmpeg (decodificación por hardware) | 1920x1080 | ~5 s, fluido | Correcta, pero con demasiado atraso para un portón |
| stream2 + OpenCV | 640x360 | ~1 s | Igual de buena a las distancias probadas |

- Decisión: stream2. Cumple el requisito (auto detenido frente al portón) con el menor atraso y la configuración más simple.
- Medidas para evitar cuadros atrasados: un hilo de lectura que guarda solo el cuadro más reciente, transporte RTSP por TCP y reconexión automática.
- Reglas en vivo: confirmación (2 lecturas iguales seguidas) y espera de 30 s desde la última vez que se vio la patente.
- Visor web de diagnóstico (Flask + MJPEG) para probar distancias y ángulos.