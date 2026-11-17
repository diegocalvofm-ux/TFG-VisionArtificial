# Atribución de los datasets públicos

Este proyecto usa los datasets públicos de esta tabla. Sus imágenes y anotaciones no están en el
repositorio: se descargan aparte a `data/raw/`, fuera del control de versiones.

En el proyecto, sus imágenes se recortan por caja, se redimensionan y se reetiquetan con los
grupos de calidad propios del TFG; las anotaciones originales se traducen con las reglas de
`configs/datasets/`.

- **Dataset**: nombre del proyecto en Roboflow, según el `README.roboflow.txt` del ZIP.
- **Fecha de descarga**: la de exportación que indica ese mismo README.
- **PENDIENTE**: dato aún no confirmado.

Archivo generado con `python -m src.data.atribucion` a partir de los YAML de
`configs/datasets/`. No se edita a mano: se cambia el YAML y se vuelve a generar.

| Fuente | Dataset | Autor | URL | Licencia | Fecha de descarga |
|---|---|---|---|---|---|
| giovi | Lettuce segmentation | PENDIENTE | PENDIENTE | CC BY 4.0 | 2026-10-27 |
| phs | lettuce_disease | PHS | https://universe.roboflow.com/phs/lettuce_disease | MIT | 2026-11-16 |
| svodk | lettuce segmentation | PENDIENTE | PENDIENTE | MIT | 2026-10-27 |
| tipburn | Lettuce_Segmentation | PENDIENTE | PENDIENTE | CC BY 4.0 | 2026-10-27 |
| yssp | disease lettuce | yssp | https://universe.roboflow.com/yssp/disease-lettuce-h69zb | CC BY 4.0 | 2026-11-16 |
