---
description: Registra un experimento en su nota de Obsidian (docs/experimentos) con los datos que des y lo medido en runs/
argument-hint: <qué registrar: objetivo, dataset, entorno, carpeta de resultados, métrica...>
---

Registra en Obsidian lo siguiente: $ARGUMENTS

El usuario te da la información (objetivo, dataset, modelo, versiones de Colab, métrica, carpeta
con los resultados...) y tú la llevas a la nota con `python -m src.experimentos.registro`, un
script con pruebas que hace lo medible y no duplica nada. Tiene tres órdenes: `nuevo`,
`ejecucion` y `campos`.

## Reglas que no se rompen (de CLAUDE.md)

1. **No inventes nada.** Cada dato sale de lo que dijo el usuario o de la salida de un comando.
   Si falta, el campo queda vacío y se lo dices. Nunca estimes una métrica, un tiempo, una
   versión ni un commit.
2. **Pregunta lo que falte** (en una sola pregunta) en vez de suponerlo. Los datos esenciales son:
   objetivo, dataset (y versión), modelo, entorno, versiones de Python y PyTorch, y, para cerrar,
   métrica principal y valor.
3. **No hagas git add, commit ni push.** El usuario revisa con `git status` y `git --no-pager diff`.
4. **Solo texto del usuario en las secciones narrativas** (hipótesis, datos, análisis de errores,
   conclusión). Escríbelo con sus palabras, sin añadir conclusiones tuyas.
5. **Nada de datos personales ni rutas absolutas.** El repositorio es público: no escribas
   nombres, correos ni el usuario del sistema en las notas, y usa siempre rutas relativas a la
   raíz del repo (`runs/v0_owlvit`, `configs/v0.yaml`), nunca absolutas de Windows o Linux. El script rechaza
   valores con rutas absolutas o con el usuario del sistema; si te lo rechaza, corrígelo, no lo
   evites.
6. **No crees `exp001` por tu cuenta.** Lo crea el autor con sus datos reales.

## Qué hacer según lo que pida

Ejecuta los comandos desde la raíz del repositorio, con el Python del `.venv` y en PowerShell.
Entre comillas lo que lleve espacios. Los comandos que le muestres al usuario, también en
PowerShell.

| Quiere... | Comando |
| --- | --- |
| Un experimento nuevo | `python -m src.experimentos.registro nuevo --objetivo "..." --dataset "..." --modelo "..." --config configs/v0.yaml --entorno local-cpu --python 3.12.10 --torch 2.14.1` |
| Anotar una ejecución medida | `python -m src.experimentos.registro ejecucion --exp exp001 --runs runs/v0_owlvit --observacion "..."` |
| Cerrarlo o corregir campos | `python -m src.experimentos.registro campos --exp exp001 --set estado=hecho --set metrica_principal=... --set valor=0.35` |

Todos aceptan `--docs DIR` si la bóveda de Obsidian no es `docs/`. A `nuevo` pásale solo las
opciones que el usuario haya dado; el resto queda vacío y el comando lo lista en "Faltan".

### Tipos de la cabecera

La cabecera YAML usa los nombres de `docs/plantillas/experimento.md`. `valor` es un **número**
(con punto decimal: `0.35`; el script rechaza `0,35` o `85%`) y `metrica_principal` es el
**nombre** de la métrica (texto, p. ej. `mAP50`). `fecha` es una fecha `AAAA-MM-DD`; todos los
demás campos son texto (un commit o una versión nunca son números) y `tags` es una lista que no
se edita por comando.

### Pasos

1. **Averigua qué nota.** Si pide un experimento nuevo, usa `nuevo`. Si habla de uno existente,
   comprueba que `docs/experimentos/expNNN.md` existe antes de actuar. La fecha la pone el
   script con el reloj del sistema; no la supongas.
2. **Ejecuciones.** `ejecucion` necesita una carpeta con los JSON del pipeline (`--runs`), que
   tiene que estar dentro del repositorio (p. ej. `runs/v0_owlvit`). **No pidas ni pases detector
   ni dispositivo: no existen esas opciones.** Todo sale de los JSON: `env.detector`,
   `env.device`, `env.commit`, `env.dirty`, `image_size`, `rss_mb`, `model_load_ms` y
   `timing_ms` (con `repeticiones` y `calentamiento`). Además calcula imágenes, medianas,
   lechugas detectadas y niveles. Si un JSON no trae algún dato (un JSON anterior a la tarea 5,
   como `runs/v0_local`), esa celda queda vacía y el script avisa con la lista de datos que
   faltan: díselo al usuario y recuérdale que lo correcto es volver a ejecutar el pipeline, no
   rellenarlo a mano. Si los resultados están en Colab o Drive, pídele que los descargue a
   `runs/` o que te dé la ruta: desde aquí no se lee Drive. Sin JSON no hay fila en la tabla.
3. **Commit.** Se usa `--commit` si lo das; si no, el `env.commit` de los JSON; no hay más
   fuentes (ya no se pregunta a git: el commit de ahora puede no ser el de la ejecución). Si
   ninguno lo trae, el campo de la cabecera queda vacío y se avisa. La cabecera nunca pisa un
   commit ya anotado. Con datos de Colab, pásalo con `--commit`. Si un JSON trae `dirty: true`,
   el script avisa de que esa ejecución no se reproduce solo con el commit: repítelo al usuario.
4. **Datos de Colab.** El usuario pega la salida del notebook (commit corto, `sys.version`,
   `torch.__version__`, GPU). Pásalos a `nuevo` o `campos`.
5. **Cerrar.** Pon `estado=hecho` y `metrica_principal`/`valor` solo si el usuario los da.
6. **Texto narrativo.** Edita la nota con Edit en las secciones que correspondan, fuera de los
   marcadores `<!-- registro:... -->` (lo de dentro lo gestiona el script y se reescribe).

## Qué decir al terminar

- Muestra la nota resultante (léela) y di qué ruta es (relativa).
- Lista los campos que han quedado vacíos y por qué (no se dieron, o los JSON no traen ese
  dato). El script avisa con "faltan datos en los JSON ... (n de N)".
- Repite los avisos del script: árbol sucio (`dirty: true`: no se reproduce solo con el commit),
  datos que faltan, JSON ignorados por formato, campos con formato incorrecto, detectores o
  commits contradictorios entre JSON y un commit ya anotado distinto.
- Recuerda que `Mediana (ms)` es la mediana **entre imágenes** de `timing_ms.total`, y que cada
  valor del JSON ya es la mediana de `Repeticiones` pasadas tras `Calentamiento` de calentamiento
  (esas dos columnas están en la tabla).
- Recuerda que no has hecho `git add` ni commit.

## Cosas que conviene saber

- Cada fila de la tabla se identifica por su **carpeta** (`Carpeta`, ruta relativa). Repetir
  `ejecucion` con la misma carpeta **actualiza** su fila, no la duplica; dos carpetas son dos
  filas aunque tengan el mismo detector. La observación escrita antes se conserva si no das otra.
- Nada de lo que el usuario escribió fuera de los marcadores se sobrescribe.
- Las columnas `Commit` y `Sucio` de la tabla salen solo del JSON (lo medido al ejecutar).
- Si algo falla, el comando devuelve el motivo y no toca la nota.
