#!/usr/bin/env python3
"""
Genera tres figuras para comparar la clasificación chilena de calidad del
aire con el modelo propuesto:

1. Matriz de reclasificación:
   usa todos los archivos diarios por estación ubicados en
   resultados/consolidados y terminados en alguno de estos sufijos:

       _consolidado_resumido.csv
       _consolidados_resumidos.csv

   Cada celda muestra el porcentaje que representa esa combinación respecto
   del total de registros válidos, por lo que la matriz completa suma 100 %.
   Solo se muestran las categorías que efectivamente aparecen en cada
   sistema de clasificación.

2. Barras apiladas al 100 %:
   usa el archivo regional:

       resultados/consolidados/<REGION>_consolidado.csv

3. Calendario o línea temporal diaria:
   usa el mismo consolidado regional y representa una franja para la norma
   chilena y otra para el modelo propuesto, limitada estrictamente a la
   primera y la última fecha disponibles.

El nombre de la región se obtiene, por defecto, del nombre de la carpeta
donde se encuentra este script. Las figuras PNG se guardan en:

    Figuras/

Uso normal:
    python generar_figuras_calidad_aire.py

Opcional, para ejecutar el script sobre otra carpeta:
    python generar_figuras_calidad_aire.py --carpeta "ruta/a/RM"

Dependencias:
    pandas, numpy y matplotlib
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle


CATEGORIAS = (
    "Bueno",
    "Regular",
    "Alerta",
    "Preemergencia",
    "Emergencia",
)

COLORES = {
    "Bueno": "#2E8B57",
    "Regular": "#F2C94C",
    "Alerta": "#F2994A",
    "Preemergencia": "#D73027",
    "Emergencia": "#7B3294",
}

COLOR_SIN_DATO = "#D9D9D9"

CONDICIONES_NORMALIZADAS = {
    "bueno": "Bueno",
    "regular": "Regular",
    "alerta": "Alerta",
    "preemergencia": "Preemergencia",
    "pre emergencia": "Preemergencia",
    "pre-emergencia": "Preemergencia",
    "emergencia": "Emergencia",
}

SUFIJOS_ESTACION = (
    "_consolidado_resumido.csv",
    "_consolidados_resumidos.csv",
)

MESES_ABREVIADOS = (
    "Ene",
    "Feb",
    "Mar",
    "Abr",
    "May",
    "Jun",
    "Jul",
    "Ago",
    "Sep",
    "Oct",
    "Nov",
    "Dic",
)


def configurar_estilo() -> None:
    """Configura una estética consistente para las tres figuras."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def normalizar_texto(valor: object) -> str:
    """Normaliza tildes, mayúsculas y espacios para comparar categorías."""
    texto = unicodedata.normalize("NFKD", str(valor))
    texto = "".join(
        caracter
        for caracter in texto
        if not unicodedata.combining(caracter)
    )
    return " ".join(texto.casefold().strip().split())


def leer_csv(ruta: Path) -> pd.DataFrame:
    """Lee un CSV tolerando separadores y codificaciones habituales."""
    ultimo_error: Exception | None = None

    for codificacion in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            dataframe = pd.read_csv(
                ruta,
                sep=None,
                engine="python",
                encoding=codificacion,
            )
        except (UnicodeDecodeError, pd.errors.ParserError) as error:
            ultimo_error = error
            continue

        dataframe.columns = [
            str(columna).replace("\ufeff", "").strip()
            for columna in dataframe.columns
        ]
        if len(dataframe.columns) > 1:
            return dataframe

    detalle = f" Detalle: {ultimo_error}" if ultimo_error else ""
    raise ValueError(
        f"No fue posible leer correctamente '{ruta.name}'.{detalle}"
    )


def convertir_fechas(
    serie: pd.Series,
    ruta: Path,
) -> pd.Series:
    """Convierte fechas ISO o día-mes-año y detecta valores inválidos."""
    texto = serie.astype("string").str.strip()
    fechas = pd.to_datetime(texto, format="%Y-%m-%d", errors="coerce")

    faltantes = fechas.isna() & texto.notna() & texto.ne("")
    if faltantes.any():
        fechas_alternativas = pd.to_datetime(
            texto.loc[faltantes],
            dayfirst=True,
            errors="coerce",
        )
        fechas.loc[faltantes] = fechas_alternativas

    invalidas = texto.isna() | texto.eq("") | fechas.isna()
    if invalidas.any():
        filas = (serie.index[invalidas] + 2).tolist()[:10]
        raise ValueError(
            f"'{ruta.name}' contiene fechas vacías o inválidas en las "
            f"filas CSV {filas}."
        )

    return fechas.dt.normalize()


def normalizar_condiciones(
    serie: pd.Series,
    columna: str,
    ruta: Path,
) -> pd.Series:
    """Convierte las condiciones a los cinco nombres canónicos."""
    resultado = pd.Series(pd.NA, index=serie.index, dtype="string")
    texto = serie.astype("string").str.strip()
    presentes = serie.notna() & texto.ne("")

    if presentes.any():
        claves = texto.loc[presentes].map(normalizar_texto)
        condiciones = claves.map(CONDICIONES_NORMALIZADAS)
        invalidas = condiciones.isna()

        if invalidas.any():
            ejemplos = (
                texto.loc[presentes]
                .loc[invalidas]
                .drop_duplicates()
                .tolist()[:10]
            )
            raise ValueError(
                f"'{ruta.name}' contiene valores no reconocidos en "
                f"'{columna}': {ejemplos}."
            )

        resultado.loc[presentes] = condiciones

    return resultado


def preparar_condiciones(ruta: Path) -> pd.DataFrame:
    """Lee y valida las columnas necesarias para las visualizaciones."""
    dataframe = leer_csv(ruta)
    requeridas = ("Fecha", "condicion_final", "condicion_mod")
    faltantes = [
        columna
        for columna in requeridas
        if columna not in dataframe.columns
    ]
    if faltantes:
        raise ValueError(
            f"'{ruta.name}' no contiene las columnas requeridas: "
            f"{', '.join(faltantes)}."
        )

    resultado = dataframe.loc[:, requeridas].copy()
    resultado["Fecha"] = convertir_fechas(resultado["Fecha"], ruta)
    resultado["condicion_final"] = normalizar_condiciones(
        resultado["condicion_final"],
        "condicion_final",
        ruta,
    )
    resultado["condicion_mod"] = normalizar_condiciones(
        resultado["condicion_mod"],
        "condicion_mod",
        ruta,
    )
    return resultado


def buscar_archivos_estacion(carpeta_consolidados: Path) -> list[Path]:
    """Encuentra los consolidados resumidos diarios de las estaciones."""
    encontrados = [
        ruta
        for ruta in carpeta_consolidados.iterdir()
        if ruta.is_file()
        and any(
            ruta.name.casefold().endswith(sufijo.casefold())
            for sufijo in SUFIJOS_ESTACION
        )
    ]
    encontrados.sort(key=lambda ruta: ruta.name.casefold())

    if not encontrados:
        sufijos = " o ".join(f"*{sufijo}" for sufijo in SUFIJOS_ESTACION)
        raise FileNotFoundError(
            "No se encontraron consolidados resumidos de estaciones en "
            f"'{carpeta_consolidados}'. Se esperaba {sufijos}."
        )

    return encontrados


def buscar_archivo_regional(
    carpeta_consolidados: Path,
    nombre_region: str,
) -> Path:
    """Localiza el consolidado regional usando el nombre de la carpeta."""
    nombre_esperado = f"{nombre_region}_consolidado.csv"
    ruta_esperada = carpeta_consolidados / nombre_esperado
    if ruta_esperada.is_file():
        return ruta_esperada

    coincidencias = [
        ruta
        for ruta in carpeta_consolidados.iterdir()
        if ruta.is_file()
        and ruta.name.casefold() == nombre_esperado.casefold()
    ]
    if len(coincidencias) == 1:
        return coincidencias[0]

    raise FileNotFoundError(
        f"No se encontró el consolidado regional '{nombre_esperado}' en "
        f"'{carpeta_consolidados}'. El nombre se construye a partir de la "
        "carpeta donde se encuentra el script."
    )


def guardar_figura(figura: plt.Figure, ruta: Path) -> None:
    """Guarda una figura en PNG con resolución apta para publicación."""
    figura.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figura)


def generar_matriz_reclasificacion(
    archivos_estacion: list[Path],
    carpeta_figuras: Path,
    nombre_region: str,
) -> tuple[Path, int, int]:
    """
    Genera una matriz con porcentajes respecto del total de registros.

    Solamente se utilizan filas que tengan ambas condiciones disponibles.
    """
    partes: list[pd.DataFrame] = []
    for ruta in archivos_estacion:
        dataframe = preparar_condiciones(ruta)
        dataframe["Archivo_estacion"] = ruta.name
        partes.append(dataframe)

    datos = pd.concat(partes, ignore_index=True)
    completos = datos.dropna(
        subset=["condicion_final", "condicion_mod"]
    ).copy()
    if completos.empty:
        raise ValueError(
            "No existen registros con condición oficial y modificada "
            "simultáneamente disponibles para construir la matriz."
        )

    categorias_oficiales = tuple(
        categoria
        for categoria in CATEGORIAS
        if completos["condicion_final"].eq(categoria).any()
    )
    categorias_propuesta = tuple(
        categoria
        for categoria in CATEGORIAS
        if completos["condicion_mod"].eq(categoria).any()
    )

    oficial = pd.Categorical(
        completos["condicion_final"],
        categories=categorias_oficiales,
        ordered=True,
    )
    propuesta = pd.Categorical(
        completos["condicion_mod"],
        categories=categorias_propuesta,
        ordered=True,
    )
    conteos = pd.crosstab(oficial, propuesta, dropna=False).reindex(
        index=categorias_oficiales,
        columns=categorias_propuesta,
        fill_value=0,
    )

    n_total = len(completos)
    porcentajes = conteos / n_total * 100
    valores_grafico = porcentajes.fillna(0).to_numpy(dtype=float)

    n_cambiados = int(
        completos["condicion_final"].ne(completos["condicion_mod"]).sum()
    )
    porcentaje_cambiado = 100 * n_cambiados / n_total

    ancho_figura = max(8.8, 1.55 * len(categorias_propuesta) + 3.2)
    alto_figura = max(6.6, 1.35 * len(categorias_oficiales) + 3.0)
    figura, eje = plt.subplots(
        figsize=(ancho_figura, alto_figura)
    )
    imagen = eje.imshow(
        valores_grafico,
        cmap="Blues",
        vmin=0,
        vmax=100,
        aspect="equal",
    )

    for fila in range(len(categorias_oficiales)):
        for columna in range(len(categorias_propuesta)):
            porcentaje = porcentajes.iloc[fila, columna]
            n_celda = int(conteos.iloc[fila, columna])
            etiqueta = f"{porcentaje:.1f}%\n(n={n_celda})"

            color_texto = (
                "white"
                if valores_grafico[fila, columna] >= 55
                else "#202020"
            )
            eje.text(
                columna,
                fila,
                etiqueta,
                ha="center",
                va="center",
                color=color_texto,
                fontsize=9.5,
                fontweight=(
                    "bold"
                    if (
                        categorias_oficiales[fila]
                        != categorias_propuesta[columna]
                        and conteos.iloc[fila, columna] > 0
                    )
                    else "normal"
                ),
            )

    for fila, categoria in enumerate(categorias_oficiales):
        if categoria not in categorias_propuesta:
            continue
        columna = categorias_propuesta.index(categoria)
        eje.add_patch(
            Rectangle(
                (columna - 0.5, fila - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#333333",
                linewidth=1.2,
            )
        )

    eje.set_xticks(
        range(len(categorias_propuesta)),
        labels=categorias_propuesta,
    )
    eje.set_yticks(
        range(len(categorias_oficiales)),
        labels=categorias_oficiales,
    )
    eje.tick_params(axis="x", rotation=30)
    eje.set_xlabel("Condición según el modelo propuesto")
    eje.set_ylabel("Condición según la normativa chilena")
    eje.set_title(
        f"Matriz de reclasificación — {nombre_region.replace('_', ' ')}",
        pad=34,
        fontweight="bold",
    )
    eje.text(
        0.5,
        1.035,
        (
            "Porcentajes calculados respecto del total de registros · "
            f"Reclasificados: {porcentaje_cambiado:.1f}% "
            f"({n_cambiados}/{n_total})"
        ),
        transform=eje.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#444444",
    )

    barra_color = figura.colorbar(
        imagen,
        ax=eje,
        fraction=0.046,
        pad=0.04,
    )
    barra_color.set_label("Porcentaje de registros (%)")
    barra_color.set_ticks([0, 20, 40, 60, 80, 100])

    eje.set_xticks(
        np.arange(-0.5, len(categorias_propuesta), 1),
        minor=True,
    )
    eje.set_yticks(
        np.arange(-0.5, len(categorias_oficiales), 1),
        minor=True,
    )
    eje.grid(which="minor", color="white", linewidth=1.5)
    eje.tick_params(which="minor", bottom=False, left=False)

    figura.tight_layout()
    salida = (
        carpeta_figuras
        / f"{nombre_region}_matriz_reclasificacion.png"
    )
    guardar_figura(figura, salida)
    return salida, n_total, n_cambiados


def calcular_distribucion(
    serie: pd.Series,
) -> tuple[pd.Series, pd.Series, int]:
    """Calcula conteos y porcentajes sobre las condiciones disponibles."""
    conteos = serie.value_counts().reindex(CATEGORIAS, fill_value=0)
    n_validos = int(conteos.sum())
    if n_validos == 0:
        raise ValueError(
            "No existen condiciones válidas para construir las barras."
        )
    porcentajes = conteos / n_validos * 100
    return conteos, porcentajes, n_validos


def generar_barras_apiladas(
    datos_region: pd.DataFrame,
    carpeta_figuras: Path,
    nombre_region: str,
) -> tuple[Path, int]:
    """Genera dos barras horizontales apiladas al 100 %."""
    conteos_oficial, porcentajes_oficial, n_oficial = (
        calcular_distribucion(datos_region["condicion_final"])
    )
    conteos_mod, porcentajes_mod, n_mod = calcular_distribucion(
        datos_region["condicion_mod"]
    )

    distribuciones = (porcentajes_oficial, porcentajes_mod)
    conteos = (conteos_oficial, conteos_mod)
    etiquetas_sistemas = (
        f"Normativa chilena\n(n={n_oficial} días)",
        f"Modelo propuesto\n(n={n_mod} días)",
    )

    figura, eje = plt.subplots(figsize=(13.2, 5.4))
    posiciones = np.arange(2)
    acumulado = np.zeros(2, dtype=float)

    for categoria in CATEGORIAS:
        anchos = np.array(
            [
                distribuciones[0].loc[categoria],
                distribuciones[1].loc[categoria],
            ],
            dtype=float,
        )
        barras = eje.barh(
            posiciones,
            anchos,
            left=acumulado,
            height=0.52,
            color=COLORES[categoria],
            edgecolor="white",
            linewidth=0.8,
            label=categoria,
        )

        for indice, barra in enumerate(barras):
            porcentaje = anchos[indice]
            if porcentaje < 3.0:
                continue
            n_categoria = int(conteos[indice].loc[categoria])
            color_texto = (
                "#202020"
                if categoria in {"Regular", "Alerta"}
                else "white"
            )
            eje.text(
                acumulado[indice] + porcentaje / 2,
                barra.get_y() + barra.get_height() / 2,
                f"{porcentaje:.1f}%\n(n={n_categoria})",
                ha="center",
                va="center",
                color=color_texto,
                fontsize=9,
                fontweight="bold",
            )

        acumulado += anchos

    eje.set_xlim(0, 100)
    eje.set_xticks(np.arange(0, 101, 10))
    eje.set_xlabel("Porcentaje de días (%)")
    eje.set_yticks(posiciones, labels=etiquetas_sistemas)
    eje.invert_yaxis()
    eje.xaxis.grid(True, color="#E5E5E5", linewidth=0.8)
    eje.set_axisbelow(True)
    eje.set_title(
        (
            "Distribución regional de días por condición de calidad "
            f"del aire — {nombre_region.replace('_', ' ')}"
        ),
        pad=18,
        fontweight="bold",
    )
    eje.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=len(CATEGORIAS),
        frameon=False,
        title="Condición",
    )

    figura.tight_layout()
    salida = (
        carpeta_figuras
        / f"{nombre_region}_barras_apiladas.png"
    )
    guardar_figura(figura, salida)
    return salida, max(n_oficial, n_mod)


def generar_calendario_diario(
    datos_region: pd.DataFrame,
    carpeta_figuras: Path,
    nombre_region: str,
) -> tuple[Path, int]:
    """Genera dos franjas diarias comparables para cada año disponible."""
    duplicadas = datos_region["Fecha"].duplicated(keep=False)
    if duplicadas.any():
        ejemplos = (
            datos_region.loc[duplicadas, "Fecha"]
            .dt.strftime("%Y-%m-%d")
            .drop_duplicates()
            .tolist()[:10]
        )
        raise ValueError(
            "El consolidado regional contiene más de una fila por fecha. "
            f"Ejemplos: {ejemplos}."
        )

    datos_ordenados = datos_region.sort_values("Fecha").set_index("Fecha")
    fecha_inicio = datos_ordenados.index.min()
    fecha_fin = datos_ordenados.index.max()
    anios = list(range(fecha_inicio.year, fecha_fin.year + 1))
    if not anios:
        raise ValueError(
            "El consolidado regional no contiene fechas para construir "
            "el calendario diario."
        )

    mapa_codigos = {
        categoria: indice + 1
        for indice, categoria in enumerate(CATEGORIAS)
    }
    colores_calendario = [COLOR_SIN_DATO] + [
        COLORES[categoria]
        for categoria in CATEGORIAS
    ]
    cmap = ListedColormap(colores_calendario)
    norm = BoundaryNorm(
        np.arange(-0.5, len(colores_calendario) + 0.5, 1),
        cmap.N,
    )

    alto = 2.35 * len(anios) + 2.1
    figura, ejes = plt.subplots(
        nrows=len(anios),
        ncols=1,
        figsize=(16.5, alto),
        squeeze=False,
    )
    ejes_planos = ejes.ravel()

    for eje, anio in zip(ejes_planos, anios):
        inicio = max(pd.Timestamp(anio, 1, 1), fecha_inicio)
        fin = min(pd.Timestamp(anio, 12, 31), fecha_fin)
        fechas_intervalo = pd.date_range(inicio, fin, freq="D")
        subconjunto = datos_ordenados.reindex(fechas_intervalo)

        matriz = np.zeros((2, len(fechas_intervalo)), dtype=int)
        matriz[0, :] = (
            subconjunto["condicion_final"]
            .map(mapa_codigos)
            .fillna(0)
            .astype(int)
            .to_numpy()
        )
        matriz[1, :] = (
            subconjunto["condicion_mod"]
            .map(mapa_codigos)
            .fillna(0)
            .astype(int)
            .to_numpy()
        )

        eje.imshow(
            matriz,
            cmap=cmap,
            norm=norm,
            aspect="auto",
            interpolation="nearest",
            extent=(
                -0.5,
                len(fechas_intervalo) - 0.5,
                1.5,
                -0.5,
            ),
        )
        eje.axhline(0.5, color="white", linewidth=2)

        posiciones_mes = []
        etiquetas_mes = []
        for mes in range(inicio.month, fin.month + 1):
            inicio_mes = max(pd.Timestamp(anio, mes, 1), inicio)
            fin_mes = min(
                pd.Timestamp(anio, mes, 1) + pd.offsets.MonthEnd(0),
                fin,
            )
            posicion_inicio = (inicio_mes - inicio).days
            posicion_fin = (fin_mes - inicio).days
            posiciones_mes.append((posicion_inicio + posicion_fin) / 2)
            etiquetas_mes.append(MESES_ABREVIADOS[mes - 1])
            eje.axvline(
                posicion_inicio - 0.5,
                color="white",
                linewidth=0.75,
                alpha=0.9,
            )

        eje.set_xticks(
            posiciones_mes,
            labels=etiquetas_mes,
        )
        eje.set_yticks(
            [0, 1],
            labels=["Norma chilena", "Modelo propuesto"],
        )
        eje.tick_params(axis="x", length=0, pad=6)
        eje.tick_params(axis="y", length=0, pad=8)
        eje.set_xlim(-0.5, len(fechas_intervalo) - 0.5)
        eje.set_title(
            str(anio),
            loc="left",
            fontsize=12,
            fontweight="bold",
            pad=7,
        )
        for borde in eje.spines.values():
            borde.set_visible(False)

    figura.suptitle(
        (
            "Calendario diario de la condición regional de calidad "
            f"del aire — {nombre_region.replace('_', ' ')}"
        ),
        y=0.985,
        fontsize=16,
        fontweight="bold",
    )
    figura.text(
        0.5,
        0.945,
        (
            "Cada columna representa un día del intervalo "
            f"{fecha_inicio:%Y-%m-%d} a {fecha_fin:%Y-%m-%d}; ambas "
            "franjas corresponden al mismo registro regional"
        ),
        ha="center",
        va="top",
        fontsize=10,
        color="#444444",
    )

    leyenda = [
        Patch(facecolor=COLORES[categoria], label=categoria)
        for categoria in CATEGORIAS
    ]
    leyenda.append(Patch(facecolor=COLOR_SIN_DATO, label="Sin dato"))
    figura.legend(
        handles=leyenda,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=6,
        frameon=False,
        title="Condición",
    )
    figura.subplots_adjust(
        left=0.13,
        right=0.985,
        top=0.89,
        bottom=0.13,
        hspace=0.68,
    )

    salida = (
        carpeta_figuras
        / f"{nombre_region}_calendario_diario.png"
    )
    guardar_figura(figura, salida)
    return salida, len(datos_region)


def ejecutar(carpeta_base: Path) -> None:
    """Coordina la lectura, validación y generación de las tres figuras."""
    carpeta_base = carpeta_base.resolve()
    nombre_region = carpeta_base.name
    carpeta_consolidados = carpeta_base / "resultados" / "consolidados"
    carpeta_figuras = carpeta_base / "Figuras"

    if not carpeta_consolidados.is_dir():
        raise FileNotFoundError(
            "No existe la carpeta esperada "
            f"'{carpeta_consolidados}'."
        )

    carpeta_figuras.mkdir(parents=True, exist_ok=True)
    configurar_estilo()

    archivos_estacion = buscar_archivos_estacion(carpeta_consolidados)
    archivo_regional = buscar_archivo_regional(
        carpeta_consolidados,
        nombre_region,
    )
    datos_region = preparar_condiciones(archivo_regional)

    matriz, n_matriz, n_cambiados = generar_matriz_reclasificacion(
        archivos_estacion,
        carpeta_figuras,
        nombre_region,
    )
    barras, n_dias = generar_barras_apiladas(
        datos_region,
        carpeta_figuras,
        nombre_region,
    )
    calendario, _ = generar_calendario_diario(
        datos_region,
        carpeta_figuras,
        nombre_region,
    )

    print("Figuras generadas correctamente:")
    print(f"  - {matriz}")
    print(f"  - {barras}")
    print(f"  - {calendario}")
    print(
        "\nMatriz de reclasificación: "
        f"{n_matriz} registros estación-día válidos; "
        f"{n_cambiados} cambiaron de categoría "
        f"({100 * n_cambiados / n_matriz:.1f}%)."
    )
    print(f"Consolidado regional: {n_dias} días representados.")


def main() -> None:
    """Punto de entrada del programa."""
    analizador = argparse.ArgumentParser(
        description=(
            "Genera la matriz de reclasificación, las barras apiladas "
            "y el calendario diario de calidad del aire."
        )
    )
    analizador.add_argument(
        "--carpeta",
        type=Path,
        default=Path(__file__).resolve().parent,
        help=(
            "Carpeta regional que contiene resultados/consolidados. "
            "Por defecto se usa la carpeta donde está el script."
        ),
    )
    argumentos = analizador.parse_args()
    ejecutar(argumentos.carpeta)


if __name__ == "__main__":
    main()
