import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

def cargar_ipress(ruta_csv: str) -> pd.DataFrame:
    """Carga y limpia la base de datos de IPRESS."""
    df = pd.read_csv(ruta_csv, encoding='latin-1')

    # Filtrar por condición
    df = df[df['Condición'] == 'EN FUNCIONAMIENTO']

    # Filtrar NaN y ceros en coordenadas
    df = df.dropna(subset=['NORTE', 'ESTE'])
    df = df[(df['NORTE'] != 0) & (df['ESTE'] != 0)]

    # Renombrar columnas
    df = df.rename(columns={'ESTE': 'Latitud', 'NORTE': 'Longitud'})

    # Asegurar tipo numérico
    df['Latitud'] = pd.to_numeric(df['Latitud'], errors='coerce')
    df['Longitud'] = pd.to_numeric(df['Longitud'], errors='coerce')

    # Eliminar nuevamente posibles NaN generados en la conversión
    df = df.dropna(subset=['Latitud', 'Longitud']).reset_index(drop=True)

    # Ajustar UBIGEO
    df["UBIGEO"] = df["UBIGEO"].astype(str).str.zfill(6)

    # Filtramos las columnas deseadas
    df.columns = df.columns.str.upper()
    df = df[[
        "INSTITUCIÓN",
        "NOMBRE DEL ESTABLECIMIENTO",
        "CLASIFICACIÓN",
        "DEPARTAMENTO",
        "PROVINCIA",
        "DISTRITO",
        "ESTADO",
        "LATITUD",
        "LONGITUD",
        "UBIGEO"
    ]]
    return df


def cargar_distritos(ruta_shp: str) -> gpd.GeoDataFrame:
    """Carga y prepara el shapefile de distritos."""
    maps = gpd.read_file(ruta_shp)
    maps = maps[['IDDIST', 'geometry']]
    maps = maps.rename({'IDDIST': 'UBIGEO'}, axis=1)
    maps = maps.to_crs(epsg=4326)  # Asegurar WGS-84
    maps["UBIGEO"] = maps["UBIGEO"].astype(str).str.zfill(6)
    return maps


def crear_dataset(maps: gpd.GeoDataFrame, df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Combina distritos con información de hospitales."""
    return pd.merge(maps, df, how="left", on="UBIGEO")


def plot_hospitales(dataset_cv, filtro="all", color="#9DDE8B", titulo=None):
    """Genera mapas temáticos de distritos según disponibilidad de hospitales."""

    # --- 1. Aplicar filtro ---
    if filtro == "with":
        dataset_filtrado = dataset_cv[dataset_cv['INSTITUCIÓN'].notna()]
        titulo_auto = "Map: Districts with hospitals"
    elif filtro == "without":
        dataset_filtrado = dataset_cv[dataset_cv['INSTITUCIÓN'].isna()]
        titulo_auto = "Map: Districts without hospitals"
    elif filtro == "top10":
        hospitales_por_distrito = (
            dataset_cv.groupby(["UBIGEO", "geometry"])["INSTITUCIÓN"]
            .count().reset_index(name="n_hospitales")
        )
        dataset_filtrado = gpd.GeoDataFrame(
            hospitales_por_distrito.nlargest(10, "n_hospitales"),
            geometry=hospitales_por_distrito["geometry"],
            crs=dataset_cv.crs
        )
        titulo_auto = "Map 3: Top 10 districts with the highest number of hospitals"
    else:
        dataset_filtrado = dataset_cv.copy()
        titulo_auto = "Map: All districts"

    # --- 2. Crear figura ---
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_aspect('equal')

    # Fondo
    dataset_cv.plot(
        ax=ax,
        facecolor="none",
        edgecolor="gray",
        linewidth=0.5,
        linestyle='dotted'
    )

    # Filtrado encima
    if filtro == "top10":
        dataset_filtrado.plot(
            ax=ax,
            column="n_hospitales",
            cmap="Reds",
            edgecolor="white",
            linewidth=1,
            legend=True
        )
    else:
        dataset_filtrado.plot(
            ax=ax,
            color=color,
            edgecolor="white",
            linewidth=1,
            linestyle='dotted',
            legend=True,
        )

    # --- 3. Título ---
    ax.set_title(titulo if titulo else titulo_auto, fontsize=14, fontweight="bold")

    return fig  

