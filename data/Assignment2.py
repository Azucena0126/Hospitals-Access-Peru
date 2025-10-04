#!/usr/bin/env python
# coding: utf-8

# In[3]:
get_ipython().system('pip install folium')
get_ipython().system('pip install branca')
# In[4]:


import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt 
import chardet
import geopandas as gpd
from geopandas import GeoSeries
from shapely.geometry import Point, LineString
import folium 
from folium import Marker, GeoJson
from folium.plugins import MarkerCluster, HeatMap
import folium as fm
from folium import Marker, GeoJson
from folium.plugins import MarkerCluster, HeatMap, StripePattern

import geopandas as gpd
from geopandas import GeoSeries
from shapely.geometry import Point, LineString

import branca as br 
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt 
import chardet
import geopandas as gpd
from geopandas import GeoSeries
from shapely.geometry import Point, LineString
import folium 
from folium import Marker, GeoJson
from folium.plugins import MarkerCluster, HeatMap
import unicodedata


# In[5]:


from IPython.display import display, HTML

display(HTML(data="""
<style>
    div#notebook-container    { width: 95%; }
    div#menubar-container     { width: 65%; }
    div#maintoolbar-container { width: 99%; }a
</style>
"""))


# # 🗺️ Task 1: Static Maps — Hospital Count by District

# In[6]:


# Carga el archivo IPRESS, filtra registros en funcionamiento y elimina filas con valores nulos o cero en NORTE/ESTE.
def cargar_ipress(ruta_csv):
        
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
    
    # Ajustar UBIGEO antes o después de filtrar
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


# In[7]:


def plot_hospitales(dataset_cv, filtro="all", color="#9DDE8B", titulo=None):
   
    # --- 1. Aplicar filtro ---
    if filtro == "with":
        dataset_filtrado = dataset_cv[dataset_cv['INSTITUCIÓN'].notna()]
        titulo_auto = "Map: Districts with hospitals"
    elif filtro == "without":
        dataset_filtrado = dataset_cv[dataset_cv['INSTITUCIÓN'].isna()]
        titulo_auto = "Map: Districts without hospitals"
    elif filtro == "top10":
        hospitales_por_distrito = (
            dataset_cv.groupby(["UBIGEO","geometry"])["INSTITUCIÓN"]
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
    fig, ax = plt.subplots(figsize=(20, 20))
    
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
            column="n_hospitales",   # escala de color según número de hospitales
            cmap="Reds",          # paleta de colores distinta
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
    plt.title(titulo if titulo else titulo_auto, fontsize=18, fontweight="bold")
    plt.show()


# In[8]:


#Carga de data IPRESS
ruta = r'./IPRESS.csv'
df = cargar_ipress(ruta)


# In[9]:


#Carga de data de distritos
maps = gpd.read_file(r'./DISTRITOS.shp')


# In[10]:


# Seleccionamos las columnas relevantes
maps = maps[['IDDIST', 'geometry']]
maps = maps.rename({'IDDIST':'UBIGEO'}, axis =1 )


# In[11]:


# Aseguramos el formato WGS-84 (EPSG:4326)
maps = maps.to_crs(epsg=4326)


# In[12]:


#Creamos el dataframe que vamos a utilizar para el ploteo
dataset_cv = pd.merge(maps, df, how="left", on="UBIGEO")


# ### Map 1: Total public hospitals per district.

# In[13]:


# Distritos con hospitales
plot_hospitales(dataset_cv, filtro="with", titulo="Distritos con hospitales públicos")


# ### Map 2: Highlight districts with zero hospitals.

# In[14]:


# Distritos sin hospitales
plot_hospitales(dataset_cv, filtro="without",color='#51829B', titulo="Distritos sin hospitales públicos")


# ### Map 3: Top 10 districts with the highest number of hospitals (distinct color scale).

# In[15]:


# Top 10 distritos con más hospitales
plot_hospitales(dataset_cv, filtro="top10", titulo="Top 10 distritos con más hospitales")


# ## Tarea 2: Análisis a nivel departamental
# ### Agregado a nivel departamental.
# Calcular el número total de hospitales operativos en cada departamento.
# 

# In[16]:


# Cargar y filtrar hospitales en funcionamiento
df = pd.read_csv('./IPRESS.csv', encoding='latin-1')
df = df[df['Condición'] == 'EN FUNCIONAMIENTO']


# In[17]:


# Agregado a nivel departamental
conteo_departamentos = (
    df.groupby('Departamento')['Nombre del establecimiento']
    .count()
    .reset_index()
    .rename(columns={'Nombre del establecimiento': 'Total_Hospitales_Operativos'})
    .sort_values('Total_Hospitales_Operativos', ascending=False)
)

# Resultado
conteo_departamentos


# ### El departamento con mayor número de hospitales.

# In[18]:


# Identificar el departamento con mayor número de hospitales
max_dep = conteo_departamentos.loc[conteo_departamentos['Total_Hospitales_Operativos'].idxmax()]

print(f"El departamento con mayor número de hospitales es {max_dep['Departamento']} con {max_dep['Total_Hospitales_Operativos']} hospitales operativos.")


# ## El departamento con menor número de hospitales.

# In[19]:


# Identificar el departamento con menor número de hospitales
min_dep = conteo_departamentos.loc[conteo_departamentos['Total_Hospitales_Operativos'].idxmin()]

print(f"El departamento con menor número de hospitales es {min_dep['Departamento']} con {min_dep['Total_Hospitales_Operativos']} hospitales operativos.")


# ## Una tabla resumen (ordenada de mayor a menor).

# In[20]:


# Tabla resumen ordenada de mayor a menor
tabla_resumen = conteo_departamentos.sort_values(
    by="Total_Hospitales_Operativos", ascending=False
).reset_index(drop=True)

display(tabla_resumen)


# ## Un gráfico de barras (matplotlib o seaborn).

# In[21]:


# Ordenamos de mayor a menor
tabla_resumen = conteo_departamentos.sort_values("Total_Hospitales_Operativos", ascending=False)

# Gráfico de barras horizontal
tabla_resumen.plot(
    x="Departamento", y="Total_Hospitales_Operativos",
    kind="barh", figsize=(8,6), legend=False, color="steelblue"
)

plt.gca().invert_yaxis()
plt.title("Hospitales operativos por departamento")
plt.xlabel("Total de hospitales")
plt.show()


# 📊 Task 2: Department-level Analysis

# In[22]:


# 1) Cargamos shapefile departamental
shp = "./Departamental INEI 2023 geogpsperu SuyoPomalia.shp"  
gdf = gpd.read_file(shp)


# In[23]:


# 2) Normalizamos texto (para unir por nombre de departamento) 
norm = lambda s: ''.join(c for c in unicodedata.normalize('NFD', str(s).upper().strip())
                         if unicodedata.category(c) != 'Mn')

# Intentamos detectar la columna de nombre de departamento en el shapefile
col_dep = next((c for c in gdf.columns if 'DEPART' in c.upper() or 'NOMB' in c.upper()), gdf.columns[0])
gdf['DEP_NORM'] = gdf[col_dep].apply(norm)


# In[24]:


# 3) Conteo por departamento (si ya tienes conteo_departamentos, omite este bloque) 

df = pd.read_csv('./IPRESS.csv', encoding='latin-1')
df = df[df['Condición'] == 'EN FUNCIONAMIENTO']
conteo = (df.groupby('Departamento')['Nombre del establecimiento']
            .count().reset_index()
            .rename(columns={'Nombre del establecimiento':'Total_Hospitales_Operativos'}))
conteo['_DEP_NORM'] = conteo['Departamento'].apply(norm)


# In[25]:


# 4) Unir shapefile + conteo 
gdf_m = gdf.merge(conteo, left_on='DEP_NORM', right_on='_DEP_NORM', how='left')
gdf_m['Total_Hospitales_Operativos'] = gdf_m['Total_Hospitales_Operativos'].fillna(0)
gdf_m = gdf_m.to_crs(epsg=4326)


# In[26]:


#  5) Coroplético 

fig, ax = plt.subplots(figsize=(9,9))
gdf_m.plot(column='Total_Hospitales_Operativos',
           ax=ax, linewidth=0.5, edgecolor='black', legend=True)
ax.set_title("Hospitales operativos por departamento (Perú)", fontsize=14, fontweight='bold')
ax.set_axis_off()
plt.tight_layout()
plt.show()


# Guardamos imagen
#plt.savefig("mapa_coropletico_departamental.png", dpi=200, bbox_inches="tight")


# 📍 Task 3: Proximity Analysis (using Population Centers)
# *Regions to analyze:*
# 
# - Lima
# - Loreto

# In[27]:


#Gettting the character format

base = open(r"./CCPP_IGN100K.shp", 'rb').read()
det = chardet.detect(base)
charenc = det['encoding']
charenc


# In[28]:


# Helpers mínimos 
norm = lambda s: ''.join(c for c in unicodedata.normalize('NFD', str(s).upper().strip())
                         if unicodedata.category(c)!='Mn')
def col_dep(gdf):
    for c in gdf.columns:
        u=c.upper()
        if any(k in u for k in ("DEPART","DPTO","DEPARTAMENTO","DEP")): return c
    return gdf.columns[0]


# In[29]:


# 1) Cargamos CCPP y asegurar CRS EPSG:4326
ccpp = gpd.read_file(r"./CCPP_IGN100K.shp")
ccpp = ccpp.set_crs(4326) if ccpp.crs is None else ccpp.to_crs(4326)
ccpp.head(5)


# In[30]:


#Limpieza de data 

# Usar/crear columnas X (lon) e Y (lat)
if not {"X","Y"}.issubset(ccpp.columns):
    ccpp["X"] = ccpp.geometry.x
    ccpp["Y"] = ccpp.geometry.y

# 1) Coordenadas válidas
ccpp["X"] = pd.to_numeric(ccpp["X"], errors="coerce")
ccpp["Y"] = pd.to_numeric(ccpp["Y"], errors="coerce")

valid = (
    np.isfinite(ccpp["X"]) & np.isfinite(ccpp["Y"]) &   # no NaN/inf
    ccpp["X"].between(-180, 180) &                     # rango lon
    ccpp["Y"].between(-90, 90) &                       # rango lat
    (ccpp["X"] != 0) & (ccpp["Y"] != 0)                # evita (0,0)
)

ccpp = ccpp[valid].copy()

# 2) Eliminar duplicados
ccpp = ccpp.drop_duplicates(subset=["X","Y"]).copy()
# (opcional) también por geometría:
# ccpp = ccpp.drop_duplicates(subset=["geometry"]).copy()

ccpp = ccpp.reset_index(drop=True)
print(f"Registros limpios: {len(ccpp):,}")


# In[31]:


## si ccpp ya está en EPSG:4326 y es de puntos:
ccpp_cent = ccpp.copy()

# limpieza (geometrías válidas y sin duplicados exactos)
ccpp_cent = ccpp_cent[ccpp_cent.geometry.notna()].copy()
ccpp_cent = ccpp_cent.drop_duplicates(subset=["geometry"]).copy()


# ## A partir de los centros de población , calcule el centroide de cada localidad (o utilice los centroides geométricos proporcionados)

# In[32]:


## Filtramos Lima 
ccpp_lima = ccpp[ccpp.DEP == "LIMA"]
ccpp_lima 


# In[33]:


# Centroide de cada localidad para Lima  
pt = ccpp_lima.unary_union.centroid
lat_lima, lon_lima = float(pt.y), float(pt.x)

m_lima = fm.Map(location=[lat_lima, lon_lima], zoom_start=9, tiles="OpenStreetMap",control_scale=True)
fm.Marker([lat_lima, lon_lima], tooltip="Centroide (Lima)").add_to(m_lima)
m_lima


# In[34]:


## Filtramos para Loreto   
ccpp_loreto = ccpp[ccpp.DEP == "LORETO"]
ccpp_loreto 


# In[35]:


# Centroide de cada localidad para Loreto  
pt = ccpp_loreto.unary_union.centroid
lat_loreto, lon_loreto = float(pt.y), float(pt.x)

m_loreto = fm.Map(location=[lat_loreto, lon_loreto], zoom_start=9, tiles="OpenStreetMap",control_scale=True)
fm.Marker([lat_loreto, lon_loreto], tooltip="Centroide (Lima)").add_to(m_loreto)
m_loreto


# ## Para cada centroide, calcule el número de hospitales operativos dentro de un área de influencia de 10 km .Identificar (por región)

# In[36]:


# === 1) Cargar hospitales EN FUNCIONAMIENTO -> GeoDataFrame (EPSG:4326) ===
ip = pd.read_csv("./IPRESS.csv", encoding="latin-1")
ip = ip[ip["Condición"].str.upper().str.strip()=="EN FUNCIONAMIENTO"].copy()

U = lambda s: str(s).upper().strip()
loncol = next((c for c in ip.columns if U(c) in {"LONGITUD","LON","NORTE"}), None)
latcol = next((c for c in ip.columns if U(c) in {"LATITUD","LAT","ESTE"}),  None)
assert loncol and latcol, "No se hallaron columnas LONG/LAT o NORTE/ESTE en IPRESS.csv"

ip[loncol] = pd.to_numeric(ip[loncol], errors="coerce")
ip[latcol] = pd.to_numeric(ip[latcol], errors="coerce")
ip = ip.dropna(subset=[loncol, latcol]).drop_duplicates(subset=[loncol, latcol])

gdf_hosp = gpd.GeoDataFrame(ip, geometry=gpd.points_from_xy(ip[loncol], ip[latcol]), crs=4326)


# In[37]:


ip.head(5)


# ### Número de hospitales operativos para Lima dentro de un área de influencia de 10 km

# In[38]:


#1) Preparamos el centroide de Lima como GeoDataFrame (WGS84) 
gdf_cent_lima = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy([lon_lima], [lat_lima]), crs=4326
)


# In[39]:


#2) Proyectar a UTM 18S (Lima) para medir en metros 
cent_utm = gdf_cent_lima.to_crs(32718)
hosp_utm = gdf_hosp.to_crs(32718)


# In[40]:


# 3) Buffer de 10 km y conteo de hospitales dentro 
buf10 = cent_utm.buffer(10_000).iloc[0]                    # 10 km
dentro_utm = hosp_utm[hosp_utm.within(buf10)]
n_hosp_lima = int(len(dentro_utm))
print(f"Hospitales operativos ≤10 km del centroide de Lima: {n_hosp_lima}")


# In[41]:


# Añadimos al mapa: círculo 10 km + hospitales dentro 
# círculo en metros (folium usa WGS84 para la ubicación, pero radio en metros)
fm.Circle([lat_lima, lon_lima], radius=10_000,
          color="#d73027", weight=2, fill=False,
          tooltip=f"Buffer 10 km – Lima ({n_hosp_lima} hosp)"
         ).add_to(m_lima)


# In[42]:


# puntos de hospitales dentro del buffer
dentro_wgs84 = dentro_utm.to_crs(4326)
for _, r in dentro_wgs84.iterrows():
    fm.CircleMarker([r.geometry.y, r.geometry.x], radius=3,
                    color="#2166ac", fill=True, fill_opacity=0.9,
                    tooltip=r.get("NOMBRE DEL ESTABLECIMIENTO","Hospital")
                   ).add_to(m_lima)

m_lima


# ### Número de hospitales operativos para Loreto dentro de un área de influencia de 10 km .

# In[43]:


#1) Preparamos el centroide de Loreto como GeoDataFrame (WGS84) 
gdf_cent_loreto = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy([lon_loreto], [lat_loreto]), crs=4326
)


# In[44]:


#2) Proyectamos a UTM 18S (Lima) para medir en metros 
cent_utm = gdf_cent_loreto.to_crs(32718)
hosp_utm = gdf_hosp.to_crs(32718)


# In[45]:


# 3) Buffer de 10 km y conteo de hospitales dentro 
buf10 = cent_utm.buffer(10_000).iloc[0]                    # 10 km
dentro_utm = hosp_utm[hosp_utm.within(buf10)]
n_hosp_loreto = int(len(dentro_utm))
print(f"Hospitales operativos ≤10 km del centroide de Loreto: {n_hosp_loreto}")


# In[46]:


# Añadimos al mapa: círculo 10 km + hospitales dentro 
# círculo en metros (folium usa WGS84 para la ubicación, pero radio en metros)
fm.Circle([lat_loreto, lon_loreto], radius=10_000,
          color="#d73027", weight=2, fill=False,
          tooltip=f"Buffer 10 km – Lima ({n_hosp_lima} hosp)"
         ).add_to(m_loreto)


# In[47]:


# puntos de hospitales dentro del buffer
dentro_wgs84 = dentro_utm.to_crs(4326)
for _, r in dentro_wgs84.iterrows():
    fm.CircleMarker([r.geometry.y, r.geometry.x], radius=3,
                    color="#2166ac", fill=True, fill_opacity=0.9,
                    tooltip=r.get("NOMBRE DEL ESTABLECIMIENTO","Hospital")
                   ).add_to(m_loreto)

m_loreto


# ## Para la región LIMA, indentificamos el centro poblacional con menos hospitales cercanos (aislamiento).

# In[48]:


# 1) Pasamos a UTM 18S (Lima) para medir en metros
cp_utm = ccpp_lima.to_crs(32718).copy()
hp_utm = gdf_hosp.to_crs(32718).copy()


# In[49]:


# 2) Crear buffers de 10 km alrededor de cada centro poblado
cp_utm = cp_utm.reset_index(drop=False).rename(columns={"index": "cp_id"})
buf = cp_utm[["cp_id", "geometry"]].copy()
buf["geometry"] = buf.geometry.buffer(10_000)   # 10 km


# In[50]:


# 3) Join espacial: hospitales dentro de cada buffer -> conteo por centro
join = gpd.sjoin(hp_utm, buf.set_geometry("geometry"), predicate="within", how="left")
conteo = join.groupby("cp_id").size().rename("n_hosp_10km")


# In[51]:


# 4) Resultado por centro
res = cp_utm.merge(conteo, on="cp_id", how="left")
res["n_hosp_10km"] = res["n_hosp_10km"].fillna(0).astype(int)


# In[52]:


# 5) Elegir columna de nombre del centro poblado
nombre_col = next((c for c in ["NOM_POBLAD","NOMBRE","N_BUSQDA","NOM_CCPP"] if c in res.columns), None)
if nombre_col is None:
    nombre_col = res.columns[0]  # fallback


# In[53]:


# 6) Centro poblacional MÁS aislado (menor número de hospitales ≤10 km)
aislado = res.sort_values(["n_hosp_10km", nombre_col], ascending=[True, True]).iloc[0]

print(f"Centro poblacional más aislado en Lima (≤10 km): {aislado[nombre_col]}")
print(f"Hospitales operativos en 10 km: {aislado['n_hosp_10km']}")


# In[54]:


# Ploteamos
ccpp_lima = ccpp_lima[ccpp_lima.NOM_POBLAD == "ACACHI"]
ccpp_lima


# In[55]:


get_ipython().system('pip install geopy')


# In[56]:


from geopy.distance import geodesic


# In[57]:


# Coordenadas del centroide de ACACHI
lat_acachi, lon_acachi = -12.73005, -76.20807


# In[58]:


# Creamod mapa centrado en ACACHI
m_acachi = fm.Map(location=[lat_acachi, lon_acachi], zoom_start=11, tiles="OpenStreetMap")


# In[59]:


# Agregar marcador del centroide
fm.Marker(
    [lat_acachi, lon_acachi],
    popup="<b>Centro poblado: ACACHI</b>",
    tooltip="Centroide ACACHI",
    icon=fm.Icon(color="blue", icon="info-sign")
).add_to(m_acachi)


# In[60]:


# Dibujar el círculo de 10 km
fm.Circle(
    [lat_acachi, lon_acachi],
    radius=10_000,   # en metros
    color="red",
    weight=2,
    fill=False,
    tooltip="Área de influencia (10 km)"
).add_to(m_acachi)


# In[61]:


# Filtrar hospitales dentro de los 10 km
hosp_cercanos = []
for _, row in gdf_hosp.iterrows():
    hosp_point = (row.geometry.y, row.geometry.x)  # (lat, lon)
    centroide_point = (lat_acachi, lon_acachi)
    dist = geodesic(centroide_point, hosp_point).km
    
    if dist <= 10:
        hosp_cercanos.append(row)
        fm.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color="green",
            fill=True,
            fill_color="green",
            popup=f"Hospital: {row['Nombre'] if 'Nombre' in row else 'SIN NOMBRE'}<br>Distancia: {dist:.2f} km",
            tooltip="Hospital operativo"
        ).add_to(m_acachi)

# Mostrar mapa
m_acachi


# ## Para la región LIMA, indentificamos el centro poblacional con mas hospitales cercanos (concentracion).

# In[62]:


# Centro poblacional con MÁS hospitales (≤10 km)
mas_hosp = res.sort_values(["n_hosp_10km", nombre_col], ascending=[False, True]).iloc[0]

print(f"Centro poblacional con MÁS hospitales en Lima (≤10 km): {mas_hosp[nombre_col]}")
print(f"Hospitales operativos en 10 km: {mas_hosp['n_hosp_10km']}")


# In[63]:


ccpp = ccpp[ccpp.NOM_POBLAD == "BARRIO OBRERO INDUSTRIAL"]
ccpp


# In[64]:


# Coordenadas del centroide de BARRIO INDUSTRIAL
lat_industrial, lon_industrial = -12.03038, -77.04702
# Creamos mapa centrado en BARRIO OBRERO INDUSTRIAL
m_industrial = fm.Map(location=[lat_industrial, lon_industrial], 
                      zoom_start=11, 
                      tiles="OpenStreetMap")
# Agregar marcador del centroide
fm.Marker(
    [lat_industrial, lon_industrial],
    popup="<b>Centro poblado: BARRIO OBRERO INDUSTRIAL</b>",
    tooltip="Centroide BARRIO OBRERO INDUSTRIAL",
    icon=fm.Icon(color="blue", icon="info-sign")
).add_to(m_industrial)
# Dibujar el círculo de 10 km
fm.Circle(
    [lat_industrial, lon_industrial],
    radius=10_000,   # en metros
    color="red",
    weight=2,
    fill=False,
    tooltip="Área de influencia (10 km)"
).add_to(m_industrial)
# Filtrar hospitales dentro de los 10 km
hosp_cercanos = []
for _, row in gdf_hosp.iterrows():
    hosp_point = (row.geometry.y, row.geometry.x)  # (lat, lon)
    centroide_point = (lat_industrial, lon_industrial)
    dist = geodesic(centroide_point, hosp_point).km
    
    if dist <= 10:
        hosp_cercanos.append(row)
        fm.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color="green",
            fill=True,
            fill_color="green",
            popup=f"Hospital: {row['Nombre'] if 'Nombre' in row else 'SIN NOMBRE'}<br>Distancia: {dist:.2f} km",
            tooltip="Hospital operativo"
        ).add_to(m_industrial)

# Mostramos mapa
m_industrial


# ## Para la región LORETO, indentificamos el centro poblacional con menos hospitales cercanos (aislamiento).

# In[65]:


import geopandas as gpd

# 1) Proyectar a UTM 19S (Loreto) para medir en metros
cp_utm = ccpp_loreto.to_crs(32719).reset_index(drop=True)
hp_utm = gdf_hosp.to_crs(32719)

# 2) Buffer de 10 km por centro poblado
buf = gpd.GeoDataFrame({"cp_id": cp_utm.index, "geometry": cp_utm.geometry.buffer(10_000)}, crs=32719)

# 3) Join espacial: hospitales dentro de cada buffer -> conteo por centro
join = gpd.sjoin(hp_utm, buf.set_geometry("geometry"), predicate="within", how="left")
conteo = join.groupby("cp_id").size().rename("n_hosp_10km")

# 4) Resultado por centro (0 si no hay hospitales)
res = cp_utm.copy()
res["n_hosp_10km"] = conteo.reindex(res.index, fill_value=0).astype(int)

# 5) Columna de nombre del centro poblado
nombre_col = next((c for c in ["NOM_POBLAD","NOMBRE","N_BUSQDA","NOM_CCPP"] if c in ccpp_loreto.columns),
                  ccpp_loreto.columns[0])

# 6) Centro poblacional MÁS aislado en Loreto (menor número de hospitales ≤10 km)
aislado_loreto = res.sort_values(["n_hosp_10km", nombre_col], ascending=[True, True]).iloc[0]

print(f"Centro poblacional más aislado en Loreto (≤10 km): {aislado_loreto[nombre_col]}")
print(f"Hospitales operativos en 10 km: {aislado_loreto['n_hosp_10km']}")


# In[66]:


ccpp_loreto = ccpp_loreto[ccpp_loreto.NOM_POBLAD == "1 DE FEBRERO"]
ccpp_loreto


# In[67]:


# Coordenadas del centroide de 1 DE FEBRERO
lat_febrero, lon_febrero = -4.36734, -73.56656

# Creamos mapa centrado en 1 DE FEBRERO
m_febrero = fm.Map(location=[lat_febrero, lon_febrero], zoom_start=11, tiles="OpenStreetMap")

# Agregar marcador del centroide
fm.Marker(
    [lat_febrero, lon_febrero],
    popup="<b>Centro poblado: ACACHI</b>",
    tooltip="Centroide ACACHI",
    icon=fm.Icon(color="blue", icon="info-sign")
).add_to(m_febrero)
# Dibujar el círculo de 10 km
fm.Circle(
    [lat_febrero, lon_febrero],
    radius=10_000,   # en metros
    color="red",
    weight=2,
    fill=False,
    tooltip="Área de influencia (10 km)"
).add_to(m_febrero)
# Filtrar hospitales dentro de los 10 km
hosp_cercanos = []
for _, row in gdf_hosp.iterrows():
    hosp_point = (row.geometry.y, row.geometry.x)  # (lat, lon)
    centroide_point = (lat_febrero, lon_febrero)
    dist = geodesic(centroide_point, hosp_point).km
    
    if dist <= 10:
        hosp_cercanos.append(row)
        fm.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color="green",
            fill=True,
            fill_color="green",
            popup=f"Hospital: {row['Nombre'] if 'Nombre' in row else 'SIN NOMBRE'}<br>Distancia: {dist:.2f} km",
            tooltip="Hospital operativo"
        ).add_to(m_acachi)

# Mostrar mapa
m_febrero


# ## Para la región LORETO, indentificamos el centro poblacional con mas hospitales cercanos (concentración).

# In[68]:


# Centro poblacional con MÁS hospitales en Loreto (≤10 km)
mas_hosp_loreto = res.sort_values(["n_hosp_10km", nombre_col], ascending=[True, True]).iloc[-1]

print(f"Centro poblacional con MÁS hospitales en Loreto (≤10 km): {mas_hosp_loreto[nombre_col]}")
print(f"Hospitales operativos en 10 km: {mas_hosp_loreto['n_hosp_10km']}")


# In[69]:


ccpp_loreto = ccpp_loreto[ccpp_loreto.NOM_POBLAD == "TRES DE OCTUBRE"]
ccpp_loreto


# In[70]:


# Coordenadas del centroide de TRES DE OCTUBRE
lat_octubre, lon_octubre = -3.78232, -73.28708


# In[71]:


# Creamos mapa centrado en TRES DE OCTUBRE
m_octubre = fm.Map(location=[lat_octubre, lon_octubre], 
                      zoom_start=11, 
                      tiles="OpenStreetMap")
# Agregar marcador del centroide
fm.Marker(
    [lat_octubre, lon_octubre],
    popup="<b>Centro poblado: BARRIO OBRERO INDUSTRIAL</b>",
    tooltip="Centroide BARRIO OBRERO INDUSTRIAL",
    icon=fm.Icon(color="blue", icon="info-sign")
).add_to(m_octubre)
# Dibujar el círculo de 10 km
fm.Circle(
    [lat_octubre, lon_octubre],
    radius=10_000,   # en metros
    color="red",
    weight=2,
    fill=False,
    tooltip="Área de influencia (10 km)"
).add_to(m_octubre)
# Filtrar hospitales dentro de los 10 km
hosp_cercanos = []
for _, row in gdf_hosp.iterrows():
    hosp_point = (row.geometry.y, row.geometry.x)  # (lat, lon)
    centroide_point = (lat_octubre, lon_octubre)
    dist = geodesic(centroide_point, hosp_point).km
    
    if dist <= 10:
        hosp_cercanos.append(row)
        fm.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color="green",
            fill=True,
            fill_color="green",
            popup=f"Hospital: {row['Nombre'] if 'Nombre' in row else 'SIN NOMBRE'}<br>Distancia: {dist:.2f} km",
            tooltip="Hospital operativo"
        ).add_to(m_octubre)

# Mostramos mapa
m_octubre


# # 3. Interactive Mapping with Folium
# ##  Task 1: National Choropleth (District Level)
# - Build a Folium choropleth of the number of hospitals per district.
# - Add a marker cluster with all hospital points.
# 
# ### Abriendo las librerias

# In[ ]:


import pandas as pd
#from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt 
import chardet
import folium as fm
from folium import Marker, GeoJson
from folium.plugins import MarkerCluster, HeatMap, StripePattern, MousePosition
import geopandas as gpd
from geopandas import GeoSeries
from shapely.geometry import Point, LineString
import branca as br 


# In[ ]:


def create_national_choropleth_optimized():
    global dataset_cv, df
    print("Verificando estructura de datos...")
    print("Columnas en dataset_cv:", list(dataset_cv.columns))
    print("Columnas en df:", list(df.columns))
    
    # Calcular número de hospitales por distrito
    hospitales_por_distrito = dataset_cv.groupby("UBIGEO")["INSTITUCIÓN"].count().reset_index()
    hospitales_por_distrito.columns = ['UBIGEO', 'n_hospitales']
    
    # Merge con el geoDataFrame de distritos
    merged = dataset_cv[['UBIGEO', 'geometry']].drop_duplicates().merge(
        hospitales_por_distrito, on='UBIGEO', how='left'
    )
    merged['n_hospitales'] = merged['n_hospitales'].fillna(0)
    
    print(f"Total distritos: {len(merged)}")
    print(f"Rango de hospitales por distrito: {merged['n_hospitales'].min()} - {merged['n_hospitales'].max()}")
    
    # Crear mapa base centrado en Perú
    m = folium.Map(
        location=[-9.189967, -75.015152], 
        zoom_start=5,
        tiles='OpenStreetMap'
    )
    
    # Choropleth optimizado
    folium.Choropleth(
        geo_data=merged,
        name="Hospitales por distrito",
        data=merged,
        columns=['UBIGEO', 'n_hospitales'],
        key_on='feature.properties.UBIGEO',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.3,
        legend_name='Número de hospitales por distrito',
        bins=8,
        nan_fill_color='lightgray',
        highlight=True
    ).add_to(m)
    
    # MARKER CLUSTER 
    if 'gdf_hosp' not in globals():
     # Buscar columnas de coordenadas en df
        lat_col = next((col for col in df.columns if 'LAT' in col.upper()), None)
        lon_col = next((col for col in df.columns if 'LON' in col.upper() or 'ESTE' in col.upper()), None)
        
        print(f"   Columna latitud: {lat_col}")
        print(f"   Columna longitud: {lon_col}")
        
        if lat_col and lon_col:
            gdf_hosp = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
                crs=4326
            )
        else:
            print("No se encontraron columnas de coordenadas")
            return m
    else:
        gdf_hosp = globals()['gdf_hosp']
    
    marker_cluster = MarkerCluster(
        name="Hospitales",
        options={
            'maxClusterRadius': 50,
            'disableClusteringAtZoom': 12
        }
    ).add_to(m)
    
    # Añadir marcadores para cada hospital
    hospitals_added = 0
    
    for idx, row in gdf_hosp.iterrows():
        if row.geometry and not row.geometry.is_empty:
            lat, lon = row.geometry.y, row.geometry.x
            
            # Crear contenido del popup
            nombre_establecimiento = row.get('NOMBRE DEL ESTABLECIMIENTO', row.get('Nombre del establecimiento', 'Hospital'))
            institucion = row.get('INSTITUCIÓN', row.get('Institución', 'N/A'))
            clasificacion = row.get('CLASIFICACIÓN', row.get('Clasificación', 'N/A'))
            departamento = row.get('DEPARTAMENTO', row.get('Departamento', 'N/A'))
            provincia = row.get('PROVINCIA', row.get('Provincia', 'N/A'))
            distrito = row.get('DISTRITO', row.get('Distrito', 'N/A'))
            
            popup_content = f"""
            <b>{nombre_establecimiento}</b><br>
            <b>Institución:</b> {institucion}<br>
            <b>Clasificación:</b> {clasificacion}<br>
            <b>Departamento:</b> {departamento}<br>
            <b>Provincia:</b> {provincia}<br>
            <b>Distrito:</b> {distrito}
            """
            
            # Definir color del icono
            clasificacion_str = str(clasificacion).lower()
            if any(f'i-{i}' in clasificacion_str for i in range(1, 5)):
                icon_color = 'red'
            elif any(f'ii-{i}' in clasificacion_str for i in [1, 2, 'e']):
                icon_color = 'blue'
            elif any(f'iii-{i}' in clasificacion_str for i in [1, 2, 'e']):
                icon_color = 'green'
            else:
                icon_color = 'gray'
            
            # Añadir marcador
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=nombre_establecimiento,
                icon=folium.Icon(color=icon_color, icon='plus', prefix='fa')
            ).add_to(marker_cluster)
            
            hospitals_added += 1
    
    print(f"✅ Se añadieron {hospitals_added} hospitales al mapa")
    
    # Añadir control de capas
    folium.LayerControl().add_to(m)
    
    return m

national_map = create_national_choropleth_optimized()
national_map.save('mapa_hospitales.html')
national_map


# ### 📍 Task 2: Proximity Visualization — Lima & Loreto

# In[90]:


m= folium.Map(location=[-9.19, -75.02], zoom_start=5)

# Centros con MENOR densidad (círculos ROJOS)
folium.Circle(
    location=[-12.73, -76.21],  # ACACHI - Lima
    radius=10000,
    color='red',
    fill=True,
    popup='ACACHI (Lima)<br>0 hospitales en 10km',
    tooltip='Menor densidad: 0 hospitales'
).add_to(m)

folium.Circle(
    location=[-4.37, -73.57],  # 1 DE FEBRERO - Loreto
    radius=10000, 
    color='red',
    fill=True,
    popup='1 DE FEBRERO (Loreto)<br>0 hospitales en 10km',
    tooltip='Menor densidad: 0 hospitales'
).add_to(m)

# Centros con MAYOR densidad (círculos VERDES)
folium.Circle(
    location=[-12.03, -77.05],  # BARRIO INDUSTRIAL - Lima
    radius=10000,
    color='green',
    fill=True,
    popup='BARRIO INDUSTRIAL (Lima)<br>25 hospitales en 10km',
    tooltip='Mayor densidad: 25 hospitales'
).add_to(m)

folium.Circle(
    location=[-3.78, -73.29],  # TRES DE OCTUBRE - Loreto
    radius=10000,
    color='green',
    fill=True,
    popup='TRES DE OCTUBRE (Loreto)<br>5 hospitales en 10km',
    tooltip='Mayor densidad: 5 hospitales'
).add_to(m)

# Guardar y mostrar
m.save('proximity_analysis.html')
m


# #### Breve análisis
# LIMA: Existe una alta concentración urbana de hospitales, lo que refleja una mejor accesibilidad en las áreas metropolitanas. A partir de la gráfica se observa una elevada densidad de hospitales en el centro y una baja densidad en las periferias, lo que evidencia una desigualdad en la distribución de estos servicios de salud. En términos de cobertura, dentro de un radio de 10 km se identifican zonas con 0 hospitales en las áreas de menor densidad y hasta 25 hospitales en las zonas más concentradas. En esta situación se requiere implementar estrategias orientadas a fortalecer los servicios de primer nivel de atención en las periferias, optimizando los recursos ya existentes.
# 
# LORETO: Presenta una gran dispersión geográfica propia de la Amazonía, lo que genera importantes desafíos de accesibilidad debido a las condiciones territoriales. En esta región, el número de servicios de salud varía entre 0 en las zonas más alejadas y un máximo de 5 en los puntos con mayor concentración. Esta situación evidencia la necesidad de implementar estrategias de atención en salud móvil y rural y las brigadas itinerantes que permitan llegar de manera efectiva a las comunidades más apartadas.
