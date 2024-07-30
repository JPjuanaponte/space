#Importar librerias necesarias 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import unidecode
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from fastapi import FastAPI 
from fastapi import HTTPException

# Carga de datos desde el archivo parquet
movies = pd.read_parquet('Data_movies_.parquet')
# Inicializar la aplcación FastApi con que luego se hará el despliegue
app = FastAPI()

# Función para normalizar texto eliminando acentos en os ingresos del usuario
def normalize_text(text):
    return unidecode.unidecode(text)

# FUNCION PARA CONTAR PELICULAS ESTRENADAS EN UN MES ESPECIFICO

@app.get("/cantidad_filmaciones_mes/") 
def cantidad_filmaciones_mes(mes: str):
    # Verificar si se ha proporcionado un valor válido para el mes
    if not isinstance(mes, str) or not mes.strip():
        return "Mes no valido. Por favor, proporciona un mes en español."
    # Mapa de meses en español a número de mes
    meses_dict = {
        "enero": 1,
        "febrero": 2,
        "marzo": 3,
        "abril": 4,
        "mayo": 5,
        "junio": 6,
        "julio": 7,
        "agosto": 8,
        "septiembre": 9,
        "octubre": 10,
        "noviembre": 11,
        "diciembre": 12
    }
    #Se obtiene el número de mes que equivale al nombre de mes
    mes_numero = meses_dict.get(mes.lower())
    if mes_numero is None:
        return "Mes no válido. Usa uno de los meses en español."

    # Convertir la columna 'release_date' a datetime
    movies['release_date'] = pd.to_datetime(movies['release_date'], format='%Y-%m-%d', errors='coerce')
    
    # Filtrar películas que se estrenaron en el mes dado
    cantidad_peliculas = movies[movies['release_date'].dt.month == mes_numero].shape[0]
    
    return f"{cantidad_peliculas} películas fueron estrenadas en el mes de {mes}"

# FUNCION PARA CONTAR PELICULAS ESTRENADAS EN UN DIA ESPECIFICO

@app.get("/cantidad_filmaciones_dia/")
def cantidad_filmaciones_dia(dia: str):
    # Verificar si se ha proporcionado un valor válido para el día
    if not isinstance(dia, str) or not dia.strip():
        return "Dia no valido. Por favor, proporciona un nombre de día en español."
    # Mapa de días en español a número de día
    dias_dict = {
        "lunes": 1,
        "martes": 2,
        "miércoles": 3,
        "jueves": 4,
        "viernes": 5,
        "sábado": 6,
        "domingo": 7
    }
    # Se obtiene el número de día correspondiente al nombre del día
    dia_numero = dias_dict.get(dia.lower())
    if dia_numero is None:
        return "Día no válido, Usa uno de los días en español o verifica ortografía."

    # Convertir la columna 'release_date' a datetime
    movies['release_date'] = pd.to_datetime(movies['release_date'], format='%Y-%m-%d', errors='coerce')
    
    # Filtrar películas que se estrenaron en el día dado
    cantidad_peliculas = movies[movies['release_date'].dt.dayofweek == (dia_numero - 1)].shape[0]
    
    return f"{cantidad_peliculas} películas fueron estrenadas en los días {dia}"

# FUNCION PARA INFORMACION DE AÑO DE ESTRENO Y CRITICA PROMEDIO PELICULA POR TITULO

@app.get("/score_titulo/")
def score_titulo(titulo_pelicula: str):
    #verificar se si ha proporcionado un nombre valido. 
    try:
        if not isinstance(titulo_pelicula, str) or not titulo_pelicula.strip():
            return {"Nombre no valido. Por favor, proporciona un nombre valido para la película."}
        
        # Normalizar el título de entrada
        titulo_normalizado = normalize_text(titulo_pelicula.lower())
        
        # Asegurarse de que la columna 'title' solo contiene strings
        movies['title'] = movies['title'].fillna('').astype(str)
        
        # Buscar la película en el DataFrame
        pelicula = movies[movies['title'].apply(lambda x: normalize_text(x.lower())) == titulo_normalizado]
        
        if pelicula.empty:
            return {"error": "Título de película no encontrado."}
        
        # Extraer información de la película
        titulo = pelicula.iloc[0]['title']
        release_year = pelicula.iloc[0]['release_year'] 
        vote_average = pelicula.iloc[0]['vote_average']
        """
        # Convertir release_year a entero si es necesario
        if pd.isna(release_year):
            release_year = "Desconocido"
        else:
            release_year = int(release_year)
        """  
        return {
            f"La película '{titulo}' fue estrenada en el año {release_year} con un puntaje de la critica promedio de {vote_average:.2f}"
        }
    except Exception as e:
        return {"error": str(e)}


# FUNCION PARA INFORMACION DE AÑO DE ESTRENO, CONDICIONADA A TENER MAS DE 2000 VALORACIONES Y EL PROMEDIO DE VOTOS EN LA CRITICA DE LA PELICULA

@app.get("/votos_titulo/")
def votos_titulo(titulop: str):
    #verificra se si ha proporcionado un nombre valido. 
    try:
        if not isinstance(titulop, str) or not titulop.strip():
            return {'error': "Nombre no válido. Por favor, proporciona un nombre válido para la película."}
    
        # Normalizar el título de entrada
        titulo_normalizado = normalize_text(titulop.lower())
    
        # Asegurarse de que la columna 'title' solo contiene strings
        movies['title'] = movies['title'].fillna('').astype(str)
    
        # Buscar la película en el DataFrame
        pelicula = movies[movies['title'].apply(lambda x: normalize_text(x.lower())) == titulo_normalizado]
    
        if pelicula.empty:
            return {'error': "Título de película no encontrado."}
    
        # Extraer información
        pelicula_info = pelicula.iloc[0]
        titulo = pelicula_info['title']
        anio_estreno = int(pelicula_info['release_year']) 
        votos = pelicula_info['vote_count']
        promedio_voto = pelicula_info['vote_average']
    
        # Verificar si la cantidad de votos es más de 2000
        if votos < 2000:
            return {
                "La película no cumple con el mínimo de 2000 valoraciones, pendiente por ejecutar.",
            }
        
        return {
                f"La película '{titulo}' fue estrenada en el año {anio_estreno} con un total de {votos} votos y un califación promedio por parte de la crítica de {promedio_voto:.2f}"
        }
        
    except Exception as e:
        return {"error": str(e)}


# FUNCION INFORMACION ACTOR

@app.get("/get_actor/")
# Función para obtener información del actor
def get_actor(nombre_actor: str):
    # Verficiar si se ha proporcionado un nombre válido del actor
    if not isinstance(nombre_actor, str) or not nombre_actor.strip():
        return "Nombre no valido. Por favor, proporciona un nombre valido de actor."
    # Normalizar el nombre del actor
    nombre_actor_normalizado = normalize_text(nombre_actor.lower())
    
    # Asegurarse de que las columnas de nombres de actores solo contienen strings
    movies['cast_names'] = movies['cast_names'].fillna('').astype(str)
    
    # Filtrar las películas donde el actor está en la columna 'cast_names'
    peliculas_actor = movies[movies['cast_names'].apply(lambda x: any(nombre_actor_normalizado in normalize_text(actor.lower()) for actor in x.split(',')))]
    
    # Verificar si el actor ha participado en alguna película
    if peliculas_actor.empty:
        return "Actor no encontrado en el dataset."
    
    # Calcular la cantidad de películas, el retorno total y el promedio de retorno
    cantidad_peliculas = peliculas_actor.shape[0]
    retorno_total = peliculas_actor['return'].sum()
    promedio_retorno = peliculas_actor['return'].mean()
    
    return (f"El actor '{nombre_actor}' ha participado en {cantidad_peliculas} filmaciones, "
            f"ha conseguido un retorno total de {retorno_total:.2f} con un promedio de retorno por filmación de {promedio_retorno:.2f}")

# FUNCION INFORMACION DIRECTOR Y PELICULAS PRODUCIDAS

@app.get("/get_director/")
def get_director(nombre_director: str):
    # Vericicar si se ha proporcionado un nombre valido de director 
    if not isinstance(nombre_director, str) or not nombre_director.strip():
        raise HTTPException(status_code=400, detail="Nombre no válido. Por favor, proporciona un nombre válido de director.")
    
    # Normalizar el nombre del director
    nombre_director_normalizado = normalize_text(nombre_director.lower())
    
    # Asegurarse de que las columnas de nombres de directores solo contienen strings
    movies['crew_names'] = movies['crew_names'].fillna('').astype(str)
    
    # Filtrar las películas donde el director está en la columna 'crew_names'
    peliculas_director = movies[movies['crew_names'].apply(
        lambda x: any(nombre_director_normalizado in normalize_text(director.lower()) for director in x.split(','))
    )]
    
    # Verificar si el director ha trabajado en alguna película
    if peliculas_director.empty:
        raise HTTPException(status_code=404, detail=f"Director '{nombre_director}' no encontrado en el dataset.")
    
    # Calcular la cantidad de películas, el retorno total y el promedio de retorno
    cantidad_peliculas = peliculas_director.shape[0]
    retorno_total = peliculas_director['return'].sum()
    promedio_retorno = peliculas_director['return'].mean()
    
    # Crear un diccionario con la información del director y las películas
    peliculas_info = []
    for _, row in peliculas_director[['title', 'release_date', 'return', 'budget', 'revenue']].iterrows():
        peliculas_info.append({
            "nombre_pelicula": row['title'],
            "fecha_produccion": row['release_date'].strftime('%Y-%m-%d') if pd.notnull(row['release_date']) else "Desconocida",
            "retorno": row['return'],
            "presupuesto": row['budget'],
            "ganancia_pelicula": row['revenue']
        })
    
    return {
        "director": nombre_director,
        "cantidad_peliculas": cantidad_peliculas,
        "retorno_total": retorno_total,
        "promedio_retorno": promedio_retorno,
        "peliculas": peliculas_info
    }

# FUNCION DE RECOMENDACION DE PELICULAS 

# Asegurar que no hay valores None en las columnas que se van a evaludr
movies['overview'] = movies['overview'].fillna('')
movies['genres_names'] = movies['genres_names'].fillna('')
movies['cast_names'] = movies['cast_names'].fillna('')
movies['crew_names'] = movies['crew_names'].fillna('')
movies['release_year'] = movies['release_year'].fillna(0).astype(str)  # Convertir a cadena para combinar

# Convertir los títulos a minúsculas para búsqueda insensible a mayúsculas
movies['title_lower'] = movies['title'].str.lower()

# Crear una nueva columna que combine 'overview', 'genres_names', 'cast_names', 'crew_names', 'release_year' y 'vote_average'
movies['combined_features'] = (movies['overview'] + ' ' +
                               movies['genres_names'] + ' ' +
                               movies['cast_names'] + ' ' +
                               movies['crew_names'] + ' ' +
                               movies['release_year'] + ' ' +
                               movies['vote_average'].astype(str))

# Vectorizar los textos
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['combined_features'])

# Configurar el modelo de vecinos más cercanos
nn_model = NearestNeighbors(n_neighbors=6, metric='cosine')  # Ajusta el número de vecinos según sea necesario
nn_model.fit(tfidf_matrix)

@app.get("/recomendacion/")
# Crear la función de recomendación
def recomendacion(titulo: str):
    # # Vericicar si se ha proporcionado un nombre valido de pelicula 
    if not isinstance(titulo, str) or not titulo.strip():
        return "Nombre no válido. Por favor, proporciona un nombre de película válido."

    # Normalizar el texto de entrada
    titulo_normalizado = titulo.strip().lower()

    # Obtener el índice de la película que coincide con el título normalizado
    idx = movies.index[movies['title_lower'] == titulo_normalizado].tolist()
    if not idx:
        return "Película no encontrada"
    idx = idx[0]

    # Obtener el vector TF-IDF de la película seleccionada
    query_vector = tfidf_matrix[idx]

    # Encontrar las películas más similares
    distances, indices = nn_model.kneighbors(query_vector, n_neighbors=6)
    
    # Obtener los índices de las películas más similares
    similar_movie_indices = indices[0][1:]  # Omitir el primer resultado porque será la misma película
    
    # Devolver los títulos de las películas más similares
    return movies['title'].iloc[similar_movie_indices].tolist()
