<center>

![Image Bonita](image_proyect\HEADER-BLOG-NEGRO-01.jpg)

# **Sistema de Recomendación de Películas startup Space** 

</center>
Este proyecto se desarrolló para Space, una startup que provee servicios de agregación de plataformas de streaming. El objetivo principal fue crear un sistema de recomendación de películas utilizando técnicas de machine learning, así como ofrecer funcionalidades adicionales a través de una API. A continuación, se describe el paso a paso de la construcción del proyecto, desde la extracción y transformación de datos hasta el despliegue de la API.

## Tecnologías Utilizadas

<center>

![Image Bonita](image_proyect\space.jpg)
</center>

### ETL (Extracción, Transformación y Carga)

* Pandas: Se usa para la extracción, transformación y carga de datos en formato tabular. Permite leer, escribir o manipular datos en estructuras como DataFrames, realizar operaciones de filtrado, agregación, y mucho más.
* Numpy: Proporciona soporte para arreglos multidimensionales y funciones matemáticas de alto nivel.
* Matplotlib: Se utiliza para crear gráficos estáticos, animados e interactivos en Python.
* Ast: Permite analizar y modificar el código fuente Python en tiempo de ejecución.
  
### EDA (Análisis Exploratorio de Datos)

* Seaborn: Basado en matplotlib, proporciona una interfaz de alto nivel para crear gráficos estadísticos atractivos y fáciles de interpretar.
* Wordcloud: Es útil para visualizar la frecuencia de palabras en un corpus de texto.
* Unidecode: Esto es útil para limpiar y estandarizar texto en diferentes idiomas.

### Modelado y Funciones de Recomendación

* Scikit-learn: Proporciona herramientas para modelado y análisis predictivo. 

MultiLabelBinarizer: Convierte etiquetas de múltiples clases en un formato binario.

cosine_similarity: Calcula la similitud entre vectores utilizando el coseno del ángulo.

TfidfVectorizer: Convierte un corpus de texto en una matriz de características basada en TF-IDF (Term Frequency-Inverse Document Frequency).

NearestNeighbors: Encuentra los vecinos más cercanos en un espacio multidimensional.

* Collections: Proporciona tipos de datos alternativos como Counter, defaultdict, OrderedDict, entre otros, que son útiles para la manipulación y análisis de datos.

### Desarrollo de la API

* FastAPI:  Permite definir endpoints, manejar solicitudes y respuestas, y es altamente eficiente para el desarrollo de servicios web.

### Control de Versiones

* GitHub: Proporciona una plataforma para alojar repositorios de código, colaborar con otros desarrolladores, gestionar versiones del código fuente, realizar seguimiento de los cambios mediante commits y pull requests.

### Despliegue

* Render: Permite desplegar aplicaciones web y APIs en la nube. 

## Estructura del Repositorio

El repositorio del proyecto incluye los siguientes archivos:

*Data_movies_.parquet: Dataset final en formato Parquet.

*EDA.ipynb: Notebook con el análisis exploratorio de datos.

*ETL.ipynb: Notebook con el proceso de extracción, 

*Funciones.py: Script con las funciones de recomendación y otras consultas adicionales.

*README.md: Este archivo.

*requirements.txt: Archivo con las librerias del proyecto.

## Proceso ETL (Extracción, Transformación y Carga)

Se configuró un entorno llamado work_space con las librerías necesarias, se cargaron los datasets credits.csv y movies_dataset.csv que fuerón entregados por la comñia en el siguiente drive [Datasets](https://drive.google.com/drive/folders/1X_LdCoGTHJDbD28_dJTxaD4fVuQC9Wt5). Se realizó una exploración inicial para identificar tipos de datos, valores nulos y duplicados. En cuanto al tratamiento de valores nulos, se rellenaron con ceros las columnas revenue y budget, mientras que otras columnas con valores nulos no fueron eliminadas para evitar la pérdida de información relevante. Las fechas se convirtieron al formato AAAA-mm-dd, y se creó una columna release_year con el año de estreno. Además, se calculó la columna return como revenue / budget. 

Por otra parte las columnas innecesarias como video, imdb_id, adult, original_title, poster_path y homepage fueron eliminadas. Luego se desanidaron los campos belongs_to_collection, genres, spoken_languages, production_companies y production_countries. Posteriormente, se fusionaron los datasets procesados y se organizó el orden de las columnas para facilitar el trabajo posterior. Finalmente, el dataset resultante se guardó en el archivo Data_movies_.parquet para su uso en el desarrollo de funciones de recomendación.

## Proceso EDA (Análisis Exploratorio de Datos)

El EDA permitió obtener insights cruciales para el desarrollo del modelo de recomendación:

* Resumen estadístico (runtime, budget, revenue, return, popularity, vote_average):
Importancia:
El resumen estadístico ofrece una vista general de la distribución de los datos y ayuda a identificar tendencias y valores extremos (outliers). Conocer los valores promedio, medianos y los rangos de estas variables permite entender el comportamiento típico de las películas, lo cual es esencial para construir un modelo de recomendación que refleje fielmente las preferencias del usuario.

* Correlaciones clave (budget y revenue, vote_count y budget):
  
<center>

![Image Bonita](image_proyect\correlacion_variables.png)
</center>

Esto Identificara las correlaciones que funcionan para entender cómo las variables influyen entre sí. Una alta correlación entre el presupuesto (budget) y los ingresos (revenue) indica que películas con mayor presupuesto tienden a generar más ingresos, lo que puede ser un factor a considerar en el modelo de recomendación.
La correlación entre vote_count (cantidad de votos) y budget sugiere que las películas con mayor presupuesto suelen tener más alcance y, por ende, reciben más votos. Estos insights ayudan a ajustar el modelo para que no solo recomiende películas con alta popularidad, sino que también tenga en cuenta el contexto financiero de las producciones.

* Estados de las películas:

<center>

![Image Bonita](image_proyect\estado_peliculas.png)
</center>

Se Analizarón los diferentes estados de las películas (Released, Rumored, Post production, etc.) es esencial para evitar recomendar películas que aún no han sido estrenadas o que están en un estado de producción incierto. Este filtro mejora la experiencia del usuario al garantizar que solo se recomienden películas disponibles para ver.

* Análisis de presupuesto a lo largo del tiempo:

<center>

![Image Bonita](image_proyect\regresion_lineal_presuvsingre.png)

![Image Bonita](image_proyect\ingreso_años.png)

![Image Bonita](image_proyect\presupuesto_años.png)

![Image Bonita](image_proyect\pesupuesto_vs_ingreso.png)
</center>

Se Evaluo cómo ha evolucionado el presupuesto cinematográfico con el tiempo permite entender las tendencias de producción en diferentes épocas. Esto es importante para ajustar el modelo y recomendar películas de acuerdo con las preferencias temporales del usuario. 

* Word clouds (genres_names y overview):
<center>

![Image Bonita](image_proyect\wordcloud_genres.png)

![Image Bonita](image_proyect\wordcloud_overview.png)
</center>

Estas nubes de palabras proporcionan una representación visual de los términos más frecuentes en géneros y descripciones (overview). Esto ayuda a identificar patrones comunes en el contenido, lo que es útil para personalizar las recomendaciones basadas en los géneros o temáticas que el usuario prefiere.

* Análisis de actores y directores:
<center>

![Image Bonita](image_proyect\actores_más_comunes.png)

![Image Bonita](image_proyect\directores_mas_comunes.png)
</center>

Este análisis permite identificar los actores y directores más influyentes o recurrentes en la base de datos, lo que puede ser un factor decisivo en las recomendaciones. 
  
* Verificación de outliers:
<center>

![Image Bonita](image_proyect\outliers_variables.png)
</center>

 la Identificación y manejo de outliers es fundamental para evitar que valores extremos (que no representan el comportamiento típico de las películas) distorsionen el modelo de recomendación.Sin embargo, aunque es necesario identificar los outliers en columnas de valores como el presupuesto o los ingresos, no se imputa información faltante en ciertos casos, como en las votaciones de las películas. Esto se debe a que imputar datos en métricas como las votaciones podría alterar las métricas existentes, especialmente si las películas aún no han recibido calificaciones. En estos casos, es más adecuado mantener los valores como faltantes en lugar de introducir información que no refleje la realidad, asegurando que las recomendaciones sean más representativas y precisas.


## Desarrollo de la API

Se desarrollaron 6 endpoints principales utilizando FastAPI, que se explican en el siguiente link [Ver Video en YouTube](https://www.youtube.com/watch?v=MBJMwcxAMpo) : 

### /cantidad_filmaciones_mes/:
 Devuelve el número de películas estrenadas en un mes específico.
### /cantidad_filmaciones_dia/:
 Devuelve el número de películas estrenadas en un día específico de la semana.
### /score_titulo/: 
Proporciona el año de estreno y el puntaje de popularidad de una película específica.
### /votos_titulo/: 
Informa sobre el año de estreno, número de votos y calificación promedio de una película con más de 2000 valoraciones.
### /get_actor/: 
Devuelve el número de películas en las que ha participado un actor y su retorno de inversión.
### /get_director/: 
Proporciona una lista de películas dirigidas por un director, su retorno total y promedio.

## Despliegue

El proyecto se desplegó utilizando Render. El API está disponible para su consumo en el siguiente enlace: [Render](https://proyect-0227.onrender.com/docs).

## Contribuciones

Cualquier contribución para mejorar este proyecto es bienvenida. Por favor, realiza un fork del repositorio y envía un pull request con tus cambios.

## Contacto

Para cualquier duda o sugerencia, puedes contactarme a través de mi perfil de [Github Juan](https://github.com/JPjuanaponte).
<center>

**JUAN PABLO APONTE MURCIA**
</center>