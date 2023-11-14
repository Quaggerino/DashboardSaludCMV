# Imports and Constants
import folium
from streamlit_folium import folium_static

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import streamlit as st
import pandas as pd
import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from database import get_all_documents, search_documents, get_document_counts
from wordcloud import WordCloud

from nltk.corpus import stopwords
nltk.download('stopwords')

import nltk
nltk.download('punkt')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ENG: Color palette for charts
# ESP: Paleta de colores para los gráficos
COLOR_PALETTE = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

lista_censura = ['sharp','alcalde','tele','concejales', 'concejal','ambulantes',]

cesfam_coords = [
    {"CESFAM": "CECOSF Porvenir Bajo", "lat": -33.033539, "lon": -71.6498462},
    {"CESFAM": "CESFAM Placilla", "lat": -33.1123818, "lon": -71.566229},
    {"CESFAM": "CESFAM Quebrada Verde", "lat": -33.042893, "lon": -71.6467337},
    {"CESFAM": "CESFAM Cordillera", "lat": -33.0432288, "lon": -71.636744},
    {"CESFAM": "CESFAM Marcelo Mena", "lat": -33.0489701, "lon": -71.6312418},
    {"CESFAM": "CESFAM Puertas Negras", "lat": -33.0582224, "lon": -71.6386927},
    {"CESFAM": "CESFAM Padre Damián", "lat": -33.0578, "lon": -71.5618254},
    {"CESFAM": "CESFAM Las Cañas", "lat": -33.057986, "lon": -71.6080029},
    {"CESFAM": "CESFAM Rodelillo", "lat": -33.058831, "lon": -71.5771786},
    {"CESFAM": "CESFAM Reina Isabel", "lat": -33.0608925, "lon": -71.5924971},
    {"CESFAM": "CESFAM Placeres", "lat": -33.0457555, "lon": -71.5835698},
    {"CESFAM": "CESFAM Esperanza", "lat": -33.0334053, "lon": -71.5824393},
    {"CESFAM": "CESFAM Barón", "lat": -33.0393692, "lon": -71.6008781},
    {"CESFAM": "CECOSF Laguna Verde", "lat": -33.104275, "lon": -71.667925},
    {"CESFAM": "CECOSF Juan Pablo II", "lat": -33.0622632, "lon": -71.5631436}
]

cesfam_df = pd.DataFrame(cesfam_coords)

# Function to calculate CSAT scores
def calculate_csat(df):
    # Categorize satisfaction scores
    df['Categoría CSAT'] = pd.cut(df['Satisfacción'], bins=[0, 3, 4, 5], labels=['Insatisfecho', 'Satisfecho', 'Muy Satisfecho'])

    # Group by CESFAM and count occurrences in each category
    grouped = df.groupby('CESFAM')['Categoría CSAT'].value_counts().unstack(fill_value=0)

    # Ensure all categories are present
    categories = ['Insatisfecho', 'Satisfecho', 'Muy Satisfecho']
    grouped = grouped.reindex(columns=categories, fill_value=0)

    # Calculate CSAT Scores
    csat_scores = (grouped['Satisfecho'] + grouped['Muy Satisfecho']) / grouped.sum(axis=1) * 100

    # Prepare a DataFrame with CESFAM, CSAT Score, and Color
    csat_df = pd.DataFrame({
        'CESFAM': csat_scores.index,
        'CSAT Score': csat_scores.values,
        'Color': csat_scores.apply(get_csat_color)
    })
    return csat_df

def calculate_nps(df):
    # Categorize recommendations
    df['NPS Category'] = pd.cut(df['Recomendación'], bins=[0, 6, 8, 10], labels=['Detractors', 'Passives', 'Promoters'])

    # Group by CESFAM and calculate NPS for each
    grouped = df.groupby('CESFAM')['NPS Category'].value_counts().unstack(fill_value=0)

    # Ensure all categories are present
    categories = ['Detractors', 'Passives', 'Promoters']
    grouped = grouped.reindex(columns=categories, fill_value=0)

    # Calculate NPS Scores
    nps_scores = (grouped['Promoters'] - grouped['Detractors']) / grouped.sum(axis=1) * 100

    # Prepare a DataFrame with CESFAM, NPS Score, and Color
    nps_df = pd.DataFrame({
        'CESFAM': nps_scores.index,
        'NPS Score': nps_scores.values,
        'Color': nps_scores.apply(get_nps_color)
    })
    return nps_df

# Mapa CESFAM
import folium
from streamlit_folium import folium_static
from branca.element import IFrame

def plot_cesfam_map(df, cesfam_df):
    nps_df = calculate_nps(df)

    # Reset the index if needed
    if nps_df.index.name == 'CESFAM':
        nps_df = nps_df.reset_index(drop=True)
    if cesfam_df.index.name == 'CESFAM':
        cesfam_df = cesfam_df.reset_index(drop=True)

    # Merge the dataframes
    merged_df = pd.merge(cesfam_df, nps_df, on='CESFAM', how='left').dropna(subset=['lat', 'lon', 'NPS Score'])

    if not merged_df.empty:
        folium_map = folium.Map(location=[-33.060, -71.619], zoom_start=12)

        # URL to a hospital emoji image (replace with your own URL if needed)
        hospital_icon_url = 'hospital.png' 

        for _, row in merged_df.iterrows():
            # Create HTML content for popup with larger font size
            popup_html = f"<div style='font-size: 12pt;'><b>{row['CESFAM']}</b><br>NPS: {row['NPS Score']:.2f}</div>"
            popup = folium.Popup(popup_html, max_width=180)

            # Custom icon
            icon = folium.CustomIcon(hospital_icon_url, icon_size=(32, 32))  # Adjust size as needed

            # Add circle marker for NPS color
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=16,  # Outer circle size
                color=row['Color'],
                fill=True,
                fill_color=row['Color']
            ).add_to(folium_map)

            # Add marker with custom icon
            folium.Marker(
                location=[row['lat'], row['lon']],
                icon=icon,
                popup=popup
            ).add_to(folium_map)

        folium_static(folium_map)
    else:
        st.error("No matching CESFAM coordinates found for the selected data.")

# ENG: Convert date strings to formatted dates
# ESP: Convierte cadenas de fecha a fechas formateadas
def convert_date(date):
    if not isinstance(date, datetime.datetime):
        date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    return date.strftime('%d %B %Y')

# ENG: Convert target values to categorical labels
# ESP: Convierte valores de target a etiquetas categóricas
def convert_target(target):
    target_map = {0: 'Irrelevante', 1: 'Negativo', 2: 'Positivo', 3: 'Sin categorizar', 4: 'Error al categorizar'}
    return target_map.get(target, 'Desconocido')

# ENG: Plot sentiment distribution
# ESP: Trazar la distribución de sentimientos
def plot_distribucion_sentimientos(df):
    # Filter out rows where 'Etiqueta' is 'Error al categorizar'
    df_filtered = df[df['Etiqueta'] != 'Error al categorizar']

    color_map = {
        'Negativo': '#EF553B', 'Positivo': '#00CC96',
        'Irrelevante': '#636EFA', 'Sin categorizar': 'white'
    }

    fig = px.pie(df_filtered, names='Etiqueta', title='Distribución de Sentimientos',
                 hole=0.3, color='Etiqueta', color_discrete_map=color_map)
    st.plotly_chart(fig, use_container_width=True)

# ENG: Determine color based on NPS score
# ESP: Determinar color basado en el puntaje NPS
def get_nps_color(score):
    return '#00CC96' if score >= 50 else '#FFA15A' if score >= 0 else '#EF553B'

# ENG: Plot NPS per CESFAM
# ESP: Trazar NPS por CESFAM
def plot_nps_per_cesfam(df):
    # Calculate NPS using calculate_nps function
    nps_df = calculate_nps(df)

    # Create the bar chart for NPS per CESFAM with conditional colors
    fig = px.bar(nps_df, x='CESFAM', y='NPS Score', title='Net Promoter Score (NPS) por CESFAM',
                 color='Color', color_discrete_map='identity')
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)



# ENG: Determine color based on CSAT score
# ESP: Determinar color basado en el puntaje CSAT
def get_csat_color(score):
    return '#00CC96' if score >= 50 else '#FFA15A' if score >= 20 else '#EF553B'

# ENG: Plot CSAT per CESFAM
# ESP: Trazar CSAT por CESFAM
# Function to plot CSAT per CESFAM
def plot_csat_per_cesfam(df):
    # Calculate CSAT using calculate_csat function
    csat_df = calculate_csat(df)

    # Create the bar chart for CSAT per CESFAM with conditional colors
    fig = px.bar(csat_df, x='CESFAM', y='CSAT Score', title='Customer Satisfaction Score (CSAT) por CESFAM',
                 color='Color', color_discrete_map='identity')

    # Set y-axis to include negative values and zero
    fig.update_layout(yaxis=dict(range=[0, 100]))

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)



# ENG: Plot word cloud from comments
# ESP: Trazar nube de palabras de comentarios
def plot_nube_de_palabras(df, etiquetas=None, exclude_words=None, n=3):
    # Filter by specific etiquetas/targets if provided
    if etiquetas:
        df = df[df['Etiqueta'].isin(etiquetas)]

    # Join all text items into one large string
    text = ' '.join(df['Razón'].dropna()).lower()

    # Tokenize the text
    words = word_tokenize(text)

    # Exclude stopwords and specific words
    filtered_words = [word for word in words if word not in stopwords.words('spanish')]
    if exclude_words:
        filtered_words = [word for word in filtered_words if word not in exclude_words]

    # Generate n-grams
    n_grams = ngrams(filtered_words, n)
    phrases = (' '.join(grams) for grams in n_grams)

    # Generate frequency distribution
    phrase_freq = Counter(phrases)

    # Check if there are phrases to display
    if phrase_freq:
        # Create and display the word cloud
        wc = WordCloud(width=800, height=800, max_words=200, background_color='white').generate_from_frequencies(phrase_freq)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        # Display a message if there are no phrases
        st.write("No hay palabras disponibles para mostrar en la nube de palabras.")


# ENG: Plot average sentiment by health center
# ESP: Trazar sentimiento promedio por centro de salud
def plot_promedio_etiqueta(df):
    # Define your color mapping
    color_map = {
        'Negativo': '#EF553B', 'Positivo': '#00CC96',
        'Irrelevante': '#636EFA', 'Sin categorizar': 'white'
    }
    
    # Define your expected category order
    category_order = ['Sin categorizar', 'Irrelevante', 'Negativo', 'Positivo']

    # Find which categories are actually present in the DataFrame
    actual_categories = [cat for cat in category_order if cat in df['Etiqueta'].unique()]

    # Update DataFrame to use only the actual categories
    df['Etiqueta'] = pd.Categorical(df['Etiqueta'], categories=actual_categories, ordered=True)

    # Group by 'CESFAM' and 'Etiqueta' and count the occurrences
    etiqueta_counts = df.groupby(['CESFAM', 'Etiqueta'], observed=True).size().reset_index(name='Cantidad')

    # Create the bar chart using only the available categories
    fig = px.bar(etiqueta_counts, x='CESFAM', y='Cantidad', color='Etiqueta',
                 title='Distribución de Sentimientos por Centro de Salud', color_discrete_map=color_map)
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)




# ENG: Plot age distribution
# ESP: Trazar distribución de edad
def plot_distribucion_edad(df):
    fig = px.histogram(df, x='Edad', nbins=20, title='Distribución de Edad',
                       labels={'count': 'Cantidad'}, color_discrete_sequence=COLOR_PALETTE)
    fig.update_layout(yaxis_title='Cantidad')
    st.plotly_chart(fig, use_container_width=True)

# ENG: Plot frequency distribution
# ESP: Trazar distribución de frecuencia
def plot_distribucion_frecuencia(df):
    fig = px.pie(df, names='Frecuencia', title='Distribución de Frecuencia', hole=0.3, color_discrete_sequence=COLOR_PALETTE)
    st.plotly_chart(fig, use_container_width=True)

# ENG: Plot gender distribution
# ESP: Trazar distribución de género
def plot_distribucion_genero(df):
    gender_counts = df['Género'].value_counts()
    fig = px.pie(names=gender_counts.index, values=gender_counts.values, title='Distribución de Género',
                 color_discrete_sequence=COLOR_PALETTE)
    st.plotly_chart(fig, use_container_width=True)

# ENG: Plot feedback by health center
# ESP: Trazar comentarios por centro de salud
def plot_feedback_por_centro(df):
    center_counts = df['CESFAM'].value_counts().reset_index()
    center_counts.columns = ['CESFAM', 'Cantidad']

    # Create an Inferno color palette
    n_colors = len(center_counts)
    inferno = plt.get_cmap('rainbow')
    inferno_colors = [inferno(i/n_colors) for i in range(n_colors)]
    inferno_hex = [matplotlib.colors.to_hex(color) for color in inferno_colors]

    # Create the bar chart
    fig = px.bar(center_counts, x='CESFAM', y='Cantidad', title='Feedback por Centro de Salud',
                 color='CESFAM', color_discrete_sequence=inferno_hex)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ENG: Plot feedback chronology
# ESP: Trazar cronología de comentarios
def plot_cronologia_feedback(df):
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    feedback_dates = df['Fecha'].dt.date.value_counts().sort_index()
    line_color = '#1f77b4'
    fig = px.line(feedback_dates, x=feedback_dates.index, y=feedback_dates.values,
                  title='Cronología de Feedback', line_shape='linear', markers=True,
                  color_discrete_sequence=[line_color])
    st.plotly_chart(fig, use_container_width=True)

# ENG: Plot a 3D chart for Age vs Satisfaction vs Recommendation
# ESP: Trazar un gráfico 3D para Edad vs Satisfacción vs Recomendación
def plot_grafico_3d(df):
    # Apply jittering to 'Satisfacción'
    jitter_amount = 2.0  # Adjust this value as needed
    df['Satisfacción_jittered'] = df['Satisfacción'] + np.random.uniform(-jitter_amount, jitter_amount, size=len(df))

    fig = go.Figure(data=[go.Scatter3d(
        z=df['Edad'],
        x=df['Satisfacción_jittered'],
        y=df['Recomendación'],
        mode='markers',
        marker=dict(
            size=12,
            color=df['Edad'],
            colorscale='Oxy',
            colorbar_title='Edad'
        ),
        hovertemplate=(
            "Edad: %{z}<br>" +
            "Satisfacción: %{text}<br>" +
            "Recomendación: %{y}<extra></extra>"  # <extra></extra> removes the trace name
        ),
        text=df['Satisfacción']  # Use the real 'Satisfacción' values for the hover text
    )])

    fig.update_layout(
        title='Edad vs Satisfacción vs Recomendación',
        scene=dict(
            zaxis_title='Edad',
            xaxis_title='Satisfacción',
            yaxis_title='Recomendación'
        ),
        height=800
    )

    st.plotly_chart(fig, use_container_width=True)



# ENG: Plot the Net Promoter Score (NPS) chart
# ESP: Trazar el gráfico de Net Promoter Score (NPS)
def plot_nps_chart(df):
    # ENG: Categorize recommendations
    # ESP: Categorizar recomendaciones
    df['NPS Category'] = pd.cut(df['Recomendación'], bins=[0, 6, 8, 10], labels=['Detractors', 'Passives', 'Promoters'])

    # ENG: Calculate NPS
    # ESP: Calcular NPS
    nps_score = ((df['NPS Category'].value_counts()['Promoters'] - df['NPS Category'].value_counts()['Detractors']) / len(df)) * 100

    # ENG: Prepare data for the chart
    # ESP: Preparar datos para el gráfico
    nps_data = df['NPS Category'].value_counts().reset_index()
    nps_data.columns = ['Category', 'Count']

    # ENG: Define color mapping for each category
    # ESP: Definir mapeo de colores para cada categoría
    nps_color_map = {
        'Detractors': '#EF553B',  # Red
        'Passives': '#FFA15A',    # Orange
        'Promoters': '#00CC96',   # Green
    }

    # ENG: Create the hollow pie chart with the custom color map
    # ESP: Crear el gráfico circular hueco con el mapa de colores personalizado
    fig = px.pie(nps_data, names='Category', values='Count', title=f'Net Promoter Score (NPS): {nps_score:.2f}', hole=0.4,
                 color='Category', color_discrete_map=nps_color_map)

    # ENG: Add the overall NPS in the middle of the chart
    # ESP: Añadir el NPS general en el medio del gráfico
    fig.update_traces(textinfo='label+percent', textposition='inside')
    fig.update_layout(annotations=[dict(text=f'NPS<br>{nps_score:.2f}', x=0.5, y=0.5, font_size=20, showarrow=False)])

    # ENG: Display the chart
    # ESP: Mostrar el gráfico
    st.plotly_chart(fig, use_container_width=True)

# ENG: Plot the Customer Satisfaction (CSAT) Score
# ESP: Trazar el puntaje de Satisfacción del Cliente (CSAT)
def plot_csat_score(df):
    # ENG: Filter responses that are 4 or 5 (satisfied responses)
    # ESP: Filtrar respuestas que son 4 o 5 (respuestas satisfechas)
    satisfied_responses = df[df['Satisfacción'] >= 4]

    # ENG: Calculate the Top 2 Boxes CSAT Score
    # ESP: Calcular el puntaje CSAT de las dos mejores respuestas
    t2b_csat_score = (len(satisfied_responses) / len(df)) * 100

    # ENG: Get the color based on the score
    # ESP: Obtener el color basado en el puntaje
    csat_color = get_csat_color(t2b_csat_score)

    # ENG: Prepare data for the chart
    # ESP: Preparar datos para el gráfico
    csat_data = {'Category': ['CSAT Score'], 'Score': [t2b_csat_score], 'Color': [csat_color]}
    csat_df = pd.DataFrame(csat_data)

    # ENG: Create the bar chart
    # ESP: Crear el gráfico de barras
    fig = px.bar(csat_df, x='Category', y='Score', title=f'Customer Satisfaction (CSAT) Score: {t2b_csat_score:.2f}%',
                 text='Score', range_y=[0, 100], color='Color', color_discrete_map='identity')

    # ENG: Update the layout to show the score inside the bar
    # ESP: Actualizar el diseño para mostrar el puntaje dentro de la barra
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')

    # ENG: Display the chart
    # ESP: Mostrar el gráfico
    st.plotly_chart(fig, use_container_width=True)


# Main Function
def main():
    # ENG: Set the title of the Streamlit page
    # ESP: Establecer el título de la página Streamlit
    st.title("Opinión de los Pacientes CESFAM: Visualización Interactiva")

    # Sidebar - Filters
    # ENG: Create a header for filters in the sidebar
    # ESP: Crear un encabezado para los filtros en la barra lateral
    st.sidebar.header("Filtros")
    query = st.sidebar.text_input("Buscar")
    columns = ['Todo', 'Edad', 'Género', 'Centro de Salud', 'Frecuencia', 'Satisfacción', 'Recomendación', 'Comentario', 'Etiqueta']
    selected_column = st.sidebar.selectbox("Buscar en", columns)

    # ENG: Enable a date filter
    # ESP: Habilitar un filtro de fecha
    use_date_filter = st.sidebar.checkbox("Habilitar filtro de fecha")
    start_date = end_date = None

    # ENG: Get date range if the date filter is enabled
    # ESP: Obtener el rango de fechas si el filtro de fecha está habilitado
    if use_date_filter:
        date_range = st.sidebar.date_input(
            "Seleccione rango de fechas",
            (datetime.date.today() - datetime.timedelta(days=7), datetime.date.today())
        )
        if len(date_range) == 2:
            start_date, end_date = date_range

    # ENG: Call search_documents function with appropriate parameters
    # ESP: Llamar a la función search_documents con los parámetros apropiados
    if query or (use_date_filter and start_date and end_date):
        data = search_documents(query, selected_column, start_date, end_date)
    else:
        data = get_all_documents()

    # Data processing
    df = pd.DataFrame(data)

    # ENG: Convert columns to the desired format
    # ESP: Convertir columnas al formato deseado
    if 'date' in df.columns:
        df['Fecha'] = df['date'].apply(convert_date)
        df.drop(columns=['date'], inplace=True)

    if 'target' in df.columns:
        df['Etiqueta'] = df['target'].apply(convert_target)
        df.drop(columns=['target'], inplace=True)

    columns_to_rename = {
        'edad': 'Edad', 'genero': 'Género', 'cesfam': 'CESFAM', 'frecuencia': 'Frecuencia',
        'satisfaccion': 'Satisfacción', 'recomendacion': 'Recomendación', 'razon': 'Razón'
    }
    df.rename(columns=columns_to_rename, inplace=True)

    # ENG: Apply filters based on user selection
    # ESP: Aplicar filtros basados en la selección del usuario
    
    # Genero Filter
    selected_gender = 'Todos'
    if 'Genero' in df.columns:
        gender_options = ['Todos'] + sorted(df['Genero'].unique())
        selected_gender = st.sidebar.selectbox('Género', gender_options)
    if selected_gender != 'Todos':
        df = df[df['Genero'] == selected_gender]

    # Edad Range Filter
    if 'Edad' in df.columns:
        min_age, max_age = st.sidebar.slider("Rango de Edad", 0, 120, (0, 120))
        df = df[(df['Edad'] >= min_age) & (df['Edad'] <= max_age)]

    # CESFAM Filter
    selected_cesfam = 'Todos'
    if 'CESFAM' in df.columns:
        cesfam_options = ['Todos'] + sorted(df['CESFAM'].unique())
        selected_cesfam = st.sidebar.selectbox('CESFAM', cesfam_options)
    if selected_cesfam != 'Todos':
        df = df[df['CESFAM'] == selected_cesfam]

    # Etiqueta Filter
    selected_etiqueta = 'Todos'
    if 'Etiqueta' in df.columns:
        # Filter out the specific value 'Error al categorizar' from the options
        etiqueta_options = ['Todos'] + sorted([etiqueta for etiqueta in df['Etiqueta'].unique() if etiqueta != 'Error al categorizar'])

        selected_etiqueta = st.sidebar.selectbox('Etiqueta', etiqueta_options)

    if selected_etiqueta != 'Todos':
        df = df[df['Etiqueta'] == selected_etiqueta]


    # Frecuencia Filter
    selected_frecuencia = 'Todos'
    if 'Frecuencia' in df.columns:
        frecuencia_options = ['Todos'] + sorted(df['Frecuencia'].unique())
        selected_frecuencia = st.sidebar.selectbox('Frecuencia', frecuencia_options)
    if selected_frecuencia != 'Todos':
        df = df[df['Frecuencia'] == selected_frecuencia]

    # Satisfaccion Filter
    selected_satisfaccion = 'Todos'
    if 'Satisfacción' in df.columns:
        satisfaccion_options = ['Todos'] + sorted(df['Satisfacción'].unique())
        selected_satisfaccion = st.sidebar.selectbox('Satisfacción', satisfaccion_options)
    if selected_satisfaccion != 'Todos':
        df = df[df['Satisfacción'] == selected_satisfaccion]

    # Recomendacion Filter
    selected_recomendacion = 'Todos'
    if 'Recomendación' in df.columns:
        recomendacion_options = ['Todos'] + sorted(df['Recomendación'].unique())
        selected_recomendacion = st.sidebar.selectbox('Recomendación', recomendacion_options)
    if selected_recomendacion != 'Todos':
        df = df[df['Recomendación'] == selected_recomendacion]

    # ENG: Display data and plots based on the filters applied
    # ESP: Mostrar datos y gráficos basados en los filtros aplicados
    if df.empty:
        st.write("No hay datos disponibles para los filtros seleccionados. Ajusta los filtros e intenta nuevamente.")
    else:
        st.write(df.drop(columns=['_id'], errors='ignore'))
        # ENG: Call plotting functions
        # ESP: Llamar a las funciones de trazado
        plot_nube_de_palabras(df, etiquetas=['Positivo', 'Negativo'], exclude_words=lista_censura, n=3)
        if not cesfam_df.empty:
            plot_cesfam_map(df, cesfam_df)
        plot_feedback_por_centro(df)
        plot_cronologia_feedback(df)
        plot_distribucion_sentimientos(df)
        plot_promedio_etiqueta(df)
        plot_nps_chart(df)
        plot_nps_per_cesfam(df)
        plot_csat_score(df)
        plot_csat_per_cesfam(df)
        plot_distribucion_frecuencia(df)
        plot_distribucion_edad(df)
        plot_distribucion_genero(df)
        plot_grafico_3d(df)

# Running the Streamlit app
if __name__ == '__main__':
    main()
