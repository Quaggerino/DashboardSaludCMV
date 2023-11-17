# Imports and Constants
import folium
from streamlit_folium import folium_static

import numpy as np
import csv
import matplotlib.pyplot as plt
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

lista_censura = []

with open('lista_censura.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)

    for row in reader:
        lista_censura.append(row[0])


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

    # Calculate overall NPS Score for the entire dataset
    promoter_count = len(df[df['NPS Category'] == 'Promoters'])
    detractor_count = len(df[df['NPS Category'] == 'Detractors'])
    total_respondents = len(df)

    overall_nps = (promoter_count - detractor_count) / total_respondents * 100 if total_respondents > 0 else 0

    # Apply color based on overall NPS Score
    nps_color = get_nps_color(overall_nps)

    # Prepare a DataFrame with CESFAM, NPS Score, and Color
    # Since we are calculating overall NPS, the CESFAM column will be redundant.
    # However, if you still need to keep CESFAMs, we will replicate the overall NPS score across all CESFAMs.
    nps_df = pd.DataFrame({
        'CESFAM': df['CESFAM'].unique(),
        'NPS Score': [overall_nps] * len(df['CESFAM'].unique()),
        'Color': [nps_color] * len(df['CESFAM'].unique())
    })

    return nps_df


# Mapa CESFAM
import folium
from streamlit_folium import folium_static
from branca.element import IFrame

def plot_cesfam_map(df, cesfam_df):
    # Ensure 'Recomendación' column exists
    if 'Recomendación' not in df.columns:
        st.error("Column 'Recomendación' not found in the dataset.")
        return

    # Categorize recommendations
    df['NPS Category'] = pd.cut(df['Recomendación'], bins=[0, 6, 8, 10], include_lowest=True, right=True, labels=['Detractors', 'Passives', 'Promoters'])

    # Calculate NPS Score for each CESFAM
    nps_scores = df.groupby('CESFAM').apply(lambda x: (len(x[x['NPS Category'] == 'Promoters']) - len(x[x['NPS Category'] == 'Detractors'])) / len(x) * 100)
    nps_df = nps_scores.reset_index()
    nps_df.columns = ['CESFAM', 'NPS Score']
    nps_df['Color'] = nps_df['NPS Score'].apply(get_nps_color)


    # Reset the index if needed for cesfam_df
    if cesfam_df.index.name == 'CESFAM':
        cesfam_df = cesfam_df.reset_index(drop=True)

    # Merge the dataframes
    merged_df = pd.merge(cesfam_df, nps_df, on='CESFAM', how='left').dropna(subset=['lat', 'lon', 'NPS Score'])

    if not merged_df.empty:
        folium_map = folium.Map(location=[-33.065, -71.619], zoom_start=12)

        # URL to a hospital emoji image
        hospital_icon_url = 'hospital.png'

        for _, row in merged_df.iterrows():
            # Create HTML content for popup with larger font size
            popup_html = f"<div style='font-size: 12pt;'><b>{row['CESFAM']}</b><br>NPS: {row['NPS Score']:.2f}</div>"
            popup = folium.Popup(popup_html, max_width=180)

            # Custom icon
            icon = folium.CustomIcon(hospital_icon_url, icon_size=(32, 32))

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

        # Use streamlit's folium_static to display the map
        st.components.v1.html(folium_map._repr_html_(), width=700, height=420)
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
    if score >= 50:
        return '#00CC96'  # Green for high scores (Promoters)
    elif score >= 0:
        return '#FFA15A'  # Orange for middle scores (Passives)
    else:
        return '#EF553B'  # Red for low scores (Detractors)


# ENG: Plot NPS per CESFAM
# ESP: Trazar NPS por CESFAM
def plot_nps_per_cesfam(df):
    # Ensure 'Recomendación' column exists
    if 'Recomendación' not in df.columns:
        st.error("Column 'Recomendación' not found in the dataset.")
        return

    # Categorize recommendations
    df['NPS Category'] = pd.cut(df['Recomendación'], bins=[0, 6, 8, 10], include_lowest=True, right=True, labels=['Detractors', 'Passives', 'Promoters'])

    # Calculate NPS Score for each CESFAM
    nps_scores = df.groupby('CESFAM').apply(lambda x: (len(x[x['NPS Category'] == 'Promoters']) - len(x[x['NPS Category'] == 'Detractors'])) / len(x) * 100)
    nps_df = nps_scores.reset_index()
    nps_df.columns = ['CESFAM', 'NPS Score']

    # Apply get_nps_color to each NPS Score
    nps_df['Color'] = nps_df['NPS Score'].apply(get_nps_color)

    # Create the bar chart for NPS per CESFAM
    fig = go.Figure(data=[
        go.Bar(x=nps_df['CESFAM'], y=nps_df['NPS Score'], marker_color=nps_df['Color'])
    ])

    fig.update_layout(
        title='Net Promoter Score (NPS) por CESFAM',
        xaxis_title='CESFAM',
        yaxis_title='NPS Score'
    )

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
def plot_nube_de_palabras(df, etiquetas=None, exclude_words=None, n=4):
    if etiquetas:
        df = df[df['Etiqueta'].isin(etiquetas)]

    text = ' '.join(df['Razón'].dropna()).lower()
    words = word_tokenize(text)

    # Get and modify the stopwords list
    spanish_stopwords = stopwords.words('spanish')
    if 'no' in spanish_stopwords:
        spanish_stopwords.remove('no')

    # Filter words
    filtered_words = [word for word in words if word not in spanish_stopwords]
    if exclude_words:
        filtered_words = [word for word in filtered_words if word not in exclude_words]

    n_grams = ngrams(filtered_words, n)
    phrases = (' '.join(grams) for grams in n_grams)
    phrase_freq = Counter(phrases)

    if phrase_freq:
        wc = WordCloud(width=800, height=800, max_words=100, background_color='white').generate_from_frequencies(phrase_freq)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
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
                       color_discrete_sequence=COLOR_PALETTE)

    # Customize the hover template
    fig.update_traces(hovertemplate='Edad: %{x}<br>Cantidad: %{y}<extra></extra>')

    # Update layout for y-axis title
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
    # Create a DataFrame from value_counts
    gender_counts = df['Género'].value_counts().reset_index()
    gender_counts.columns = ['Género', 'Cantidad']  # Rename columns

    fig = px.pie(gender_counts, names='Género', values='Cantidad',
                 title='Distribución de Género',
                 color_discrete_sequence=COLOR_PALETTE)

    # Optionally, you can still use update_traces to further customize the hover template
    # fig.update_traces(hovertemplate="<b>Género:</b> %{label}<br><b>Cantidad:</b> %{value}<extra></extra>")

    st.plotly_chart(fig, use_container_width=True)
    
# ENG: Plot feedback by health center
# ESP: Trazar comentarios por centro de salud
def plot_feedback_por_centro(df):
    center_counts = df['CESFAM'].value_counts().reset_index()
    center_counts.columns = ['CESFAM', 'Cantidad']

    # Create the treemap with a colormap
    fig = px.treemap(center_counts, path=['CESFAM'], values='Cantidad',
                     color='Cantidad',  # Apply color based on the 'Cantidad'
                     color_continuous_scale='Thermal',  # Use Thermal colormap
                     title='Feedback por Centro de Salud',
                     hover_data={'Cantidad': ':.0f'})  # Format hover label

    # Adjust the figure size (width and height in pixels)
    fig.update_layout(width=800, height=900)

    # Hide the color scale legend and customize hover template
    fig.update_traces(hovertemplate='<b>%{label}</b><br>Cantidad: %{customdata[0]}')
    fig.update_layout(coloraxis_showscale=False)

    # Display the treemap in Streamlit
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

    # Customize the hover template
    fig.update_traces(hovertemplate='Fecha: %{x}<br>Cantidad: %{y}<extra></extra>')

    st.plotly_chart(fig, use_container_width=True)

# ENG: Plot a 3D chart for Age vs Satisfaction vs Recommendation
# ESP: Trazar un gráfico 3D para Edad vs Satisfacción vs Recomendación
def plot_grafico_3d(df):
    # Apply jittering to 'Satisfacción'
    jitter_amount = 0.8  # Maximum jitter amount
    df['Satisfacción_jittered'] = df['Satisfacción'] + np.random.uniform(0, jitter_amount, size=len(df))

    # Ensure 'Satisfacción_jittered' does not exceed the maximum value (e.g., 5)
    df['Satisfacción_jittered'] = df['Satisfacción_jittered'].clip(lower=1, upper=5)
    fig = go.Figure(data=[go.Scatter3d(
        z=df['Edad'],
        x=df['Satisfacción_jittered'],
        y=df['Recomendación'],
        mode='markers',
        marker=dict(
            size=20,
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
            zaxis=dict(title='Edad'),
            xaxis=dict(title='Satisfacción', range=[1, 5.9]),  # Set range for Satisfacción
            yaxis=dict(title='Recomendación', range=[0, 10.9])  # Set range for Recomendación
        ),
        height=800
    )

    st.plotly_chart(fig, use_container_width=True)



# ENG: Plot the Net Promoter Score (NPS) chart
# ESP: Trazar el gráfico de Net Promoter Score (NPS)
def plot_nps_chart(df):
    # Ensure 'Recomendación' column exists
    if 'Recomendación' not in df.columns:
        st.error("Column 'Recomendación' not found in the dataset.")
        return

    # Categorize recommendations
    df['NPS Category'] = pd.cut(df['Recomendación'], bins=[0, 6, 8, 10], include_lowest=True, right=True, labels=['Detractors', 'Passives', 'Promoters'])

    # Check if NPS Category has valid data
    if df['NPS Category'].isnull().all():
        st.error("No valid data for NPS categorization.")
        return

    # Calculate overall NPS Score
    promoter_count = len(df[df['NPS Category'] == 'Promoters'])
    detractor_count = len(df[df['NPS Category'] == 'Detractors'])
    total_respondents = len(df)

    # Avoid division by zero
    if total_respondents == 0:
        st.error("No respondents in the data.")
        return

    overall_nps = (promoter_count - detractor_count) / total_respondents * 100

    # Prepare data for the pie chart
    nps_data = df['NPS Category'].value_counts().reset_index()
    nps_data.columns = ['Category', 'Count']

    # Define color mapping for each category
    nps_color_map = {
        'Detractors': '#EF553B',  # Red
        'Passives': '#FFA15A',    # Orange
        'Promoters': '#00CC96',   # Green
    }

    # Create the hollow pie chart with the custom color map
    fig = px.pie(nps_data, names='Category', values='Count', title=f'Net Promoter Score (NPS): {overall_nps:.2f}', hole=0.4,
                 color='Category', color_discrete_map=nps_color_map)

    # Add the overall NPS in the middle of the chart
    fig.update_traces(textinfo='label+percent', textposition='inside')
    fig.update_layout(annotations=[dict(text=f'NPS<br>{overall_nps:.2f}', x=0.5, y=0.5, font_size=20, showarrow=False)])

    # Display the chart
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
    csat_data = {'Categoria': ['CSAT Score'], 'Score': [t2b_csat_score], 'Color': [csat_color]}
    csat_df = pd.DataFrame(csat_data)

    # ENG: Create the bar chart
    # ESP: Crear el gráfico de barras
    fig = px.bar(csat_df, x='Categoria', y='Score', title=f'Customer Satisfaction (CSAT) Score: {t2b_csat_score:.2f}%',
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


    # Button to refresh data
    # Sidebar refresh button
    # Inject custom CSS to style the sidebar button
    if st.sidebar.button('Actualizar Datos', use_container_width=True):
        # Clear the cache of the functions
        get_all_documents.clear()
        search_documents.clear()
        # Reload the data
        st.experimental_rerun()
    
            
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

    # Genero Filter
    selected_gender = 'Todos'
    if 'Género' in df.columns:
        gender_options = ['Todos'] + sorted(df['Género'].unique())
        selected_gender = st.sidebar.selectbox('Género', gender_options)
    if selected_gender != 'Todos':
        df = df[df['Género'] == selected_gender]

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
        # Filter out '0' and 0 from the unique values of 'Recomendación'
        unique_recomendaciones = [recomendacion for recomendacion in sorted(df['Recomendación'].unique()) if recomendacion != '0' and recomendacion != 0]
        
        # Add 'Todos' to the options and assign to recomendacion_options
        recomendacion_options = ['Todos'] + unique_recomendaciones

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
        n_value = st.slider("Ajustar Nivel de Detalle de la Nube de Palabras:", 1, 4, 4)
        plot_nube_de_palabras(df, etiquetas=['Positivo', 'Negativo'], exclude_words=lista_censura, n=n_value)
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
