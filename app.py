import streamlit as st
import pandas as pd
from database import get_all_documents, search_documents, get_document_counts
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import datetime  # For handling date input
import re  # For regex date processing

import nltk
nltk.download('punkt')

from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
import matplotlib.pyplot as plt
import streamlit as st

COLOR_PALETTE = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

def main():
    st.title("Opinión de los Pacientes CESFAM: Visualización Interactiva")


    # Sidebar Filters
    st.sidebar.header("Filtros")
    query = st.sidebar.text_input("Buscar")
    columns = ['Todo', 'Edad', 'Género', 'Centro de Salud', 'Frecuencia', 'Satisfacción', 'Recomendación', 'Comentario', 'Etiqueta']
    selected_column = st.sidebar.selectbox("Buscar en", columns)

    use_date_filter = st.sidebar.checkbox("Habilitar filtro de fecha")

    if use_date_filter:
        date_range = st.sidebar.date_input(
            "Seleccione rango de fechas",
            (datetime.date.today() - datetime.timedelta(days=7), datetime.date.today())
        )
        
        if len(date_range) == 2:  # Check if two dates are returned
            start_date, end_date = date_range
        elif len(date_range) == 1:  # Check if one date is returned
            start_date = end_date = date_range[0]  # Use the single date as both start and end dates
        else:
            start_date = end_date = None

        if query or (start_date and end_date):  # Ensure the date filter is used if enabled
            data = search_documents(query, selected_column, start_date, end_date)
        else:
            data = get_all_documents()
    else:
        start_date = None
        end_date = None
        data = get_all_documents()


    # data

    df = pd.DataFrame(data)


    # Convert columns
    if 'date' in df.columns:
        df['Fecha'] = df['date'].apply(convert_date)
        df.drop(columns=['date'], inplace=True)

    if 'target' in df.columns:
        df['Etiqueta'] = df['target'].apply(convert_target)
        df.drop(columns=['target'], inplace=True)  # Dropping the 'target' column after creating 'Etiqueta'


    columns_to_rename = {
    'edad': 'Edad',
    'genero': 'Género',
    'cesfam': 'CESFAM',
    'frecuencia': 'Frecuencia',
    'satisfaccion': 'Satisfacción',
    'recomendacion': 'Recomendación',
    'razon': 'Razón'
    }
    df.rename(columns=columns_to_rename, inplace=True)

    # Gender Filter
    selected_gender = 'Todos'  # Default value
    if 'Genero' in df.columns:
        gender_options = ['Todos'] + sorted(df['Genero'].unique())
        selected_gender = st.sidebar.selectbox('Género', gender_options)
    if selected_gender != 'Todos':
        df = df[df['Genero'] == selected_gender]

    # Age Range Filter
    if 'Edad' in df.columns:  # Check if 'Edad' column exists
        min_age, max_age = st.sidebar.slider(
            "Rango de Edad",
            0, 120, (0, 120)
        )
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
        etiqueta_options = ['Todos'] + sorted(df['Etiqueta'].unique())
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
        satisfaccion_options = ['Todos'] + sorted(df['Satisfacción'].unique().astype(str))
        selected_satisfaccion = st.sidebar.selectbox('Satisfacción', satisfaccion_options)
    if selected_satisfaccion != 'Todos':
        df = df[df['Satisfacción'] == selected_satisfaccion]

    # Recomendacion Filter
    selected_recomendacion = 'Todos'
    if 'Recomendación' in df.columns:
        recomendacion_options = ['Todos'] + sorted(df['Recomendación'].unique().astype(str))
        selected_recomendacion = st.sidebar.selectbox('Recomendación', recomendacion_options)
    if selected_recomendacion != 'Todos':
        df = df[df['Recomendación'] == selected_recomendacion]




    # Display the data
    # Plot data if available

    if df.empty:
        st.write("No hay datos disponibles para los filtros seleccionados. Ajusta los filtros e intenta nuevamente.")
    else:

        st.write(df.drop(columns=['_id'], errors='ignore'))
        plot_nube_de_palabras(df, n=3)
        plot_distribucion_sentimientos(df)
        plot_satisfaccion_y_recomendacion(df)
        plot_distribucion_frecuencia(df)
        plot_distribucion_edad(df)
        plot_distribucion_genero(df)
        plot_frecuencia_visitas(df)
        plot_feedback_por_centro(df)
        plot_promedio_satisfaccion_recomendacion(df)
        plot_cronologia_feedback(df)
        plot_grafico_3d(df)




def convert_date(date):
    if not isinstance(date, datetime.datetime):
        date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    return date.strftime('%d %B %Y')

def convert_target(target):
    target_map = {0: 'Irrelevante', 1: 'Negativo', 2: 'Positivo', 3: 'Sin categorizar'}
    return target_map.get(target, 'Desconocido')

def plot_distribucion_sentimientos(df):
    fig = px.pie(df, names='Etiqueta', title='Distribución de Sentimientos', hole=0.3, color_discrete_sequence=COLOR_PALETTE)
    st.plotly_chart(fig, use_container_width=True)

def plot_nube_de_palabras(df, n=2):
    # Join all the comments into a single text
    text = ' '.join(df['Razón'].dropna())
    
    # Tokenize the text into words
    words = word_tokenize(text)
    
    # Create bigrams or n-grams (you can adjust n for different sizes of phrases)
    n_grams = ngrams(words, n)
    
    # Join the n-grams into phrases
    phrases = (' '.join(grams) for grams in n_grams)
    
    # Create a frequency distribution of the phrases
    phrase_freq = Counter(phrases)
    
    # Generate the word cloud with the specified maximum number of words
    wc = WordCloud(width=800, height=800, max_words=100, background_color='white').generate_from_frequencies(phrase_freq)
    
    # Plot the word cloud using matplotlib
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    
    # Display the figure with streamlit
    st.pyplot(fig)


def plot_distribucion_edad(df):
    fig = px.histogram(df, x='Edad', nbins=20, title='Distribución de Edad', color_discrete_sequence=COLOR_PALETTE)
    st.plotly_chart(fig, use_container_width=True)

def plot_satisfaccion_y_recomendacion(df):
    fig = px.bar(df, x=['Satisfacción', 'Recomendación'], title='Calificaciones de Satisfacción y Recomendación', color_discrete_sequence=COLOR_PALETTE)
    st.plotly_chart(fig, use_container_width=True)

def plot_distribucion_frecuencia(df):
    fig = px.pie(df, names='Frecuencia', title='Distribución de Frecuencia', hole=0.3, color_discrete_sequence=COLOR_PALETTE)
    st.plotly_chart(fig, use_container_width=True)

def plot_distribucion_genero(df):
    gender_counts = df['Género'].value_counts()
    fig = px.pie(names=gender_counts.index, values=gender_counts.values, title='Distribución de Género', color_discrete_sequence=COLOR_PALETTE)
    st.plotly_chart(fig, use_container_width=True)

def plot_feedback_por_centro(df):
    center_counts = df['CESFAM'].value_counts()
    fig = px.bar(x=center_counts.index, y=center_counts.values, title='Feedback por Centro de Salud', color_discrete_sequence=COLOR_PALETTE)
    st.plotly_chart(fig, use_container_width=True)

def plot_frecuencia_visitas(df):
    frequency_counts = df['Frecuencia'].value_counts()
    fig = px.pie(names=frequency_counts.index, values=frequency_counts.values, title='Frecuencia de Visitas', color_discrete_sequence=COLOR_PALETTE)
    st.plotly_chart(fig, use_container_width=True)

def plot_promedio_satisfaccion_recomendacion(df):
    avg_ratings = df.groupby('CESFAM')[['Satisfacción', 'Recomendación']].mean().reset_index()
    fig = px.bar(avg_ratings, x='CESFAM', y=['Satisfacción', 'Recomendación'], title='Promedio de Satisfacción y Recomendación por Centro de Salud', color_discrete_sequence=COLOR_PALETTE)
    st.plotly_chart(fig, use_container_width=True)

def plot_cronologia_feedback(df):
    feedback_dates = df['Fecha'].value_counts().sort_index()
    fig = px.line(x=feedback_dates.index, y=feedback_dates.values, title='Cronología de Feedback', color_discrete_sequence=COLOR_PALETTE)
    st.plotly_chart(fig, use_container_width=True)

def plot_grafico_3d(df):
    fig = go.Figure(data=[go.Scatter3d(
        x=df['Edad'], 
        y=df['Satisfacción'], 
        z=df['Recomendación'], 
        mode='markers',
        marker=dict(size=5),
    )])
    fig.update_layout(title='Edad vs Satisfacción vs Recomendación', scene=dict(
        xaxis_title='Edad',
        yaxis_title='Satisfacción',
        zaxis_title='Recomendación'
    ))
    st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
