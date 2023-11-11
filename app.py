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

COLOR_PALETTE = [
    '#636EFA',  # Blue
    '#EF553B',  # Red
    '#00CC96',  # Green
    '#AB63FA',  # Purple
    '#FFA15A'   # Orange
]

def main():
    st.title("Opinión de los Pacientes CESFAM: Visualización Interactiva")


  # Sidebar - Filters
    st.sidebar.header("Filtros")
    query = st.sidebar.text_input("Buscar")
    columns = ['Todo', 'Edad', 'Género', 'Centro de Salud', 'Frecuencia', 'Satisfacción', 'Recomendación', 'Comentario', 'Etiqueta']
    selected_column = st.sidebar.selectbox("Buscar en", columns)

    use_date_filter = st.sidebar.checkbox("Habilitar filtro de fecha")
    start_date = end_date = None

    # Get date range if the date filter is enabled
    if use_date_filter:
        date_range = st.sidebar.date_input(
            "Seleccione rango de fechas",
            (datetime.date.today() - datetime.timedelta(days=7), datetime.date.today())
        )
        if len(date_range) == 2:
            start_date, end_date = date_range

    # Call search_documents if there is a query, or if the date filter is enabled and dates are selected
    if query or (use_date_filter and start_date and end_date):
        data = search_documents(query, selected_column, start_date, end_date)
    else:
        # If there is no query and the date filter is not used, get all documents
        data = get_all_documents()

    # Data processing
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




    # Display the data
    # Plot data if available

    if df.empty:
        st.write("No hay datos disponibles para los filtros seleccionados. Ajusta los filtros e intenta nuevamente.")
    else:
        
        st.write(df.drop(columns=['_id'], errors='ignore'))
        plot_nube_de_palabras(df, n=3)
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




def convert_date(date):
    if not isinstance(date, datetime.datetime):
        date = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    return date.strftime('%d %B %Y')

def convert_target(target):
    target_map = {0: 'Irrelevante', 1: 'Negativo', 2: 'Positivo', 3: 'Sin categorizar'}
    return target_map.get(target, 'Desconocido')

def plot_distribucion_sentimientos(df):
    # Define your color mapping here
    color_map = {
        'Negativo': '#EF553B',      # Example color for 'Negativo'
        'Positivo': '#00CC96',    # Example color for 'Positivo'
        'Irrelevante': '#636EFA',  # Example color for 'Irrelevante'
        'Sin categorizar': 'white'  # Example color for 'Sin categorizar'
    }

    # Create the pie chart with the custom color map
    fig = px.pie(df, names='Etiqueta', title='Distribución de Sentimientos', hole=0.3, color='Etiqueta', color_discrete_map=color_map)

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

def get_nps_color(score):
    if score >= 50:
        return '#00CC96'  # Green
    elif score >= 0:
        return '#FFA15A'  # Orange
    else:
        return '#EF553B'  # Red

def plot_nps_per_cesfam(df):
    # Categorize recommendations
    df['NPS Category'] = pd.cut(df['Recomendación'], bins=[0, 6, 8, 10], labels=['Detractors', 'Passives', 'Promoters'])

    # Group by CESFAM and NPS Category and count the occurrences
    nps_counts = df.groupby(['CESFAM', 'NPS Category']).size().reset_index(name='Counts')

    # Calculate NPS for each CESFAM
    nps_scores = nps_counts.pivot(index='CESFAM', columns='NPS Category', values='Counts').fillna(0)
    nps_scores['NPS Score'] = (nps_scores['Promoters'] - nps_scores['Detractors']) / nps_scores.sum(axis=1) * 100
    
    # Add a column with the color based on the NPS score
    nps_scores['Color'] = nps_scores['NPS Score'].apply(get_nps_color)
    
    # Create the bar chart for NPS per CESFAM with conditional colors
    fig = px.bar(nps_scores.reset_index(), x='CESFAM', y='NPS Score', title='Net Promoter Score (NPS) por CESFAM',
                 color='Color', color_discrete_map='identity')
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

def get_csat_color(score):
    if score >= 50:
        return '#00CC96'  # Green
    elif score >= 20:
        return '#FFA15A'  # Orange
    else:
        return '#EF553B'  # Red

def plot_csat_per_cesfam(df):
    # Filter responses that are 4 or 5 for each CESFAM
    df['CSAT Category'] = pd.cut(df['Satisfacción'], bins=[0, 3, 4, 5], labels=['Unsatisfied', 'Satisfied', 'Very Satisfied'])
    csat_counts = df.groupby(['CESFAM', 'CSAT Category']).size().reset_index(name='Counts')
    
    # Calculate CSAT for each CESFAM
    csat_scores = csat_counts.pivot(index='CESFAM', columns='CSAT Category', values='Counts').fillna(0)
    csat_scores['CSAT Score'] = (csat_scores['Satisfied'] + csat_scores['Very Satisfied']) / csat_scores.sum(axis=1) * 100

    # Add a column with the color based on the CSAT score
    csat_scores['Color'] = csat_scores['CSAT Score'].apply(get_csat_color)
    
    # Create the bar chart for CSAT per CESFAM with conditional colors
    fig = px.bar(csat_scores.reset_index(), x='CESFAM', y='CSAT Score', title='Customer Satisfaction Score (CSAT) por CESFAM',
                 color='Color', color_discrete_map='identity')
    
    # Display the chart
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


def plot_promedio_etiqueta(df):
    # Define your color mapping here
    color_map = {
        'Negativo': '#EF553B',      # Example color for 'Negativo'
        'Positivo': '#00CC96',    # Example color for 'Positivo'
        'Irrelevante': '#636EFA',  # Example color for 'Irrelevante'
        'Sin categorizar': 'white'  # Example color for 'Sin categorizar'
    }

    # Ensure the 'Etiqueta' column is a category type with the desired order
    category_order = [ 'Sin categorizar','Irrelevante', 'Negativo', 'Positivo']
    df['Etiqueta'] = pd.Categorical(df['Etiqueta'], categories=category_order, ordered=True)

    # Group by 'CESFAM' and 'Etiqueta' and count the occurrences
    etiqueta_counts = df.groupby(['CESFAM', 'Etiqueta']).size().reset_index(name='Cantidad')

    # Create the bar chart with the custom color map
    fig = px.bar(etiqueta_counts, x='CESFAM', y='Cantidad', color='Etiqueta', title='Distribución de Sentimientos por Centro de Salud',
                 color_discrete_map=color_map)

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)



def plot_distribucion_edad(df):
    fig = px.histogram(df, x='Edad', nbins=20, title='Distribución de Edad', 
                       labels={'count': 'Cantidad'},  # Change Y-axis label to 'Cantidad'
                       color_discrete_sequence=COLOR_PALETTE)
    
    # Update layout for custom axis titles
    fig.update_layout(
        yaxis_title='Cantidad',  # Set the Y-axis title
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_distribucion_frecuencia(df):
    fig = px.pie(df, names='Frecuencia', title='Distribución de Frecuencia', hole=0.3, color_discrete_sequence=COLOR_PALETTE)
    st.plotly_chart(fig, use_container_width=True)

def plot_distribucion_genero(df):
    gender_counts = df['Género'].value_counts()
    fig = px.pie(names=gender_counts.index, values=gender_counts.values, title='Distribución de Género', color_discrete_sequence=COLOR_PALETTE)
    st.plotly_chart(fig, use_container_width=True)

def plot_feedback_por_centro(df):
    center_counts = df['CESFAM'].value_counts().reset_index()
    center_counts.columns = ['CESFAM', 'Cantidad']

    # Define your custom color palette
    CUSTOM_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Create the bar chart with the selected color palette
    fig = px.bar(center_counts, x='CESFAM', y='Cantidad', title='Feedback por Centro de Salud', color='CESFAM',
                 color_discrete_sequence=CUSTOM_PALETTE)

    # Remove the legend
    fig.update_layout(showlegend=False)

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)


def plot_cronologia_feedback(df):
    # Ensure 'Fecha' is a datetime column
    df['Fecha'] = pd.to_datetime(df['Fecha'])

    # Now, when you sort, it will be in chronological order
    feedback_dates = df['Fecha'].dt.date.value_counts().sort_index()

    # Choose a color for the line
    line_color = '#1f77b4'  # Replace with any color you prefer

    # Create the line chart
    fig = px.line(feedback_dates, x=feedback_dates.index, y=feedback_dates.values,
                  title='Cronología de Feedback', line_shape='linear', 
                  markers=True, color_discrete_sequence=[line_color])

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)


def plot_grafico_3d(df):
    # Create a color scale based on the 'Recomendación' values
    # Normalize 'Recomendación' to range between the min and max values for the color scale
    recomendacion_normalized = (df['Recomendación'] - df['Recomendación'].min()) / (df['Recomendación'].max() - df['Recomendación'].min())

    # Create the figure
    fig = go.Figure(data=[go.Scatter3d(
        z=df['Edad'], 
        x=df['Satisfacción'], 
        y=df['Recomendación'], 
        mode='markers',
        marker=dict(
            size=5,
            color=recomendacion_normalized,  # Set color to the normalized 'Recomendación'
            colorscale='Hot',  # Use the 'Inferno' colorscale
            colorbar_title='Recomendación'
        ),
    )])

    # Update the layout with a specified height, e.g., 800 pixels
    fig.update_layout(
        title='Edad vs Satisfacción vs Recomendación',
        scene=dict(
            zaxis_title='Edad',
            xaxis_title='Satisfacción',
            yaxis_title='Recomendación'
        ),
        height=800  # Set the height of the figure
    )

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)


def plot_nps_chart(df):
    # Categorize recommendations
    df['NPS Category'] = pd.cut(df['Recomendación'], bins=[0, 6, 8, 10], labels=['Detractors', 'Passives', 'Promoters'])

    # Calculate NPS
    nps_score = ((df['NPS Category'].value_counts()['Promoters'] - df['NPS Category'].value_counts()['Detractors']) / len(df)) * 100

    # Prepare data for the chart
    nps_data = df['NPS Category'].value_counts().reset_index()
    nps_data.columns = ['Category', 'Count']

    # Define color mapping for each category based on your COLOR_PALETTE
    nps_color_map = {
        'Detractors': '#EF553B',  # Red
        'Passives': '#FFA15A',    # Orange
        'Promoters': '#00CC96',   # Green
    }

    # Create the hollow pie chart with the custom color map
    fig = px.pie(nps_data, names='Category', values='Count', title=f'Net Promoter Score (NPS): {nps_score:.2f}', hole=0.4,
                 color='Category', color_discrete_map=nps_color_map)

    # Add the overall NPS in the middle of the chart
    fig.update_traces(textinfo='label+percent', textposition='inside')
    fig.update_layout(annotations=[dict(text=f'NPS<br>{nps_score:.2f}', x=0.5, y=0.5, font_size=20, showarrow=False)])

    st.plotly_chart(fig, use_container_width=True)

def plot_csat_score(df):
    # Assuming 'Satisfacción' is the column with satisfaction scores from 1 to 5
    # Filter responses that are 4 or 5
    satisfied_responses = df[df['Satisfacción'] >= 4]

    # Calculate the Top 2 Boxes CSAT Score
    t2b_csat_score = (len(satisfied_responses) / len(df)) * 100
    csat_color = get_csat_color(t2b_csat_score)  # Get the color based on the score

    # Prepare data for the chart
    csat_data = {
        'Category': ['CSAT Score'],
        'Score': [t2b_csat_score],
        'Color': [csat_color]  # Add the color here
    }
    csat_df = pd.DataFrame(csat_data)

    # Create the bar chart
    fig = px.bar(csat_df, x='Category', y='Score', title=f'Customer Satisfaction (CSAT) Score: {t2b_csat_score:.2f}%',
                 text='Score', range_y=[0, 100], color='Color', color_discrete_map='identity')  # Use color column

    # Update the layout to show the score inside the bar
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

  

if __name__ == '__main__':
    main()
