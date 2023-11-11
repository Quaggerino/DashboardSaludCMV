# Imports
import os
import streamlit as st
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import timedelta
import logging
import pandas as pd

# ENG: Configure logging
# ESP: Configurar el registro de eventos
logging.basicConfig(level=logging.INFO)

# ENG: Load environment variables from .env.development
# ESP: Cargar variables de entorno de .env.development
load_dotenv('.env.development')

# ENG: MongoDB connection setup
# ESP: Configuración de la conexión a MongoDB
uri = os.getenv('MONGODB_URI')
client = MongoClient(uri)
db = client['cmvalparaisoDas']
collection = db['opinionSaludValparaiso']

# ENG: Function to search documents in MongoDB based on various filters
# ESP: Función para buscar documentos en MongoDB basada en varios filtros
@st.cache_data
def search_documents(query, column='Todo', start_date=None, end_date=None):
    # ENG: Mapping of human-readable columns to database fields
    # ESP: Mapeo de columnas legibles por humanos a campos de base de datos
    column_mapping = {
        'ID': '_id', 'Edad': 'edad', 'Género': 'genero', 'Centro de Salud': 'cesfam',
        'Frecuencia': 'frecuencia', 'Satisfacción': 'satisfaccion', 'Recomendación': 'recomendacion',
        'Comentario': 'razon', 'Fecha': 'date', 'Etiqueta': 'target'
    }

    # ENG: Helper function to convert values to integers if possible
    # ESP: Función auxiliar para convertir valores a enteros si es posible
    def try_int(val):
        try:
            return int(val)
        except ValueError:
            return None

    db_column = column_mapping.get(column, column)
    query_value = try_int(query) if db_column in ['edad', 'satisfaccion', 'recomendacion', 'target'] else query

    # ENG: Construct the base query
    # ESP: Construir la consulta base
    base_query = {}
    if query_value is not None:
        if column != 'Todo':
            base_query[db_column] = query_value
        else:
            regex_query = {"$regex": f"{query}", "$options": 'i'}
            or_query = [{"genero": regex_query}, {"cesfam": regex_query}, {"frecuencia": regex_query}, {"razon": regex_query}]
            if isinstance(query_value, int):
                or_query.extend([{"edad": query_value}, {"satisfaccion": query_value}, {"recomendacion": query_value}, {"target": query_value}])
            base_query["$or"] = or_query

    # ENG: Add date filtering to the query
    # ESP: Añadir filtrado por fecha a la consulta
    if start_date and end_date:
        start_date = pd.to_datetime(start_date).to_pydatetime()
        end_date = pd.to_datetime(end_date).to_pydatetime()
        end_date = end_date + timedelta(days=1)
        base_query['date'] = {'$gte': start_date, '$lt': end_date}

    # ENG: Search in the database using the constructed query
    # ESP: Buscar en la base de datos usando la consulta construida
    results = collection.find(base_query)
    return list(results)

# ENG: Function to get all documents from MongoDB
# ESP: Función para obtener todos los documentos de MongoDB
@st.cache_data
def get_all_documents():
    return list(collection.find({}))

# ENG: Function to get counts of different document categories
# ESP: Función para obtener recuentos de diferentes categorías de documentos
def get_document_counts():
    total_documents = collection.count_documents({})
    uncategorized_documents = collection.count_documents({'target': 3})
    return {
        "total": total_documents,
        "uncategorized": uncategorized_documents
    }
