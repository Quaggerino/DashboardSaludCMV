import os
import streamlit as st
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import timedelta
import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


# Load environment variables from .env.development
load_dotenv('.env.development')

# MongoDB connection
uri = os.getenv('MONGODB_URI')
client = MongoClient(uri)
db = client['cmvalparaisoDas']
collection = db['opinionSaludValparaiso']


@st.cache_data
def search_documents(query, column='Todo', start_date=None, end_date=None):
    column_mapping = {
        'ID': '_id',
        'Edad': 'edad',
        'Género': 'genero',
        'Centro de Salud': 'cesfam',
        'Frecuencia': 'frecuencia',
        'Satisfacción': 'satisfaccion',
        'Recomendación': 'recomendacion',
        'Comentario': 'razon',
        'Fecha': 'date',    # Updated to 'date'
        'Etiqueta': 'target'
    }
    
    # Function to try converting query to integer
    def try_int(val):
        try:
            return int(val)
        except ValueError:
            return val
        
    db_column = column_mapping.get(column, column)
    query = try_int(query) if db_column in ['edad', 'satisfaccion', 'recomendacion', 'target'] else query
    
    # Basic filtering
    if column == 'Todo':
        regex_query = {"$regex": f"{query}", "$options": 'i'}
        or_query = [{"genero": regex_query}, {"cesfam": regex_query}, {"frecuencia": regex_query}, {"razon": regex_query}]
        results = collection.find({"$or": or_query})
    elif column in ['ID', 'Género', 'Centro de Salud', 'Frecuencia', 'Comentario']:
        regex_query = {"$regex": f"{query}", "$options": 'i'}
        results = collection.find({db_column: regex_query})
    else:
        results = collection.find({db_column: query})
    
     # Date filtering
    if column == 'Fecha' or (start_date and end_date):
        if start_date:
            if isinstance(start_date, str):  # Check if start_date is a string
                start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            else:  # If it's already a datetime.date object, convert it to datetime
                start_date = datetime.datetime.combine(start_date, datetime.time())
                
        if end_date:
            if isinstance(end_date, str):  # Check if end_date is a string
                end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
            else:  # If it's already a datetime.date object, convert it to datetime and add one day
                end_date = datetime.datetime.combine(end_date, datetime.time()) + timedelta(days=1)
        
        date_query = {'date': {}}
        if start_date:
            date_query['date']['$gte'] = start_date
        if end_date:
            date_query['date']['$lt'] = end_date
        
        results = collection.find(date_query)  # Using find instead of filter

    
    return list(results)

@st.cache_data
def get_all_documents():
    return list(collection.find({}))

def get_document_counts():
    total_documents = collection.count_documents({})
    uncategorized_documents = collection.count_documents({'target': 3})
    return {
        "total": total_documents,
        "uncategorized": uncategorized_documents
    }
