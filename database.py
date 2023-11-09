import os
import streamlit as st
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import timedelta
import datetime
import logging
import pandas as pd

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
        'Fecha': 'date',
        'Etiqueta': 'target'
    }
    
    def try_int(val):
        try:
            return int(val)
        except ValueError:
            return None
        
    db_column = column_mapping.get(column, column)
    query_value = try_int(query) if db_column in ['edad', 'satisfaccion', 'recomendacion', 'target'] else query

    # Construct the base query
    base_query = {}
    if query_value is not None:
        if column != 'Todo':
            # If the column is specific and the query is not None, use it in the base query
            base_query[db_column] = query_value
        else:
            # If we're searching across all columns
            regex_query = {"$regex": f"{query}", "$options": 'i'}
            or_query = [{"genero": regex_query}, {"cesfam": regex_query}, {"frecuencia": regex_query}, {"razon": regex_query}]
            if isinstance(query_value, int):
                # If the query can be an integer, add integer fields to the OR query
                or_query.extend([{"edad": query_value}, {"satisfaccion": query_value}, {"recomendacion": query_value}, {"target": query_value}])
            base_query["$or"] = or_query
    
    # Add date filtering to the base query if date range is provided
    if start_date and end_date:
        # Convert dates to datetime objects
        start_date = pd.to_datetime(start_date).to_pydatetime()
        end_date = pd.to_datetime(end_date).to_pydatetime()
        
        # MongoDB expects the end date for the range to be exclusive, hence adding one day
        end_date = end_date + timedelta(days=1)

        base_query['date'] = {'$gte': start_date, '$lt': end_date}

    # Use the constructed base query to search in the database
    results = collection.find(base_query)
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
