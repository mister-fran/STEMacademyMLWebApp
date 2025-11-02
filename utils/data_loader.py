"""
Data loading utilities for the ML Academy app
"""
import pandas as pd
import streamlit as st
from .config import DATA_PATHS
@st.cache_data
def load_huspriser_dataset():
    """Load housing prices dataset"""
    #Import using DATA_PATHS from config.py
    return pd.read_csv(DATA_PATHS['huspriser'])

@st.cache_data
def load_diabetes_dataset():
    """Load the diabetes dataset"""
    return pd.read_csv(DATA_PATHS['diabetes'])

@st.cache_data
def load_gletsjer_dataset():
    """Load gletsjer dataset"""
    return pd.read_csv(DATA_PATHS['gletsjer'])

@st.cache_data
def load_partikel_dataset():
    """Load partikel dataset"""
    return pd.read_csv(DATA_PATHS['partikel'])

