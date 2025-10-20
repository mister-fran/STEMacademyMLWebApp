"""
Configuration settings for the Streamlit ML Academy app
"""

APP_CONFIG = {
    "title": "Machine Learning - STEM Academy",
    "description": "Lær machine learning gennem praktiske eksempler",
    "version": "1.0.0"
}

import os
# Get the directory containing the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Build path to your file
file_path = os.path.join(BASE_DIR, "data", "HousingPrices_selected.csv")



# Data paths
DATA_PATHS = {
    "huspriser": filepath,
    "diabetes": "data\diabetes_data_rounded.csv",
    "gletsjer": "data\gletsjer_data_rounded.csv"
}

# ML Model settings
MODEL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42
}

# UI Settings
UI_CONFIG = {
    "dataframe_height": 200,
    "sidebar_width": 300

}


