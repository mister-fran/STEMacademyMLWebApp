"""
Configuration settings for the Streamlit ML Academy app
"""
import os

APP_CONFIG = {
    "title": "Machine Learning - STEM Academy",
    "description": "Lær machine learning gennem praktiske eksempler",
    "version": "1.0.0"
}


# # Data paths
# DATA_PATHS = {
#     "huspriser": "data\HousingPrices_selected.csv",
#     "diabetes": "data\diabetes_data_rounded.csv",
#     "gletsjer": "data\gletsjer_data_rounded.csv"
# }

DATA_PATHS = {
    "huspriser": os.path.join("data", "HousingPrices_selected.csv"),
    "diabetes": os.path.join("data", "diabetes_data_rounded.csv"),
    "gletsjer": os.path.join("data", "gletsjer_data_rounded.csv"),
    'VejledningHUSPRISER': os.path.join("data", "VejledningHUSPRISER.pdf"),
    'VejledningDIABETES': os.path.join("data", "VejledningDIABETES.pdf"),
    'VejledningGLETSJER': os.path.join("data", "VejledningGLETSJER.pdf"),
    'VejledningPARTIKEL': os.path.join("data", "VejledningPARTIKEL.pdf")
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