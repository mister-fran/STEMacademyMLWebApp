"""
Configuration settings for the Streamlit ML Academy app
"""
from os import path

APP_CONFIG = {
    "title": "Machine Learning - STEM Academy",
    "description": "Lær machine learning gennem praktiske eksempler",
    "version": "1.0.0"
}



#Validate Paths
def validate_paths(paths):
    base_dir = path.abspath(path.dirname(path.dirname(__file__)))
    for key, pathh in paths.items():
        full_path = path.abspath(path.join(base_dir, pathh))
        if not full_path.startswith(base_dir):
            raise ValueError(f"Invalid path: {pathh}")
    return paths

DATA_PATHS = {
    "huspriser": path.join("data", "HousingPrices_selected.csv"),
    "diabetes": path.join("data", "diabetes_data_rounded.csv"),
    "gletsjer": path.join("data", "gletsjer_data_rounded.csv"),
    "partikel": path.join("data", "partikel_data_50000_rounded.csv"),
    "VejledningHUSPRISER": path.join("data", "VejledningHUSPRISER.pdf"),
    "VejledningDIABETES": path.join("data", "VejledningDIABETES.pdf"),
    "VejledningGLETSJER": path.join("data", "VejledningGLETSJER.pdf"),
    "VejledningPARTIKEL": path.join("data", "VejledningPARTIKEL.pdf")
}

DATA_PATHS = validate_paths(DATA_PATHS)

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