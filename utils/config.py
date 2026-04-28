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
    "partikel": os.path.join("data", "partikel_data_50000_rounded.csv"),
    "pendul1": os.path.join("data", "pendulum_data_wide_2.csv"),
    "pendul_short": os.path.join("data", "pendulum_data_short.csv"), #These two need to be constructed
    "pendul_long": os.path.join("data", "pendulum_data_long.csv"),
    "VejledningHUSPRISER": os.path.join("data", "VejledningHUSPRISER.pdf"),
    "VejledningDIABETES": os.path.join("data", "VejledningDIABETES.pdf"),
    "VejledningGLETSJER": os.path.join("data", "VejledningGLETSJER.pdf"),
    "VejledningPARTIKEL": os.path.join("data", "VejledningPARTIKEL.pdf"),
    "VejledningPENDUL": os.path.join("data", "VejledningPENDUL.pdf"),
    "TemplatePENDUL": os.path.join("data", "Penduldata_template.xlsx"),
    "EksempelPENDUL": os.path.join("data","Eksempel_Datasæt.xlsx"),
    "ForsideBILLEDE": os.path.join("data", "png_overblik_stem_academy_Page_1.png")
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