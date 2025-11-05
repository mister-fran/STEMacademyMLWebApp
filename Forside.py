import streamlit as st
import pandas as pd

from utils.config import DATA_PATHS
import os
#Hi

def main():
    # Configure the page
    st.set_page_config(
        page_title="STEM Academy - Machine Learning",  # This appears in browser tab
        page_icon=":mortar_board:",  # Icon in browser tab
        layout="wide"  # Optional: use full width
    )
    
    st.title("Machine Learning - STEM Academy")

    # Welcome message
    st.write('Velkommen til Machine Learning - STEM Academy! Denne hjemmeside er et værktøj til forløbet om Machine Learning for STEM Academy. Hjemmesiden er i stand til at køre alle de gennemgåede ML-modeller online.')
    st.write('Vælg Standard i venstre side for at begynde. Du kan køre alt hvad der ligger i .ipynb filerne online. ')
    st.markdown(' - Huspriser: Regression  \n - Diabetes: Classification  \n - Gletsjer: Regression  \n - Partikel: Classification')
    st.write('Under Avanceret har du adgang til flere hyperparametre samt valg af inputvariable. Her kan du også uploade dine egne datasæt og prøve modellerne af på dem. Husk du kan hente vejledningen ved at trykke på knappen i sidepanelet i venstre side. Under fanen Ekstra Materiale kan du finde videoer der forklarer nogle af de gennemgåede koncepter på en lidt anden og måske mere visuel måde. God arbejdslyst!')
    # Add a download link for guidance PDF in the sidebar
    
    st.sidebar.write("") # Add vertical space above button

    #Add Download Buttons for PDFS

    # Download button for PDF HUSPRISER
    if os.path.exists(DATA_PATHS['VejledningHUSPRISER']):
        try:
            with open(DATA_PATHS['VejledningHUSPRISER'], "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
            
            st.sidebar.download_button(
                label="📥 Hent vejledning til Huspriser",
                data=pdf_bytes,
                file_name="vejledningHUSPRISER.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.sidebar.error(f"Fejl ved indlæsning af PDF: {e}")
    else:
        st.sidebar.warning("⚠️ Vejledning PDF ikke fundet.")

    # Download button for PDF DIABETES
    if os.path.exists(DATA_PATHS['VejledningDIABETES']):
        try:
            with open(DATA_PATHS['VejledningDIABETES'], "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
            
            st.sidebar.download_button(
                label="📥 Hent vejledning til Diabetes",
                data=pdf_bytes,
                file_name="vejledningDIABETES.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.sidebar.error(f"Fejl ved indlæsning af PDF: {e}")
    else:
        st.sidebar.warning("⚠️ Vejledning PDF ikke fundet.")
    
    # Download button for PDF GLETSJER
    if os.path.exists(DATA_PATHS['VejledningGLETSJER']):
        try:
            with open(DATA_PATHS['VejledningGLETSJER'], "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
            
            st.sidebar.download_button(
                label="📥 Hent vejledning til Gletsjer",
                data=pdf_bytes,
                file_name="vejledningGLETSJER.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.sidebar.error(f"Fejl ved indlæsning af PDF: {e}")
    else:
        st.sidebar.warning("⚠️ Vejledning PDF ikke fundet.")

    # Download button for PDF PARTIKEL
    if os.path.exists(DATA_PATHS['VejledningPARTIKEL']):
        try:
            with open(DATA_PATHS['VejledningPARTIKEL'], "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
            
            st.sidebar.download_button(
                label="📥 Hent vejledning til Partikel",
                data=pdf_bytes,
                file_name="vejledningPARTIKEL.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.sidebar.error(f"Fejl ved indlæsning af PDF: {e}")
    else:
        st.sidebar.warning("⚠️ Vejledning PDF ikke fundet.")

    st.markdown("---")
    #st.write("**Navigation:** Brug sidepanelet til venstre for at navigere mellem de forskellige sider.")

if __name__ == "__main__":
    main()