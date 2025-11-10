#Setup additional page for adding links to additional material on ML for STEM Academy
import streamlit as st
from utils.config import DATA_PATHS
import os

def main():
    st.set_page_config(
        page_title="Ekstra Materiale",  # This appears in browser tab
        page_icon=":books:",  # Icon in browser tab
        layout="wide"  # Optional: use full width
    )

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

    st.title("Ekstra materiale")

    # Welcome message
    st.write('Her finder du ekstra materiale og ressourcer til Machine Learning for STEM Academy. ')
    st.write('Hvis du vil have en visuel og måske mere intuitiv måde at forstå tingene på kan du her finde videomateriale der gennemgår/forklarer nogle af de gennemgåede koncepter.')
    
    st.subheader("Decision Trees")
    st.markdown('- Video om classification med decision trees og gini koefficienten. [Link](https://www.youtube.com/watch?v=_L39rN6gz7Y)')
    st.markdown('- Video som forklarer ROC kurver, AUC, false positive rate og true positive rate, og hvor vi vælger at sætte vores beslutningsgrænse påvirker resultatet. (Efter 6:24 er ikke relevant) [Link](https://www.youtube.com/watch?v=QBVzZBsif20)')
    st.markdown('- Video om regression med decision trees, og hvordan squared error eller residual bliver beregnet. [Link](https://www.youtube.com/watch?v=g9c66TUylZ4)')

    st.subheader("Neurale Netværk") 
    st.markdown('- Video med Neurale netværk forklaret med en enkelt node. Der er et rigtigt godt eksempel fra 4:30 med at forudsige et skud fra en kanon (det skrå kast). [Link](https://www.youtube.com/watch?v=GkiITbgu0V0&t=270s)')   
    st.markdown('- Video med neurale netværk, forklaret lidt mere I dybden med matematikken og hvordan activation functions virker og gør at vi kan lave bedre forudsigelser. [Link](https://www.youtube.com/watch?v=CqOfi41LfDw)')

    st.subheader("Guthub")
    st.markdown('- Find folderen "STEMAcademyML_Materiale".[Link](https://github.com/troelspetersen/STEMacademyML/tree/main) Her ligger alt materialet også. Denne kan du også åbne filer fra direkte på Google Colab. [Link](https://colab.research.google.com/)')
    
if __name__ == "__main__":
    main()