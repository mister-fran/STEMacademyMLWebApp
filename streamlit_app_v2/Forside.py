import streamlit as st
import pandas as pd


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
    st.write('Vælg et niveau i venstre side for at for at begynde. Under hvert niveau findes tre datasæt som matcher det som står i vejledningen på PDF (Link i venstre side). Disse er vist inde på de pågældende sider så du har overblik over hvilke variable data indeholder. ' \
    'Under hvert datasæt du få lov til at køre de tilhørende modeller:')
    st.markdown(' - Huspriser: Regression og Classification  \n - Diabetes: Classification  \n - Gletsjer: Regression')
    st.write('Under "Avanceret" kan du kan også uploade dit eget datasæt og prøve at analysere det med de gennemgåede modeller. Husk du kan hente vejledningen ved at trykke på knappen i sidepanelet i venstre side. God arbejdslyst!')
    # Add a download link for guidance PDF in the sidebar
    pdf_path = 'data/vejledning.pdf'  # Put your PDF file here
    
    st.sidebar.write("") # Add vertical space above button

    # Download button for PDF
    try:
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
        
        st.sidebar.download_button(
            label="Hent vejledning",
            data=pdf_bytes,
            file_name="vejledning.pdf",
            mime="application/pdf"
        )
    except FileNotFoundError:
        st.sidebar.warning("Vejledning PDF ikke fundet.")

    st.markdown("---")
    #st.write("**Navigation:** Brug sidepanelet til venstre for at navigere mellem de forskellige sider.")

if __name__ == "__main__":
    main()