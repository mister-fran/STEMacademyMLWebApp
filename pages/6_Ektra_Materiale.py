#Setup additional page for adding links to additional material on ML for STEM Academy
import streamlit as st
from utils.config import DATA_PATHS
import os

#Reset session state
PAGE_ID = "Ekstra_Materiale"  
# If we arrived here from another page, reset this page's state
if st.session_state.get("_active_page") != PAGE_ID:
    st.write(st.session_state.get("_active_page"))
    st.session_state.clear()
    st.session_state["_active_page"] = PAGE_ID
    st.rerun()

def main():
    st.set_page_config(
        page_title="Ekstra Materiale",  # This appears in browser tab
        layout="wide"  # Optional: use full width
    )


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
    st.markdown('- Neuralt netværk: Video forklarer intuitivt hvordan konceptet virker med gennemregnet eksempel. God forklaring af activation functions og hvordan disse bruges til at lave forudsigelser. [Link](https://www.youtube.com/watch?v=CqOfi41LfDw)')
    st.markdown('- Video med Neurale netværk forklaret med en enkelt node. Der er et rigtigt godt eksempel fra 4:30 med at forudsige et skud fra en kanon (det skrå kast). [Link](https://www.youtube.com/watch?v=GkiITbgu0V0&t=270s)')   
    st.markdown('- Tensorflow Playground (Mega fed). Interaktiv hjemmeside hvor man kan lege med et neuralt netværk med forskellige datasæt, aktiveringsfunktioner, inputvariable mm. [Link](https://playground.tensorflow.org/) ' \
                'Opgaver til at komme i gang: Prøv at danne fungerende modeller for de forskellige datasæt i venstre side (forskellige fordelinger af orange og blå prikker). Prøv at give den forskellige inputparametre og se om du stadig kan forudsige svaret. Prøv forskellige loss functions, activation functions. Tilføj noise og prøv igen.')

    st.subheader("Github og Colab")
    st.markdown('- Find folderen "STEMAcademyML_Materiale" [Link](https://github.com/troelspetersen/STEMacademyML/tree/main). Her ligger alt undervisningsmaterialet også. Rå, fulde datasæt under "Releases" i højre sidepanel.')
    st.markdown('- Du kan åbne filer direkte fra Github på Google Colab. [Link](https://colab.research.google.com/)')
    
if __name__ == "__main__":
    main()