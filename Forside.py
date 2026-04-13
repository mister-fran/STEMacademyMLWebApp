import streamlit as st
import pandas as pd

from utils.config import DATA_PATHS
import os
#Hi

def main(): 
    # Configure the page
    st.set_page_config(
        page_title="STEM Academy - Machine Learning",  # This appears in browser tab
        layout="wide"  # Optional: use full width
    )
    st.title("Machine Learning - STEM Academy")

    # Welcome message
    st.write('Velkommen til Machine Learning - STEM Academy!')
    st.write('Denne hjemmeside tilbyder et self-contained forløb til undervisning i Machine Learning (ML) i gymnasiet med udgangspunkt i forskellige udgangsniveauer; Hjemmesiden tilbyder forløb til elever der aldrig har set ML før, besidder basal viden, krævende elever der vil udfordre sig selv.')
    st.write('Til klasser med tekniske linjefag kan dette eventuelt kombineres med undervisning i programmeringsproget python, enten lokalt eller på Google Colab, som er et, nemt at anvende, webinterface til at køre pythonkode. ')
    st.write('Forløbene er delt ind efter sværhedsgrader:')
    st.markdown(' - Intro - Modul Pendul: Forløb til dem der aldrig har arbejdet med ML før med udgangspunkt i det simple pendul. Udelukkende introduktion til regression med det neurale netværk. \n'\
        ' - Standard: Til dem der har basal viden indenfor ML. Forløbet er mere matematisk og går mere i dybden med de bagvedliggende koncepter. Fire datasæt tilbydes med afsæt i forskellige problemstillinger fra hverdagen/fysikken. Introduktion til både regression og classification med det neurale netværk og boosted decision trees. Vejledningen til "Huspriser" indeholder en detaljeret introduktion til decision trees og det anbefales at begynde med denne hvis aldrig du har set decision trees før.\n'\
            ' - Avanceret: Hvis du allerede har en grundlæggende forståelse for modellernes virken, kan du prøve kræfter med udvidet adgang til hyperparametre og mere frihed i modellerne. \n'\
                ' - Ekstrem - Upload Eget Datasæt: Find og tilpas et datasæt så det bliver egnet til at blive analyseret af modellen. Modellen har samme funktionalitet som under Avanceret.')
    st.write('Overblik til forløb:') #GØR denne større 
    

    st.image(DATA_PATHS["ForsideBILLEDE"], caption="", use_container_width=True)

    # Add a download link for guidance PDF in the sidebar
    
    st.sidebar.write("") # Add vertical space above button

    st.markdown("---")
    #st.write("**Navigation:** Brug sidepanelet til venstre for at navigere mellem de forskellige sider.")

if __name__ == "__main__":
    main()