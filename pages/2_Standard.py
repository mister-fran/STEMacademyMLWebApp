import streamlit as st
import pandas as pd
#Load data from dataloader
from utils.data_loader import load_huspriser_dataset, load_diabetes_dataset, load_gletsjer_dataset, load_partikel_dataset
import os
from utils.config import DATA_PATHS
from utils.plots import plotting, plotting_glet, plotting_partikel, Plotting_class, plotting_reg_own, plotting_class_own

#Importer pakker
# Data
import numpy as np
import scipy as scipy

# Plotting
import matplotlib.pyplot as plt

# Sklearn: et librabry med en masse funtioner vi bruger i Machine Learning
import sklearn as sklearn

# LightGBM - pakke til at køre decision tree
import lightgbm as lgb
from lightgbm import early_stopping
st.set_page_config(page_title="Standard Niveau", page_icon="🎯")

#Session state
#Reset session state
PAGE_ID = "Standard"  # change per page
# If we arrived here from another page, reset this page's state
if st.session_state.get("_active_page") != PAGE_ID:
    st.write(st.session_state.get("_active_page"))
    st.session_state.clear()
    st.session_state["_active_page"] = PAGE_ID
    st.rerun()

def main():
    st.title("🎯 Standard Niveau")

    # Load datasets using cached functions
    DS1 = load_huspriser_dataset()
    DS2 = load_diabetes_dataset()
    DS3 = load_gletsjer_dataset()
    DS4 = load_partikel_dataset()

    # Dataset selection
    st.sidebar.header("Datasæt")
    dataset = st.sidebar.radio("Vælg et datasæt:", ["Huspriser", "Diabetes", "Gletsjer", "Partikel"])

    # Add description
    st.write('Alternativ til at køre .ipynb filen lokalt på din computer. Indeholder samme funktionaliteter som .ipynb filerne med uden at man skal skrive/se kode selv.')
    st.write("Vælg et datasæt for at begynde.")    

    # Add a download link for guidance PDF in the sidebar
    pdf_path = 'data/vejledning.pdf'  # Put your PDF file here
    
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
    
    # Content based on dataset - Standard level
    if dataset == "Huspriser":
        #HER BEGYNDER VORES .ipynb
        st.subheader("Standard Niveau - Huspriser")
        st.write("Som nævnt, har du fået betroet opgaven at skrive en machine learning algoritme der kan forudsiger priser på huse. På denne hjemmeside behøver vi ikke importere nogen pakker da det er tilrettelagt således at man skal kunne lege med ML-modellerne uden at skulle bekymre sig om koden bag dem.")

        st.subheader("Inspicer dataen")
        st.write("Først vil vi gerne undersøge hvilken data vi har med at gøre. Vær opmærksom på at salgsprisen er i hele millioner.")
        st.dataframe(DS1, height=200, use_container_width=True)
        
        #Tilrettelæg data
        variabler = DS1.columns
        input_variabler = [v for v in variabler if v != 'Salgspris']
        input_data = DS1[input_variabler].to_numpy()
        truth_data = DS1['Salgspris'].to_numpy()

        st.subheader("Decision Tree")
        st.write("Et decision tree er bygget op af lag og grene. Ved hver gren stiller den et spørgsmål, og bevæger sig ned i det næste lag baseret på om spørgsmålet er sandt eller falsk. Og ved at lære af en masse data, kan den finde ud af hvilke spørgsmål der er bedst at stille.")

        st.subheader("Parameter")
        st.write("For et decision tree kan vi justere på hvor mange lag der skal være i vores træ, altså hvor mange lag af spørgsmål der må stilles. Vi kan justere på den parameter herunder.")

        #Make a slider to choose depth
        DT_N_lag = st.slider("Antal lag i træet", min_value=1, max_value=10, value=2, step=1)

        st.write("Her bygger og træner vi modellen og bruger Graphviz til at visualisere det.")

        # Her bliver modellen trænet på data
        estimator = sklearn.tree.DecisionTreeRegressor(max_depth=DT_N_lag, min_samples_leaf = 20,random_state=42)

        estimator.fit(input_data, truth_data)   # Dette er den "magiske" linje - her optimerer Machine Learning algoritmen sine interne vægte til at give bedste svar

        # laver visuel graf af træet
        dot = sklearn.tree.export_graphviz(estimator, out_file=None, feature_names=input_variabler, filled=True, max_depth=50, precision=2)         
        dot = dot.replace("squared_error", "error").replace("mse", "error")
        st.graphviz_chart(dot)
        st.write("Max dybde af træet:", estimator.get_depth())
        a = np.unique(estimator.predict(input_data)).size
        st.write("Forskellige priser den kan forudsige:",a )

        st.subheader("Spørgsmål")
        st.markdown("""- Inspicer træet. Forstår du/I, hvad de forskellige tal betyder?
Hvilken type bolig passer flest eksempler ned i, i lag 2? Hvad er algoritmens bud på deres pris (dvs. gennemsnitsprisen)?
- Prøv at ændre på hvor mange lag der er i træet fra 2 til 3.
Hvilken parameter bliver brugt oftest til at opdele data? Tror du/I at den så er den vigtigste parameter?
Kan du/I ud fra træet sige mere generelt hvilke parametre der betyder mest for prisen? Hvilke betyder mindst?""")
        
        st.subheader("Boosted Decision Tree")
        st.write("Nu hvor vi har set hvordan træet virker, vil vi gerne prøve at forudsige værdien på huse som vi ikke kender salgsprisen på. Som vi har set, kan det være svært at minimere vores 'loss function'. En måde at forbedre på er ved at køre boosted decision trees, hvilket vil sige at vi kører flere træer, hvor den hver gang lærer af fejlene fra det forrige træ, og på den måde bliver 'boostet' for hvert træ den laver. Herunder kan vi ændre hvor mange gange den må 'booste', altså hvor mange træer den må lave og lærer af.")
        
        boosting_rounds = st.slider("Antal boosting rounds", min_value=1, max_value=100, value=10, step=1)
        st.write("Vi kan også vælge hvor stor en andel af data vi vil bruge. ")
        andel_af_data = st.slider("Andel af data til træning", min_value=0.001, max_value=1.0, value=1.0, step=0.001)
        
        #Vi omdefinerer vores input og truth data til kun at indeholde en del af dataene.
        input_data_justeret, truth_data_justeret = sklearn.utils.resample(
            input_data, truth_data, 
            n_samples=int(andel_af_data * len(input_data)), 
            random_state=42, 
            replace=False
            )
        st.write("""Vi splitter datasættet i et træningssæt og et testsæt.
Træningssættet bruges til at træne modellen, hvor modellen får salgspriserne at vide.
Testsættet bruges til at give den trænede model data uden salgspriser, som den så skal forudsige, men hvor vi stadig kender svaret.""")
        data_træning, data_test, sande_pris_træning, sande_pris_test = \
        sklearn.model_selection.train_test_split(input_data_justeret, truth_data_justeret, test_size=0.25, random_state=42)
    
        # Her bygger vi modellen op med flere træer, træner på data og forudsiger priser
        #Implement button to run below model
        if st.button("Kør model"):
            gbm_test = lgb.LGBMRegressor(objective='regression', n_estimators=boosting_rounds, verbosity=-1)

            
            gbm_test.fit(data_træning, sande_pris_træning, eval_set=[(data_test, sande_pris_test)], 
                        eval_metric='mse', callbacks=[early_stopping(15)])
            st.session_state['gbm_model'] = gbm_test

            forudsagte_pris = gbm_test.predict(data_test, num_iteration=gbm_test.best_iteration_)
            plotting(sande_pris_test, forudsagte_pris)

            res = sklearn.inspection.permutation_importance(gbm_test, data_test, sande_pris_test, scoring="neg_mean_squared_error")
        
            st.write("Nu vil vi gerne inspicere hvor god vores model er til at forudsige på data hvor den ikke kender prisen. Det venstre plot viser residualerne, altså sande værdi - forudsagte værdi. Det højre plot er sande værdi vs forudsagte værdi. Her er også konturer (de sorte linjer), der viser tætheden af punkterne.")
            st.subheader("Spørgsmål")
            st.markdown("""
                        - Prøv at ændre på hvor mange gange gange den må booste, ved at ændre boosting_rounds fra 1 til 10, 100 eller 1000. Kan du se en forbedring?
                        - Hvor har modellen sværest ved at forudsige prisen? Er det ved de billigste huse, de dyreste, eller dem i mellem? Hvad kan det være? Hvilke huse tror du der er mest data på?
                        - Leg rundt med andelen af data du bruger. Hvordan ændres resultatet alt efter hvor meget data den har. Hvor meget data skal du bruge for at have en rimelig model og forudsigelse?""")
            st.subheader("Hvilke variabler er vigtigst?")
            st.write("Vi kan tjekke om vores intuition for hvilke variabler der er vigtigst med 'permutation importance'. Det er et mål for hvis værdierne i en kolonne bliver byttet rundt randomly, hvor meget påvirker det så resultatet. Hvis det er en vigtig variable, vil det påvirke resultatet meget. Her bliver det mål på hvor meget større mean squared error bliver, når den variabel bliver 'scramblet'.")


            imp_mse = res.importances_mean                
            order = np.argsort(imp_mse)[::-1]
            labels = np.asarray(variabler[:-1])[order]
            vals = imp_mse[order]

            fig, ax = plt.subplots(figsize=(8, 6))
            y = np.arange(len(vals))
            ax.barh(y, vals)
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.set_xlabel("Increase in MSE (permutation)")
            ax.set_ylabel("Feature")
            ax.set_title("Permutation Importance")
            ax.invert_yaxis()
            fig.tight_layout()
            st.pyplot(fig)

            st.write("Er resultatet som du forventede?")


        st.subheader("Sæt egne værdier ind og se modellens forudsigelse")
        #Make input fields for each variable except 'Salgspris'
        input_values = {}
        for var in variabler[:-1]:
            input_values[var] = st.number_input(f"Indtast værdi for {var}", value=0)
        #Make a button to predict price
        if st.button("Forudsig pris på hus"):
            input_array = np.array([input_values[var] for var in input_variabler]).reshape(1, -1)
            predicted_price = st.session_state['gbm_model'].predict(input_array, num_iteration=st.session_state['gbm_model'].best_iteration_)
            st.write(f"Den forudsagte salgspris er: {predicted_price[0]:,.2f} mio.kr.")

        #NN 
        st.subheader("Neurale Netværk")
        st.write("Neurale Netværk (NN) kommer fra at opbygningen af det, minder om den måde vores neuroner i hjernen snakker sammen på. På samme måde som et decision tree er der forskellige lag og vi kan styre hvor mange lag der er, men nu er det ikke kun sandt eller falsk, i stedet fungerer noderne som knapper der kan fintunes. ")
        st.write("Neurale netværk er mere følsomme overfor det data vi giver dem. Den fungerer bedst hvis værdierne af data er mellem 0 og 1. Derfor bruger vi en funktion til at skalere vores data, kaldet StandardScaler.")
        scaler = sklearn.preprocessing.StandardScaler()
        data_træning = scaler.fit_transform(data_træning)
        data_test = scaler.transform(data_test)
        
    
        st.write("I et neuralt netværk kan vi justere på hvor mange lag og hvor mange noder hvert lag skal have:")

        #Make six slider, one for each layer. that is six layers in total. sliders decide amount of nodes per layer
        layer_one = st.slider("Antal noder i lag 1", min_value=1, max_value=32, value=32, step=1)
        layer_two = st.slider("Antal noder i lag 2", min_value=1, max_value=32, value=16, step=1)
        layer_three = st.slider("Antal noder i lag 3", min_value=1, max_value=32, value=8, step=1)
        layer_four = st.slider("Antal noder i lag 4", min_value=1, max_value=32, value=4, step=1)
        layer_five = st.slider("Antal noder i lag 5", min_value=1, max_value=32, value=2, step=1)
        layer_six = st.slider("Antal noder i lag 6", min_value=1, max_value=32, value=2, step=1)


        st.write("""Nedenfor træner vi modellen. Vi kan også regne ud hvor mange parametre modellen bruger.
Herefter plotter vi for at se hvor godt modellen klarer sig. Denne kan tage op til ~et minut at køre.""")
        if st.button("Kør Neuralt Netværk"):
            # Her definerer og træner vi modellen
            mlp = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(layer_one, layer_two, layer_three, layer_four, layer_five, layer_six), 
            max_iter=2000, early_stopping=True, random_state=42)
            mlp.fit(data_træning, sande_pris_træning) 
            st.session_state['mlp'] = mlp #Gem modellen til senere
            # Her giver vi den trænede model test data som den ikke har set før, og beder om at forudsige prisen
            forudsagte_pris = mlp.predict(data_test)  
            # Beregn antal parametre i modellen
            # Coef er vægtene er intercept er bias. Den henter antallet directe fra modellen.
            n_params = sum(coef.size + intercept.size for coef, intercept in zip(mlp.coefs_, mlp.intercepts_))
            st.write(f"Antal parametre i NN: {n_params}")
            plotting(sande_pris_test, forudsagte_pris)
            st.subheader("Spørgsmål:")
            st.markdown("""- Prøv at justere på antal neuroner i det neurale netværk - Kan du mindske usikkerheden?
- Får du det samme antal parametre når du regner efter?
- Hvilken algoritme klarer sig bedst? Boosted decision tree eller neutralt netværk?
- Leg rundt med andelen af data du bruger. Hvordan ændres resultatet alt efter hvor meget data den har. Hvor meget data skal du bruge for at have en rimelig model og forudsigelse?""")

        st.subheader("Sæt egne værdier ind og se modellens forudsigelse")
        #Make input fields for each variable except 'Salgspris'
        input_values = {}
        for var in variabler[:-1]:
            input_values[var] = st.number_input(f"Indtast værdi for {var} ", value=0)
        #Make a button to predict price
        if st.button("Forudsig pris på hus "):
            input_array = np.array([input_values[var] for var in input_variabler]).reshape(1, -1)
            input_array = scaler.transform(input_array)
            predicted_price = st.session_state['mlp'].predict(input_array)
            st.write(f"Den forudsagte salgspris er: {predicted_price[0]:,.2f} mio.kr.")

    #DIABETES .ipynb
    elif dataset == "Diabetes":
        #HER BEGYNDER VORES .ipynb
        st.subheader("Standard Niveau - Diabetes")
        st.write("Som nævnt, har du fået betroet opgaven at skrive en machine learning algoritme der kan forudsiger om en person har diabetes eller ej. På denne hjemmeside behøver vi ikke importere nogen pakker da det er tilrettelagt således at man skal kunne lege med ML-modellerne uden at skulle bekymre sig om koden bag dem.")

        #Inspicer dataen
        st.subheader("Inspicer dataen")
        st.write("Først vil vi gerne undersøge hvilken data vi har med at gøre.")
        st.dataframe(DS2, height=200, use_container_width=True)
        
        #Tilrettelæg data
        variabler = DS2.columns
        input_variabler = [v for v in variabler if v != 'Diabetes']
        input_data = DS2[input_variabler].to_numpy()
        truth_data = DS2['Diabetes'].to_numpy()

        st.subheader("Decision Tree")
        st.write("Et decision tree er bygget op af lag og grene. Ved hver gren stiller den et spørgsmål, og bevæger sig ned i det næste lag baseret på om spørgsmålet er sandt eller falsk. Og ved at lære af en masse data, kan den finde ud af hvilke spørgsmål der er bedst at stille.")

        st.subheader("Parameter")
        st.write("For et decision tree kan vi justere på hvor mange lag der skal være i vores træ, altså hvor mange lag af spørgsmål der må stilles. Vi kan justere på den parameter herunder.")

        #Make a slider to choose depth
        DT_N_lag = st.slider("Antal lag i træet", min_value=1, max_value=10, value=2, step=1)

        st.write("Her bygger og træner vi modellen og bruger Graphviz til at visualisere det.")

        # Her bliver modellen trænet på data
        estimator = sklearn.tree.DecisionTreeClassifier(max_depth=DT_N_lag, min_samples_leaf = 20,random_state=42)
        estimator.fit(input_data, truth_data)   # Dette er den "magiske" linje - her optimerer Machine Learning algoritmen sine interne vægte til at give bedste svar
        
        # laver visuel graf af træet
        dot = sklearn.tree.export_graphviz(estimator, out_file=None, feature_names=input_variabler, filled=True, max_depth=50, precision=2)         
        dot = dot.replace("squared_error", "error").replace("mse", "error")
        st.graphviz_chart(dot)
        st.write("Max dybde af træet:", estimator.get_depth())


        st.subheader("Spørgsmål")
        st.markdown("""
- Inspicer træet. Forstår du/I, hvad de forskellige tal betyder?
  Hvilken kasse falder de fleste personer ned I? hvor er der mindst? 
  
Tallene i value er [raske, diabetikere].
- Hvilke(n) af kasserne bliver kategoriseret som at have diabetes? 
- Hvor stor en del af patienter med diabetes vil den forudsige til at have diabetes?
- Prøv at ændre på hvor mange lag der er i træet fra 2 til 3.
  Hvilken parameter bliver brugt oftest til at opdele data? Kan du/I ud fra træet sige mere generelt hvilke parametre der betyder mest for om en person har diabetes? Hvilke betyder mindst?
""")
        
        st.subheader("Boosted Decision Tree")
        st.write("Nu hvor vi har set hvordan træet virker, vil vi gerne prøve at forudsige om patienter vi ikke kender diagnosen på har diabetes eller ej. Som vi har set, kan det være svært at minimere vores 'loss function'. En måde at forbedre på er ved at køre boosted decision trees, hvilket vil sige at vi kører flere træer, hvor den hver gang lærer af fejlene fra det forrige træ, og på den måde bliver 'boostet' for hvert træ den laver. Herunder kan vi ændre hvor mange gange den må 'booste', altså hvor mange træer den må lave og lærer af.")
        
        boosting_rounds = st.slider("Antal boosting rounds", min_value=1, max_value=1000, value=1, step=1)
        st.write("Vi kan også vælge hvor stor en andel af data vi vil bruge. ")
        andel_af_data = st.slider("Andel af data til træning", min_value=0.001, max_value=1.0, value=1.0, step=0.001)
        
        #Vi omdefinerer vores input og truth data til kun at indeholde en del af dataene.
        input_data_justeret, truth_data_justeret = sklearn.utils.resample(
            input_data, truth_data, 
            n_samples=int(andel_af_data * len(input_data)), 
            random_state=42, 
            replace=False
            )
        st.write("""Vi splitter data i et træningssæt og et testsæt.
Træningssættet bruges til at træne modellen, hvor modellen får at vide hvilke personer der har diabetes.
Testsættet bruges til at give den trænede model data uden at vide hvilke personer der har diabetes, og bagefter kan vi tjekke om dens forudsigelser var korrekte.""")
        data_træning, data_test, sande_gruppe_træning, sande_gruppe_test = \
    sklearn.model_selection.train_test_split(input_data_justeret, truth_data_justeret, test_size=0.25, random_state=42)

        # Her bygger vi modellen op med flere træer, træner på data og forudsiger priser
        #Implement button to run below model
        st.write("Her definerer vi beslutningsgrænsen. Som standard bruges 0.5.")
        beslutningsgrænse = st.slider("Beslutningsgrænse", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        if st.button("Kør model"):  
            gbm_test = lgb.LGBMClassifier(objective='binary', n_estimators=boosting_rounds, verbosity=-1) 

            # Her træner vi vores model på vores trænings data
            gbm_test.fit(data_træning, sande_gruppe_træning, eval_set=[(data_test, sande_gruppe_test)], 
                        callbacks=[early_stopping(15)])
            st.session_state['gbm_test'] = gbm_test
            # Her får vi sandsynlighederne for om hver person har diabetes eller ej
            forudsagte_score = gbm_test.predict_proba(data_test, num_iteration=gbm_test.best_iteration_)[:, 1]

            # her får vi en liste med 0'er og 1'ere. Hvis en person har en sandsynlighed over beslutningsgrænsen, bliver den sat til 1,
            # altså forudsagt som diabetiker
            forudsagte_gruppe = gbm_test.predict_proba(data_test, num_iteration=gbm_test.best_iteration_)[:,1]
            forudsagte_gruppe = (forudsagte_gruppe > beslutningsgrænse).astype(int)

            Plotting_class(sande_gruppe_test, forudsagte_score, forudsagte_gruppe, beslutningsgrænse)


            st.subheader("Evaluer resultat med AUC og histogram")
            st.write("Nu vil vi gerne inspicere hvor god vores model er til at forudsige om en person har diabetes eller ej. Det venstre plot viser en ROC-kurve dvs. hvor stor en andel af sande gæt har vi per andel af forkerte gæt. Jo tættere denne er på venstre øverste hjørne jo bedre. Selve scoren Area Under Curve (AUC) angiver bare hvor tæt på hjørnet grafen er. 1 angiver en perfekt score.")
            st.write("Det højre plot viser fordelingen af korrekte og forkerte gæt farvekodet efter hvad data rent faktisk svarede til. Dvs vi kigger på hvad modellen har gættet på ud fra hvad vores data rent faktisk svarede til. Der vil altid være nogen der bliver forudsagt forkert, vores opgave er at minimere antallet.")
            st.subheader("Spørgsmål")
            st.markdown("""
- Prøv at ændre på hvor mange gange gange den må booste, ved at ændre boosting_rounds fra 1 til 10, 100 eller 1000. Kan du se en forbedring?
- Hvor laver modellen flest fejl? Raske der grupperes som diabetikere, eller omvendt?
- Som standard deler den ved 0.5 sandsynlighed. Kunne det være en fordel at dele ved en anden sandsynlighed? (Prøv evt. at ændre på beslutningsgrænsen oppe hvor modellen bliver trænet)
- Leg rundt med andelen af data du bruger. Hvordan ændres resultatet alt efter hvor meget data den har. Hvor meget data skal du bruge for at have en rimelig model og forudsigelse?
""")
            st.subheader("Hvilke variabler er vigtigst?")
            st.write("Vi kan tjekke om vores intuition for hvilke variabler der er vigtigst med 'permutation importance'. Det er et mål for hvis værdierne i en kolonne bliver byttet tilfældigt rundt, hvor meget påvirker det så resultatet. Hvis det er en vigtig variabel, vil det påvirke resultatet meget. Her bliver det målt på hvor meget større mean squared error (fejlen) bliver, når den variabel bliver 'scramblet'.")
            
            res = sklearn.inspection.permutation_importance(gbm_test, data_test, sande_gruppe_test, scoring="neg_mean_squared_error")

            imp_mse = res.importances_mean                
            order = np.argsort(imp_mse)[::-1]
            labels = np.asarray(variabler[:-1])[order]
            vals = imp_mse[order]

            fig, ax = plt.subplots(figsize=(8, 6))
            y = np.arange(len(vals))
            ax.barh(y, vals)
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.set_xlabel("Increase in log_loss (permutation)")
            ax.set_ylabel("Feature")
            ax.set_title("Permutation Importance")
            ax.invert_yaxis()
            fig.tight_layout()
            st.pyplot(fig)

            st.markdown("- Er resultatet som du forventede?")

        st.subheader("Sæt egne værdier ind og se modellens forudsigelse")
        #Make input fields for each variable except 'Salgspris'
        input_values = {}
        for var in variabler[:-1]:
            input_values[var] = st.number_input(f"Indtast værdi for {var} ", value=0)
        #Make a button to predict price
        if st.button("Forudsig sandsynlighed for diabetes "):
            input_array = np.array([input_values[var] for var in input_variabler]).reshape(1, -1)
            predicted_prob = st.session_state['gbm_test'].predict_proba(input_array, num_iteration=st.session_state['gbm_test'].best_iteration_)
            st.write(f"Den forudsagte sandsynlighed for diabetes er: {predicted_prob[0][1]:,.2f}")

        #NN 
        st.subheader("Neurale Netværk")
        st.write("Neurale Netværk (NN) kommer fra at opbygningen af det, minder om den måde vores neuroner i hjernen snakker sammen på. På samme måde som et decision tree er der forskellige lag og vi kan styre hvor mange lag der er, men nu er det ikke kun sandt eller falsk, i stedet fungerer noderne som knapper der kan fintunes. ")
        st.write("Neurale netværk er mere følsomme overfor det data vi giver dem. Den fungerer bedst hvis værdierne af data er mellem 0 og 1. Derfor bruger vi en funktion til at skalere eller normalisere vores data, kaldet StandardScaler.")
        scaler = sklearn.preprocessing.StandardScaler()
        data_træning_transformed = scaler.fit_transform(data_træning)
        data_test_transformed = scaler.transform(data_test)

        st.write("I et neuralt netværk kan vi justere på hvor mange lag og hvor mange noder hvert lag skal have:")

        #Make six slider, one for each layer. that is six layers in total. sliders decide amount of nodes per layer
        layer_one = st.slider("Antal noder i lag 1", min_value=1, max_value=128, value=64, step=1)
        layer_two = st.slider("Antal noder i lag 2", min_value=1, max_value=128, value=32, step=1)
        layer_three = st.slider("Antal noder i lag 3", min_value=1, max_value=128, value=16, step=1)
        layer_four = st.slider("Antal noder i lag 4", min_value=1, max_value=128, value=8, step=1)
        layer_five = st.slider("Antal noder i lag 5", min_value=1, max_value=128, value=4, step=1)
        layer_six = st.slider("Antal noder i lag 6", min_value=1, max_value=128, value=2, step=1)


        st.write("""Nedenfor træner vi modellen. Vi kan også regne ud hvor mange parametre modellen bruger.
Herefter plotter vi for at se hvor godt modellen klarer sig.
                 Det kan godt tage op til ~et minut at køre denne model.""")
        if st.button("Kør Neuralt Netværk"):
            # Her definerer og træner vi modellen
            mlp = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(layer_one, layer_two, layer_three, layer_four, layer_five, layer_six), 
            max_iter=2000, early_stopping=True, random_state=42)
            mlp.fit(data_træning_transformed, sande_gruppe_træning) 
            st.session_state['mlp'] = mlp #Gem modellen til senere

            # Her giver vi den trænede model test data som den ikke har set før, og beder om at forudsige prisen
            forudsagte_gruppe = mlp.predict_proba(data_test_transformed)[:,1]  
            forudsagte_gruppe = (forudsagte_gruppe > beslutningsgrænse).astype(int)
            forudsagte_score = mlp.predict_proba(data_test_transformed)[:, 1]

            # Beregn antal parametre i modellen
            # Coef er vægtene er intercept er bias. Den henter antallet direkte fra modellen.
            n_params = sum(coef.size + intercept.size for coef, intercept in zip(mlp.coefs_, mlp.intercepts_))
            st.write(f"Antal parametre i NN: {n_params}")
            Plotting_class(sande_gruppe_test, forudsagte_score, forudsagte_gruppe, beslutningsgrænse)
            st.subheader("Spørgsmål:")
            st.markdown("""
- Prøv at justere på antal neuroner i det neurale netværk - Kan du forbedre AUC og antallet at syge klassificeret som raske?
- Får du det samme antal parametre når du regner efter?
- Hvilken algoritme klarer sig bedst? Boosted decision tree eller neutralt netværk? Kan du få NN til at klare sig lige så godt som BDT?
- Prøv at justere på dit BDT og NN så de rammer samme AUC. Hvilken algoritme er så hurtigst?
- Leg rundt med andelen af data du bruger. Hvordan ændres resultatet alt efter hvor meget data den har. Hvor meget data skal du bruge for at have en rimelig model og forudsigelse?
                        """)

        st.subheader("Sæt egne værdier ind og se modellens forudsigelse")
        #Make input fields for each variable except 'Salgspris'
        input_values = {}
        for var in variabler[:-1]:
            input_values[var] = st.number_input(f"Indtast værdi for {var}", value=0)
        #Make a button to predict price
        if st.button("Forudsig sandsynlighed for diabetes"):
            input_array = np.array([input_values[var] for var in input_variabler]).reshape(1, -1)
            input_array = scaler.transform(input_array)
            predicted_prob = st.session_state['mlp'].predict_proba(input_array)
            st.write(f"Den forudsagte sandsynlighed for diabetes er: {predicted_prob[0][1]:,.2f}")

            
    elif dataset == "Gletsjer":
        #HER BEGYNDER VORES .ipynb
        st.subheader("Standard Niveau - Gletsjer")
        st.write("Nedenfor skal du hjælpe gletsjervidenskabsfakultetet med at udvikle deres ML model til at bestemme dybden af gletsjere. På denne hjemmeside behøver vi ikke importere nogen pakker da det er tilrettelagt således at man skal kunne lege med ML-modellerne uden at skulle bekymre sig om koden bag dem.")

        #Inspicer dataen
        st.subheader("Inspicer dataen")
        st.write("Først vil vi gerne undersøge hvilken data vi har med at gøre.")
        st.dataframe(DS3, height=200, use_container_width=True)
        
        #Tilrettelæg data
        variabler = DS3.columns
        input_variabler = [v for v in variabler if v != 'gletsjer_dybde']
        input_data = DS3[input_variabler].to_numpy()
        truth_data = DS3['gletsjer_dybde'].to_numpy()

        st.subheader("Decision Tree")
        st.write("Et decision tree er bygget op af lag og grene. Ved hver gren stiller den et spørgsmål, og bevæger sig ned i det næste lag baseret på om spørgsmålet er sandt eller falsk. Og ved at lære af en masse data, kan den finde ud af hvilke spørgsmål der er bedst at stille.")

        st.subheader("Parameter")
        st.write("For et decision tree kan vi justere på hvor mange lag der skal være i vores træ, altså hvor mange lag af spørgsmål der må stilles. Vi kan justere på den parameter herunder.")

        #Make a slider to choose depth
        DT_N_lag = st.slider("Antal lag i træet", min_value=1, max_value=10, value=2, step=1)

        st.write("Her bygger og træner vi modellen og bruger Graphviz til at visualisere det.")

        # Her bliver modellen trænet på data
        estimator = sklearn.tree.DecisionTreeRegressor(max_depth=DT_N_lag, min_samples_leaf = 20,random_state=42)

        estimator.fit(input_data, truth_data)   # Dette er den "magiske" linje - her optimerer Machine Learning algoritmen sine interne vægte til at give bedste svar

        # laver visuel graf af træet
        dot = sklearn.tree.export_graphviz(estimator, out_file=None, feature_names=input_variabler, filled=True, max_depth=50, precision=2)         
        dot = dot.replace("squared_error", "error").replace("mse", "error")
        st.graphviz_chart(dot)
        st.write("Max dybde af træet:", estimator.get_depth())
        a = np.unique(estimator.predict(input_data)).size
        st.write("Forskellige dybder den kan forudsige:",a )

        st.subheader("Spørgsmål")
        st.markdown("""- Inspicer træet. Forstår du/I, hvad de forskellige tal betyder?
  Hvad er gældende for gletsjerne i lag 2 og hvad er algortimens bud på deres dybde?
- Prøv at ændre på hvor mange lag der er i træet fra 2 til 3.
  Hvilken parameter bliver brugt oftest til at opdele data? Tror du/I at den så er den vigtigste parameter?
  Kan du/I ud fra træet sige mere generelt hvilke parametre der betyder mest for dybden? Hvilke betyder mindst?""")
        
        st.subheader("Boosted Decision Tree")
        st.write("Nu hvor vi har set hvordan træet virker, vil vi gerne prøve at forudsige værdien på gletsjere som vi ikke kender dybden på. Som vi har set, kan det være svært at minimere vores 'loss function'. En måde at forbedre på er ved at køre boosted decision trees, hvilket vil sige at vi kører flere træer, hvor den hver gang lærer af fejlene fra det forrige træ, og på den måde bliver 'boostet' for hvert træ den laver. Herunder kan vi ændre hvor mange gange den må 'booste', altså hvor mange træer den må lave og lære af.")
        
        boosting_rounds = st.slider("Antal boosting rounds", min_value=1, max_value=1000, value=1, step=1)
        st.write("Vi kan også vælge hvor stor en andel af data vi vil bruge. ")
        andel_af_data = st.slider("Andel af data til træning", min_value=0.001, max_value=1.0, value=1.0, step=0.001)
        
        #Vi omdefinerer vores input og truth data til kun at indeholde en del af dataene.
        input_data_justeret, truth_data_justeret = sklearn.utils.resample(
            input_data, truth_data, 
            n_samples=int(andel_af_data * len(input_data)), 
            random_state=42, 
            replace=False
            )
        st.write("""Vi splitter data i et træningssæt og et testsæt.
Træningssættet bruges til at træne modellen, hvor modellen får de rigtige dybder at vide at vide.
Testsættet bruges til at give den trænede model data uden dybder, som den så skal forudsige, men hvor vi stadig kender svaret. Dette bruges til at evaluere modellens performance.""")
        data_træning, data_test, sand_dybde_træning, sand_dybde_test = \
    sklearn.model_selection.train_test_split(input_data_justeret, truth_data_justeret, test_size=0.25, random_state=42)
    
        # Her bygger vi modellen op med flere træer, træner på data og forudsiger priser
        #Implement button to run below model
        if st.button("Kør model"):
            gbm_test = lgb.LGBMRegressor( objective='regression', n_estimators=boosting_rounds, verbosity=-1)

            gbm_test.fit(data_træning, sand_dybde_træning, eval_set=[(data_test, sand_dybde_test)], 
                        eval_metric='mse', callbacks=[early_stopping(15)])

            forudsagt_dybde = gbm_test.predict(data_test, num_iteration=gbm_test.best_iteration_)
            plotting_glet(sand_dybde_test, forudsagt_dybde)

            res = sklearn.inspection.permutation_importance(gbm_test, data_test, sand_dybde_test, scoring="neg_mean_squared_error")

            st.write("Nu vil vi gerne inspicere hvor god vores model er til at forudsige på data hvor den ikke kender dybden i forvejen. Det venstre plot viser residualerne, altså (sand værdi - forudsagt værdi). Det højre plot er sand værdi vs forudsagt værdi. Her er også konturer (de sorte linjer), der viser tætheden af punkterne.")
            st.subheader("Spørgsmål")
            st.markdown("""
- Prøv at ændre på hvor mange gange gange den må booste, ved at ændre boosting_rounds fra 1 til 2 til 10, 100 eller 1000. Kan du se en forbedring?
- Hvilke gletsjere er der mest data på?
- Hvad gætter modellen på hvis ikke den før lov til at booste mange gange? Er der bestemte områder hvor modellen har sværere ved at forudsige dybden?
- Leg rundt med andelen af data du bruger. Hvordan ændres resultatet alt efter hvor meget data den har. Hvor meget data skal du bruge for at have en rimelig model og forudsigelse?""")
            st.subheader("Hvilke variable er vigtigst?")
            st.write("Vi kan tjekke om vores intuition for hvilke variable der er vigtigst med 'permutation importance'. Det er et mål for hvis værdierne i en kolonne bliver byttet rundt randomly, hvor meget påvirker det så resultatet. Hvis det er en vigtig variable, vil det påvirke resultatet meget. Her bliver det mål på hvor meget større mean squared error bliver, når den variabel bliver 'scramblet'.")


            imp_mse = res.importances_mean                
            order = np.argsort(imp_mse)[::-1]
            labels = np.asarray(variabler[:-1])[order]
            vals = imp_mse[order]

            fig, ax = plt.subplots(figsize=(8, 6))
            y = np.arange(len(vals))
            ax.barh(y, vals)
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.set_xlabel("Increase in MSE (permutation)")
            ax.set_ylabel("Feature")
            ax.set_title("Permutation Importance")
            ax.invert_yaxis()
            fig.tight_layout()
            st.pyplot(fig)

            st.markdown("""
- Er resultatet som du forventede? 
- Kan du give en mulig grund til hvorfor netop disse variable har størst betydning?
                        """)

        #NN 
        st.subheader("Neurale Netværk")
        st.write("Neurale Netværk (NN) kommer fra at opbygningen af det, minder om den måde vores neuroner i hjernen snakker sammen på. På samme måde som et decision tree er der forskellige lag og vi kan styre hvor mange lag der er, men nu er det ikke kun sandt eller falsk, i stedet fungerer noderne som knapper der kan fintunes.")
        st.write("Neurale netværk er mere følsomme overfor det data vi giver dem. Den fungerer bedst hvis værdierne af data er mellem 0 og 1. Derfor bruger vi en funktion til at skalere vores data, kaldet StandardScaler.")
        scaler = sklearn.preprocessing.StandardScaler()
        data_træning = scaler.fit_transform(data_træning)
        data_test = scaler.transform(data_test)
        
    
        st.write("I et neuralt netværk kan vi justere på hvor mange lag og hvor mange noder hvert lag skal have:")

        #Make six slider, one for each layer. that is six layers in total. sliders decide amount of nodes per layer
        layer_one = st.slider("Antal noder i lag 1", min_value=1, max_value=32, value=32, step=1)
        layer_two = st.slider("Antal noder i lag 2", min_value=1, max_value=32, value=16, step=1)
        layer_three = st.slider("Antal noder i lag 3", min_value=1, max_value=32, value=8, step=1)
        layer_four = st.slider("Antal noder i lag 4", min_value=1, max_value=32, value=4, step=1)
        layer_five = st.slider("Antal noder i lag 5", min_value=1, max_value=32, value=2, step=1)
        layer_six = st.slider("Antal noder i lag 6", min_value=1, max_value=32, value=2, step=1)


        st.write("""Nedenfor træner vi modellen. Vi kan også regne ud hvor mange parametre modellen bruger.
Herefter plotter vi for at se hvor godt modellen klarer sig.
                 Det kan godt tage op til ~et minut at køre denne model.""")
        if st.button("Kør Neuralt Netværk"):
            # Her definerer og træner vi modellen
            mlp = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(layer_one, layer_two, layer_three, layer_four, layer_five, layer_six), 
            max_iter=1000, early_stopping=True, random_state=42)
            mlp.fit(data_træning, sand_dybde_træning) 
            # Her giver vi den trænede model test data som den ikke har set før, og beder om at forudsige dybden
            forudsagt_dybde = mlp.predict(data_test)  

            # Beregn antal parametre i modellen
            # Coef er vægtene er intercept er bias. Den henter antallet directe fra modellen.
            n_params = sum(coef.size + intercept.size for coef, intercept in zip(mlp.coefs_, mlp.intercepts_))
            st.write(f"Antal parametre i NN: {n_params}")
            plotting_glet(sand_dybde_test, forudsagt_dybde)
            st.subheader("Spørgsmål:")
            st.markdown("""
- Prøv at justere på antal neuroner i det neurale netværk - Bliver modellen bedre dårligere/kører den hurtigere langsommere?
- Får du det samme antal parametre når du regner efter?
- Hvilken algoritme klarer sig bedst? Boosted decision tree eller neutralt netværk? Kan du få NN til at klare sig lige så godt som BDT? Eller omvendt?
- Leg rundt med andelen af data du bruger. Hvordan ændres resultatet alt efter hvor meget data den har. Hvor meget data skal du bruge for at have en rimelig model og forudsigelse?
                        """)


    #DIABETES .ipynb
    elif dataset == "Partikel":
        #HER BEGYNDER VORES .ipynb
        st.subheader("Standard Niveau - Partikel")
        st.write("Som nævnt, er du blevet udnævnt personligt til at klassificere elektroner på CERN. På denne hjemmeside behøver vi ikke importere nogen pakker da det er tilrettelagt således at man skal kunne lege med ML-modellerne uden at skulle bekymre sig om koden bag dem.")

        #Inspicer dataen
        st.subheader("Inspicer dataen")
        st.write("Først vil vi gerne undersøge hvilken data vi har med at gøre.")
        st.dataframe(DS4, height=200, use_container_width=True)
        
        #Tilrettelæg data
        variable = DS4.columns
        input_variable = [v for v in variable if v != 'p_Truth_isElectron']
        input_data = DS4[input_variable].to_numpy()
        truth_data = DS4['p_Truth_isElectron'].to_numpy()

        st.subheader("Decision Tree")
        st.write("Et decision tree er bygget op af lag og grene. Ved hver gren stiller den et spørgsmål, og bevæger sig ned i det næste lag baseret på om spørgsmålet er sandt eller falsk. Og ved at lære af en masse data, kan den finde ud af hvilke spørgsmål der er bedst at stille.")

        st.subheader("Parameter")
        st.write("For et decision tree kan vi justere på hvor mange lag der skal være i vores træ, altså hvor mange lag af spørgsmål der må stilles. Vi kan justere på den parameter herunder.")

        #Make a slider to choose depth
        DT_N_lag = st.slider("Antal lag i træet", min_value=1, max_value=10, value=2, step=1)

        st.write("Her bygger og træner vi modellen og bruger Graphviz til at visualisere det.")

        # Her bliver modellen trænet på data
        estimator = sklearn.tree.DecisionTreeClassifier(max_depth=DT_N_lag, min_samples_leaf = 20,random_state=42)
        estimator.fit(input_data, truth_data)   # Dette er den "magiske" linje - her optimerer Machine Learning algoritmen sine interne vægte til at give bedste svar

        # laver visuel graf af træet
        dot = sklearn.tree.export_graphviz(estimator, out_file=None, feature_names=input_variable, filled=True, max_depth=50, precision=2)         
        dot = dot.replace("squared_error", "error").replace("mse", "error")
        st.graphviz_chart(dot)
        st.write("Max dybde af træet:", estimator.get_depth())


        st.subheader("Spørgsmål")
        st.markdown("""
- Inspicer træet. Forstår du/I, hvad de forskellige tal betyder?
  Hvad sker der fra lag til lag og hvor mange samples er der i hver kasse?
- Prøv at ændre på hvor mange lag der er i træet fra 2 til 3.
  Hvilke parametre bruges til at opdele data? 
- Hvordan ændres værdien af gini ift. om der kun er elektroner/ikke-elektroner eller begge typer?
                    """)
        
        st.subheader("Boosted Decision Tree")
        st.write("Nu hvor vi har set hvordan træet virker, vil vi gerne prøve at forudsige typen af partikler som vi ikke kender typen af på forhånd. Som vi har set, kan det være svært at minimere vores 'loss function'. En måde at forbedre på er ved at køre boosted decision trees, hvilket vil sige at vi kører flere træer, hvor den hver gang lærer af fejlene fra det forrige træ, og på den måde bliver 'boostet' for hvert træ den laver. Herunder kan vi ændre hvor mange gange den må 'booste', altså hvor mange træer den må lave og lærer af.")
        
        boosting_rounds = st.slider("Antal boosting rounds", min_value=1, max_value=1000, value=100, step=1)
        st.write("Vi kan også vælge hvor stor en andel af data vi vil bruge. ")
        andel_af_data = st.slider("Andel af data til træning", min_value=0.001, max_value=1.0, value=1.0, step=0.001)
        
        #Vi omdefinerer vores input og truth data til kun at indeholde en del af dataene.
        input_data_justeret, truth_data_justeret = sklearn.utils.resample(
            input_data, truth_data, 
            n_samples=int(andel_af_data * len(input_data)), 
            random_state=42, 
            replace=False
            )
        st.write("""Vi splitter data i et træningssæt og et testsæt.
Træningssættet bruges til at træne modellen, hvor modellen får at vide om data er en elektron eller ej.
Testsættet bruges til at give den trænede model ny data (som den ikke kender svaret til), som den så skal forudsige, men hvor vi stadig kender svaret.""")
        data_train, data_test, label_train, label_test = \
    sklearn.model_selection.train_test_split(input_data_justeret, truth_data_justeret, test_size=0.25, random_state=42)
    
        # Her bygger vi modellen op med flere træer, træner på data og forudsiger priser
        #Implement button to run below model
        if st.button("Kør model"):
            gbm_test = lgb.LGBMClassifier(n_estimators=boosting_rounds,# num_leaves=6,
                              boosting_type='gbdt', objective='binary', 
                              random_state=42)

            gbm_test.fit(data_train, label_train, eval_set=[(data_test, label_test)], 
            callbacks=[early_stopping(15)])

            # Her får vi sandsynlighederne for om hver person har diabetes eller ej
            Forudsigelse = gbm_test.predict_proba(data_test, num_iteration=gbm_test.best_iteration_)[:,1]
            
            plotting_partikel(label_test, Forudsigelse)


            st.subheader("Evaluer resultat med AUC og histogram")
            st.write("Nu vil vi gerne inspicere hvor god vores model er til at forusige på data hvor den ikke ved om data tilsvarer en elektron eller ej. Det venstre plot viser en ROC-kurve dvs. hvor stor en andel af sande gæt har vi per andel af forkerte gæt. Jo tættere denne er på venstre øverste hjørne jo bedre. Dvs. når raten af forkerte gæt er 0.1 er raten af korrekte gæt allrede omkring 0.9.")
            st.write("Selve scoren Area Under Curve (AUC) angiver bare hvor tæt på hjørnet grafen er. 1 angiver en perfekt score.")
            st.write("Det højre plot viser fordelingen af korrekte og forkerte gæt farvekodet efter hvad data rent faktisk svarede til. Dvs vi kigger på hvad modellen har gættet på ud fra hvad vores data rent faktisk svarede til. Den røde linjer svarer til den grænse modellen bruger til at afgøre hvad den skal gætte på alt efter hvilken sandsynlighed den forudsiger.")
            st.subheader("Spørgsmål")
            st.markdown("""
- Ændr på antallet af boosting_rounds og se hvad der sker med modellene og resultatet. Kan du se forskel i performance for f.eks. 1, 10, 100, 1000 boosting_rounds?
- Hvad sker der med fordelingen af data i højre plot når du ændrer på boosting_rounds? Kan du stadig godt klassificere elektroner ved boosting_rounds=1 eller boosting_rounds=10? (Bemærk den stiplede linje er defineret ved 0.5 og modellen har ikke indflydelse på den.)
- Leg rundt med andelen af data du bruger. Hvordan ændres resultatet alt efter hvor meget data den har. Hvor meget data skal du bruge for at have en rimelig model og forudsigelse?
                        """)
            st.subheader("Hvilke variabler er vigtigst?")
            st.write("Vi kan tjekke om vores hvilke variabler der er vigtigst for modellen til at lave en forudsigelse med 'permutation importance'. Det er et mål for hvis værdierne i en kolonne bliver byttet rundt randomly, hvor meget påvirker det så resultatet. Hvis det er en vigtig variabel, vil det påvirke resultatet meget. Her bliver det mål på hvor meget større mean squared error bliver, når den variabel bliver 'scramblet'.")
            

            perm_importance = sklearn.inspection.permutation_importance(gbm_test, data_test, label_test,scoring='neg_log_loss', random_state=42)
            order = perm_importance.importances_mean.argsort()[::1]
            labels = np.asarray(variable[:-1])[order]
            vals = perm_importance.importances_mean[order]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            y = np.arange(len(vals))
            ax.barh(y, vals)
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.set_xlabel("Increase in log_loss (permutation)")
            ax.set_ylabel("Feature")
            ax.set_title("Permutation Importance")
            #ax.invert_yaxis()
            fig.tight_layout()
            st.pyplot(fig)

        #NN 
        st.subheader("Neurale Netværk")
        st.write("Neurale Netværk (NN) kommer fra at opbygningen af det, minder om den måde vores neuroner i hjernen snakker sammen på. På samme måde som et decision tree er der forskellige lag og vi kan styre hvor mange lag der er, men nu er det ikke kun sandt eller falsk, i stedet fungerer noderne som knapper der kan fintunes. ")
        st.write("Neurale netværk er mere følsomme overfor det data vi giver dem. Den fungerer bedst hvis værdierne af data er mellem 0 og 1. Derfor bruger vi en funktion til at skalere vores data, kaldet StandardScaler.")
        scaler = sklearn.preprocessing.StandardScaler()
        data_train_transformed = scaler.fit_transform(data_train)
        data_test_transformed = scaler.transform(data_test)    
        
        st.write("I et neuralt netværk kan vi justere på hvor mange lag og hvor mange noder hvert lag skal have:")

        #Make six slider, one for each layer. that is six layers in total. sliders decide amount of nodes per layer
        layer_one = st.slider("Antal noder i lag 1", min_value=1, max_value=32, value=32, step=1)
        layer_two = st.slider("Antal noder i lag 2", min_value=1, max_value=32, value=16, step=1)
        layer_three = st.slider("Antal noder i lag 3", min_value=1, max_value=32, value=8, step=1)
        layer_four = st.slider("Antal noder i lag 4", min_value=1, max_value=32, value=4, step=1)
        layer_five = st.slider("Antal noder i lag 5", min_value=1, max_value=32, value=2, step=1)
        layer_six = st.slider("Antal noder i lag 6", min_value=1, max_value=32, value=2, step=1)


        st.write("""Nedenfor træner vi modellen. Vi kan også regne ud hvor mange parametre modellen bruger.
Herefter plotter vi for at se hvor godt modellen klarer sig.
                 Det kan godt tage op til ~et minut at køre denne model.""")
        if st.button("Kør Neuralt Netværk"):
            # Her definerer og træner vi modellen
            mlp = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(layer_one, layer_two, layer_three, layer_four, layer_five, layer_six), 
            max_iter=2000, early_stopping=True, random_state=42)
            mlp.fit(data_train_transformed, label_train) 

            # Her giver vi den trænede model test data som den ikke har set før, og beder om at forudsige prisen
            Forudsigelse = mlp.predict_proba(data_test_transformed)[:,1]

            # Beregn antal parametre i modellen
            # Coef er vægtene er intercept er bias. Den henter antallet direkte fra modellen.
            n_params = sum(coef.size + intercept.size for coef, intercept in zip(mlp.coefs_, mlp.intercepts_))
            st.write(f"Antal parametre i NN: {n_params}")
            plotting_partikel(label_test, Forudsigelse)
            st.subheader("Spørgsmål:")
            st.markdown("""
- Sammenlign modellen med boosted decision tree ovenover. Hvilken algoritme klarer sig bedst?
- Ændr antallet af neuroner per lag/antallet af lag og se hvordan performance ændrer sig.
- Får du det samme antal parametre når du regner efter?
- Leg rundt med andelen af data du bruger. Hvordan ændres resultatet alt efter hvor meget data den har. Hvor meget data skal du bruge for at have en rimelig model og forudsigelse?""")        
    # elif dataset == "Upload dit eget datasæt":
    #     st.subheader("Standard Niveau - Upload dit eget datasæt")
    #     st.write("Indhold for Standard Niveau og Upload dit eget datasæt.")
    #     # Add standard-level content for uploaded dataset here

if __name__ == "__main__":
    main()