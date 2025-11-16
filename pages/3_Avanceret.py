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
st.set_page_config(page_title="Avanceret Niveau", page_icon="🎯")



def main():
    st.title("🎯 Avanceret Niveau")

    # Load datasets using cached functions
    DS3 = load_gletsjer_dataset()
    DS4 = load_partikel_dataset()

    # Dataset selection
    st.sidebar.header("Datasæt")
    dataset = st.sidebar.radio("Vælg et datasæt:", ["Gletsjer", "Partikel", "Upload eget datasæt - Regression", "Upload eget datasæt - Classification"])

    # Add description
    st.write('Alternativ til at køre .ipynb filen lokalt på din computer. Indeholder samme funktionaliteter som .ipynb filerne med uden at man skal skrive/se kode selv. ' \
    'Avanceret indeholder mulighed for at ændre på de nævnte hyperparamtre i spørgsmålene plus vælge inputvariable.')
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
    if dataset == "Gletsjer":
        #HER BEGYNDER VORES .ipynb
        st.subheader("Avanceret Niveau - Gletsjer")
        st.write("Nedenfor skal du hjælpe gletsjervidenskabsfakultetet med at udvikle deres ML model til at bestemme dybden af gletsjere. På denne hjemmeside behøver vi ikke importere nogen pakker da det er tilrettelagt således at man skal kunne lege med ML-modellerne uden at skulle bekymre sig om koden bag dem.")

        #Inspicer dataen
        st.subheader("Inspicer dataen")
        st.write("Først vil vi gerne undersøge hvilken data vi har med at gøre.")
        st.dataframe(DS3, height=200, use_container_width=True)
        
        #Tilrettelæg data
        #Gør så jeg selv kan vælge hvilke varaible jeg vil have med

        st.write("Vælg hvilke variable du vil bruge til at træne din model.")
        #Remove gletsjer_dybde from options
        options = [col for col in DS3.columns.tolist() if col != 'gletsjer_dybde']
        input_variabler = st.multiselect("Vælg input variabler", options=options, default=options)
        variabler = input_variabler + ['gletsjer_dybde']
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
        st.subheader('Advanced - Hyperparametre')
        num_leaves = st.slider("Maksimalt antal kasser", min_value=10, max_value=100, value=31, step=1)
        boosting_type = st.selectbox("Hvilken algoritme bruger vi til at booste?", options=['gbdt', 'dart'], index=0)
        max_depth = st.slider("Hvor mange lad må der maksimalt være i vores træ?", min_value=-1, max_value=100, value=-1, step=1)
        learning_rate_bdt = st.slider("Hvor store skridt tager modellen?", min_value=0.001, max_value=0.5, value=0.01, step=0.001)
        min_child_samples = st.slider("Minimum antal samples i hver kasse", min_value=1, max_value=100, value=20, step=1)
        
        if st.button("Kør model"):
            gbm_test = lgb.LGBMRegressor( objective='regression', n_estimators=boosting_rounds,num_leaves=num_leaves, boosting_type=boosting_type, max_depth=max_depth, learning_rate=learning_rate_bdt, min_child_samples=min_child_samples, verbosity=-1)

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

        st.subheader('Advanced - Hyperparametre')
        activation = st.selectbox("Hvilken activation function skal vi bruge?", options=['relu', 'tanh', 'logistic'], index=0)
        learning_rate_nn = st.selectbox("Hvilken slags learning rate skal vi bruge?", options=['constant', 'invscaling', 'adaptive'], index=0)
        max_iter = st.slider("Maksimalt antal iterationer (svarer til boosting_rounds for BDT)", min_value=1, max_value=2000, value=200, step=1)
        alpha = st.slider("Intern regulariseringsparameter for at forhindre overfitting", min_value=0.0001, max_value=0.1, value=0.0001, step=0.0001, format="%.4f")
        early_stopping_nn= st.checkbox("Brug early stopping?", value=True)
        st.write("""Nedenfor træner vi modellen. Vi kan også regne ud hvor mange parametre modellen bruger.
Herefter plotter vi for at se hvor godt modellen klarer sig.
                 Det kan godt tage op til ~et minut at køre denne model.""")
        if st.button("Kør Neuralt Netværk"):
            # Her definerer og træner vi modellen
            mlp = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(layer_one, layer_two, layer_three, layer_four, layer_five, layer_six), 
            max_iter=max_iter, activation=activation,learning_rate=learning_rate_nn, alpha = alpha, early_stopping=early_stopping_nn, random_state=42)
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
            st.subheader("Avancerede Spørgsmål:")
            st.markdown("""
                        - Leg rundt med hyperparametre (HP) i begge modeller (BDT og NN). Tilføj og fjern, ændr deres værdier, kør modellen og se hvordan den performer.  
                        - Prøv at optimere dene så du får den bedste performance ved at prøve forskellige kombinationer af HP af. Kan du komme i tanke op måder man ville kunne optimere/strukturere denne process på?
                        - Slå sklearn's GridSearchCV og RandomizedSearchCV op og find ud af hvad de gør. Hvad er fordele/ulemper ved begge?
                        - Fjern early_stopping. Hvad gør det. Er det en fordel? Kan du risikere at overtræne hvis du ikke gør?
                        - Er alle kolonner lige vigtige? Kør modellerne med kun de 5 bedste/værste variable og se deres performance.
                        """)


    #Partikel
    elif dataset == "Partikel":
        #HER BEGYNDER VORES .ipynb
        st.subheader("Avanceret Niveau - Partikel")
        st.write("Som nævnt, er du blevet udnævnt personligt til at klassificere elektroner på CERN. På denne hjemmeside behøver vi ikke importere nogen pakker da det er tilrettelagt således at man skal kunne lege med ML-modellerne uden at skulle bekymre sig om koden bag dem.")

        #Inspicer dataen
        st.subheader("Inspicer dataen")
        st.write("Først vil vi gerne undersøge hvilken data vi har med at gøre.")
        st.dataframe(DS4, height=200, use_container_width=True)
        
        #Tilrettelæg data
        #Gør så jeg selv kan vælge hvilke varaible jeg vil have med
        st.write("Vælg hvilke variable du vil bruge til at træne din model.")
        #Remove pTruth_isElectron from options
        options = [col for col in DS4.columns.tolist() if col != 'p_Truth_isElectron']
        input_variable = st.multiselect("Vælg input variabler", options=options, default=options)
        variable = input_variable + ['p_Truth_isElectron']
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
        st.subheader('Advanced - Hyperparametre')
        num_leaves = st.slider("Maksimalt antal kasser", min_value=10, max_value=100, value=31, step=1)
        boosting_type = st.selectbox("Hvilken algoritme bruger vi til at booste?", options=['gbdt', 'dart', 'rf'], index=0)
        max_depth = st.slider("Hvor mange lad må der maksimalt være i vores træ?", min_value=-1, max_value=100, value=-1, step=1)
        learning_rate_bdt = st.slider("Hvor store skridt tager modellen?", min_value=0.001, max_value=0.5, value=0.01, step=0.001)
        min_child_samples = st.slider("Minimum antal samples i hver kasse", min_value=1, max_value=100, value=20, step=1)
        
        if st.button("Kør model"):
            gbm_test = lgb.LGBMClassifier(n_estimators=boosting_rounds, num_leaves=num_leaves, max_depth=max_depth, learning_rate=learning_rate_bdt, min_child_samples=min_child_samples,
                              boosting_type=boosting_type, objective='binary', 
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

        st.subheader('Advanced - Hyperparametre')
        activation = st.selectbox("Hvilken activation function skal vi bruge?", options=['relu', 'tanh', 'logistic'], index=0)
        learning_rate_nn = st.selectbox("Hvilken slags learning rate skal vi bruge?", options=['constant', 'invscaling', 'adaptive'], index=0)
        max_iter = st.slider("Maksimalt antal iterationer (svarer til boosting_rounds for BDT)", min_value=1, max_value=2000, value=200, step=1)
        alpha = st.slider("Intern regulariseringsparameter for at forhindre overfitting", min_value=0.0001, max_value=0.1, value=0.0001, step=0.0001, format="%.4f")
        early_stopping_nn= st.checkbox("Brug early stopping?", value=True)
        
        st.write("""Nedenfor træner vi modellen. Vi kan også regne ud hvor mange parametre modellen bruger.
Herefter plotter vi for at se hvor godt modellen klarer sig.
                 Det kan godt tage op til ~et minut at køre denne model.""")
        if st.button("Kør Neuralt Netværk"):
            # Her definerer og træner vi modellen
            mlp = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(layer_one, layer_two, layer_three, layer_four, layer_five, layer_six), 
            max_iter=max_iter, activation=activation, learning_rate=learning_rate_nn, alpha=alpha, early_stopping=early_stopping_nn, random_state=42)
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
            st.subheader("Avancerede Spørgsmål:")
            st.markdown("""
                        - Leg rundt med hyperparametre (HP) i begge modeller (BDT og NN). Tilføj og fjern, ændr deres værdier, kør modellen og se hvordan den performer.  
                        - Prøv at optimere dene så du får den bedste performance ved at prøve forskellige kombinationer af HP af. Kan du komme i tanke op måder man ville kunne optimere/strukturere denne process på?
                        - Slå sklearn's GridSearchCV og RandomizedSearchCV op og find ud af hvad de gør. Hvad er fordele/ulemper ved begge?
                        - Fjern early_stopping. Hvad gør det. Er det en fordel? Kan du risikere at overtræne hvis du ikke gør?
                        - Er alle kolonner lige vigtige? Kør modellerne med kun de 5 bedste/værste variable og se deres performance.
                        """)

    if dataset == "Upload eget datasæt - Regression":
        #HER BEGYNDER VORES .ipynb
        st.subheader("Upload eget datasæt - Regression")
        st.write("Her kan du uploade eget datasæt og køre tilhørende ML modeller på det.")

        #Inspicer dataen
        st.subheader("Inspicer dataen")
        #Let the user upload their own csv file
        uploaded_file = st.file_uploader("Upload din egen CSV fil her", type=["csv"])
        if uploaded_file is not None:
            DS_OWN = pd.read_csv(uploaded_file)
            st.dataframe(DS_OWN, height=200, use_container_width=True)

        #Tilrettelæg data
        #Gør så jeg selv kan vælge hvilke varaible jeg vil have med

        st.write("Vælg hvilke variable du vil bruge til at træne din model og hvad vil du lave regression mod?")
        #Let user to choose variable to do regression on
        if uploaded_file is not None:
            options = DS_OWN.columns.tolist()
            target_variable = st.selectbox("Vælg target variabel (den du vil lave regression mod)", options=options, index=len(options)-1)
            input_variable = st.multiselect("Vælg input variabler", options=[col for col in options if col != target_variable], default=[col for col in options if col != target_variable])
            variable = input_variable + [target_variable]
            input_data = DS_OWN[input_variable].to_numpy()
            truth_data = DS_OWN[target_variable].to_numpy()
            #HER RYKKER VI ALT IND FOR AT UNDGÅ FEJL______________________________________________ 
            st.subheader("Boosted Decision Tree")
            st.write("Valg af hyperparametre for BDT.")

            andel_af_data = st.slider("Andel af data til træning", min_value=0.001, max_value=1.0, value=1.0, step=0.001)
            #Vi omdefinerer vores input og truth data til kun at indeholde en del af dataene.
            input_data_justeret, truth_data_justeret = sklearn.utils.resample(
                input_data, truth_data, 
                n_samples=int(andel_af_data * len(input_data)), 
                random_state=42, 
                replace=False
                )
            data_træning, data_test, sand_dybde_træning, sand_dybde_test = \
            sklearn.model_selection.train_test_split(input_data_justeret, truth_data_justeret, test_size=0.25, random_state=42)

            boosting_rounds = st.slider("Antal boosting rounds", min_value=1, max_value=1000, value=1, step=1)
            num_leaves = st.slider("Maksimalt antal kasser", min_value=10, max_value=100, value=31, step=1)
            boosting_type = st.selectbox("Hvilken algoritme bruger vi til at booste?", options=['gbdt', 'dart'], index=0)
            max_depth = st.slider("Hvor mange lad må der maksimalt være i vores træ?", min_value=-1, max_value=100, value=-1, step=1)
            learning_rate_bdt = st.slider("Hvor store skridt tager modellen?", min_value=0.001, max_value=0.5, value=0.01, step=0.001)
            min_child_samples = st.slider("Minimum antal samples i hver kasse", min_value=1, max_value=100, value=20, step=1)

            if st.button("Kør model"):
                gbm_test = lgb.LGBMRegressor( objective='regression', n_estimators=boosting_rounds,num_leaves=num_leaves, boosting_type=boosting_type, max_depth=max_depth, learning_rate=learning_rate_bdt, min_child_samples=min_child_samples, verbosity=-1)

                gbm_test.fit(data_træning, sand_dybde_træning, eval_set=[(data_test, sand_dybde_test)], 
                            eval_metric='mse', callbacks=[early_stopping(15)])

                forudsagt_dybde = gbm_test.predict(data_test, num_iteration=gbm_test.best_iteration_)
                plotting_reg_own(sand_dybde_test, forudsagt_dybde)

                res = sklearn.inspection.permutation_importance(gbm_test, data_test, sand_dybde_test, scoring="neg_mean_squared_error")

                st.subheader("Permutation Importance")

                imp_mse = res.importances_mean                
                order = np.argsort(imp_mse)[::-1]
                labels = np.asarray(variable[:-1])[order]
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

            #NN 
            st.subheader("Neurale Netværk")

            scaler = sklearn.preprocessing.StandardScaler()
            data_træning = scaler.fit_transform(data_træning)
            data_test = scaler.transform(data_test)


            st.subheader("Valg af hyperparametre for NN.")

            #Make six slider, one for each layer. that is six layers in total. sliders decide amount of nodes per layer
            layer_one = st.slider("Antal noder i lag 1", min_value=1, max_value=128, value=32, step=1)
            layer_two = st.slider("Antal noder i lag 2", min_value=1, max_value=128, value=16, step=1)
            layer_three = st.slider("Antal noder i lag 3", min_value=1, max_value=128, value=8, step=1)
            layer_four = st.slider("Antal noder i lag 4", min_value=1, max_value=128, value=4, step=1)
            layer_five = st.slider("Antal noder i lag 5", min_value=1, max_value=128, value=2, step=1)
            layer_six = st.slider("Antal noder i lag 6", min_value=1, max_value=128, value=2, step=1)

            activation = st.selectbox("Hvilken activation function skal vi bruge?", options=['relu', 'tanh', 'logistic'], index=0)
            learning_rate_nn = st.selectbox("Hvilken slags learning rate skal vi bruge?", options=['constant', 'invscaling', 'adaptive'], index=0)
            max_iter = st.slider("Maksimalt antal iterationer (svarer til boosting_rounds for BDT)", min_value=1, max_value=2000, value=200, step=1)
            alpha = st.slider("Intern regulariseringsparameter for at forhindre overfitting", min_value=0.0001, max_value=0.1, value=0.0001, step=0.0001, format="%.4f")
            early_stopping_nn= st.checkbox("Brug early stopping?", value=True)

            if st.button("Kør Neuralt Netværk"):
                # Her definerer og træner vi modellen
                mlp = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(layer_one, layer_two, layer_three, layer_four, layer_five, layer_six), 
                max_iter=max_iter, activation=activation,learning_rate=learning_rate_nn, alpha = alpha, early_stopping=early_stopping_nn, random_state=42)
                mlp.fit(data_træning, sand_dybde_træning)
                # Her giver vi den trænede model test data som den ikke har set før, og beder om at forudsige dybden
                forudsagt_dybde = mlp.predict(data_test)  

                # Beregn antal parametre i modellen
                # Coef er vægtene er intercept er bias. Den henter antallet directe fra modellen.
                n_params = sum(coef.size + intercept.size for coef, intercept in zip(mlp.coefs_, mlp.intercepts_))
                st.write(f"Antal parametre i NN: {n_params}")
                plotting_reg_own(sand_dybde_test, forudsagt_dybde)

    elif dataset == "Upload eget datasæt - Classification":
        #HER BEGYNDER VORES .ipynb
        st.subheader("Upload eget datasæt - Classification")
        st.write("Her kan du uploade eget datasæt og køre tilhørende ML modeller på det.")

        #Inspicer dataen
        st.subheader("Inspicer dataen")
        #Add possibility for user to upload their own csv file
        uploaded_file = st.file_uploader("Upload din egen CSV fil her", type=["csv"])
        if uploaded_file is not None:
            DS_OWN = pd.read_csv(uploaded_file)
            st.dataframe(DS_OWN, height=200, use_container_width=True)
        
        st.write("Vælg hvilke variable du vil bruge til at træne din model og hvad vil du lave regression mod?")
        #Let user to choose variable to do regression on
        if uploaded_file is not None:
            options = DS_OWN.columns.tolist()
            target_variable = st.selectbox("Vælg target variabel (den du vil lave regression mod)", options=options, index=len(options)-1)
            input_variable = st.multiselect("Vælg input variabler", options=[col for col in options if col != target_variable], default=[col for col in options if col != target_variable])
            variable = input_variable + [target_variable]
            input_data = DS_OWN[input_variable].to_numpy()
            truth_data = DS_OWN[target_variable].to_numpy()
            #HER RYKKER VI ALT IND FOR AT UNDGÅ FEJL______________________________________________
            st.subheader("Boosted Decision Tree")
            st.write("Vælg af hyperparametre for BDT.")
            
            andel_af_data = st.slider("Andel af data til træning", min_value=0.001, max_value=1.0, value=1.0, step=0.001)
            #Vi omdefinerer vores input og truth data til kun at indeholde en del af dataene.
            input_data_justeret, truth_data_justeret = sklearn.utils.resample(
                input_data, truth_data, 
                n_samples=int(andel_af_data * len(input_data)), 
                random_state=42, 
                replace=False
                )
            data_train, data_test, label_train, label_test = \
            sklearn.model_selection.train_test_split(input_data_justeret, truth_data_justeret, test_size=0.25, random_state=42)
    
            boosting_rounds = st.slider("Antal boosting rounds", min_value=1, max_value=1000, value=100, step=1)
            num_leaves = st.slider("Maksimalt antal kasser", min_value=10, max_value=100, value=31, step=1)
            boosting_type = st.selectbox("Hvilken algoritme bruger vi til at booste?", options=['gbdt', 'dart', 'rf'], index=0)
            max_depth = st.slider("Hvor mange lad må der maksimalt være i vores træ?", min_value=-1, max_value=100, value=-1, step=1)
            learning_rate_bdt = st.slider("Hvor store skridt tager modellen?", min_value=0.001, max_value=0.5, value=0.01, step=0.001)
            min_child_samples = st.slider("Minimum antal samples i hver kasse", min_value=1, max_value=100, value=20, step=1)
            
            beslutningsgrænse = st.slider("Beslutningsgrænse", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            
            if st.button("Kør model"):
                gbm_test = lgb.LGBMClassifier(n_estimators=boosting_rounds, num_leaves=num_leaves, max_depth=max_depth, learning_rate=learning_rate_bdt, min_child_samples=min_child_samples,
                                  boosting_type=boosting_type, objective='binary', 
                                  random_state=42)
    
                gbm_test.fit(data_train, label_train, eval_set=[(data_test, label_test)], 
                callbacks=[early_stopping(15)])
    
                # Her får vi sandsynlighederne for om hver person har diabetes eller ej
                Forudsigelse = gbm_test.predict_proba(data_test, num_iteration=gbm_test.best_iteration_)[:,1]
                forudsagte_klasse = gbm_test.predict_proba(data_test, num_iteration=gbm_test.best_iteration_)[:,1]
                forudsagte_klasse = (forudsagte_klasse > beslutningsgrænse).astype(int)
    
                plotting_class_own(label_test, Forudsigelse, forudsagte_klasse, beslutningsgrænse)
    
    
                st.subheader("Permutation Importance")            
    
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
            scaler = sklearn.preprocessing.StandardScaler()
            data_train_transformed = scaler.fit_transform(data_train)
            data_test_transformed = scaler.transform(data_test)    
            
            st.subheader("Valg af hyperparametre for NN.")
            #Make six slider, one for each layer. that is six layers in total. sliders decide amount of nodes per layer
            layer_one = st.slider("Antal noder i lag 1", min_value=1, max_value=128, value=32, step=1)
            layer_two = st.slider("Antal noder i lag 2", min_value=1, max_value=128, value=16, step=1)
            layer_three = st.slider("Antal noder i lag 3", min_value=1, max_value=128, value=8, step=1)
            layer_four = st.slider("Antal noder i lag 4", min_value=1, max_value=128, value=4, step=1)
            layer_five = st.slider("Antal noder i lag 5", min_value=1, max_value=128, value=2, step=1)
            layer_six = st.slider("Antal noder i lag 6", min_value=1, max_value=128, value=2, step=1)
    
            activation = st.selectbox("Hvilken activation function skal vi bruge?", options=['relu', 'tanh', 'logistic'], index=0)
            learning_rate_nn = st.selectbox("Hvilken slags learning rate skal vi bruge?", options=['constant', 'invscaling', 'adaptive'], index=0)
            max_iter = st.slider("Maksimalt antal iterationer (svarer til boosting_rounds for BDT)", min_value=1, max_value=2000, value=200, step=1)
            alpha = st.slider("Intern regulariseringsparameter for at forhindre overfitting", min_value=0.0001, max_value=0.1, value=0.0001, step=0.0001, format="%.4f")
            early_stopping_nn= st.checkbox("Brug early stopping?", value=True)
            beslutningsgrænse_nn = st.slider("Beslutningsgrænse ", min_value=0.0, max_value=1.0, value=0.5, step=0.01)        
            if st.button("Kør Neuralt Netværk"):
                # Her definerer og træner vi modellen
                mlp = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(layer_one, layer_two, layer_three, layer_four, layer_five, layer_six), 
                max_iter=max_iter, activation=activation, learning_rate=learning_rate_nn, alpha=alpha, early_stopping=early_stopping_nn, random_state=42)
                mlp.fit(data_train_transformed, label_train) 
    
                # Her giver vi den trænede model test data som den ikke har set før, og beder om at forudsige prisen
                Forudsigelse = mlp.predict_proba(data_test_transformed)[:,1]
                forudsagte_klasse_nn = mlp.predict_proba(data_test_transformed)[:,1]
                forudsagte_klasse_nn = (forudsagte_klasse_nn > beslutningsgrænse_nn).astype(int)
    
                # Beregn antal parametre i modellen
                # Coef er vægtene er intercept er bias. Den henter antallet direkte fra modellen.
                n_params = sum(coef.size + intercept.size for coef, intercept in zip(mlp.coefs_, mlp.intercepts_))
                st.write(f"Antal parametre i NN: {n_params}")
                plotting_class_own(label_test, Forudsigelse, forudsagte_klasse_nn, beslutningsgrænse_nn)
            

if __name__ == "__main__":
    main()