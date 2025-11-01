import streamlit as st
import pandas as pd
#Load data from dataloader
from utils.data_loader import load_huspriser_dataset, load_diabetes_dataset, load_gletsjer_dataset
import os
from utils.config import DATA_PATHS
from utils.plots import plotting
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



def main():
    st.title("🎯 Standard Niveau")

    # Load datasets using cached functions
    DS1 = load_huspriser_dataset()
    DS2 = load_diabetes_dataset()
    DS3 = load_gletsjer_dataset()

    # Dataset selection
    st.sidebar.header("Datasæt")
    dataset = st.sidebar.radio("Vælg et datasæt:", ["Huspriser", "Diabetes", "Gletsjer"])

    # Add description
    st.write('"Standard" indeholder kun de mest centrale koncepter indenfor ML og ligger sig tæt op af vejledningen. Dette er et godt sted at starte hvis du ikke har arbejdet med ML før.')
    st.write("Vælg et datasæt for at begynde.")

    # Display the selected dataset with scrolling enabled and limited to 5 rows tall
    st.subheader(f"Visualisering af {dataset}")
    if dataset == "Huspriser":
        st.dataframe(DS1, height=200, use_container_width=True)
    elif dataset == "Diabetes":
        st.dataframe(DS2, height=200, use_container_width=True)
    elif dataset == "Gletsjer":
        st.dataframe(DS3, height=200, use_container_width=True)
    

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
        st.write("For et decision tree kan vi justede på hvor mange lag der skal være i vores træ, altså hvor mange lag af spørgsmål der må stilles. Vi kan justere på den parameter herunder.")

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
        st.write("""vi bruger train_test_split til at splitte data i et træningssæt og et testsæt.
træningssættet bruges til at træne modellen, hvor modellen får salgspriserne at vide.
testsættet bruges til at give den trænede model data uden salgspriser, som den så skal forudsige, men hvor vi stadig kender svaret""")
        data_træning, data_test, sande_pris_træning, sande_pris_test = \
        sklearn.model_selection.train_test_split(input_data_justeret, truth_data_justeret, test_size=0.25, random_state=42)
    
        # Her bygger vi modellen op med flere træer, træner på data og forudsiger priser
        #Implement button to run below model
        if st.button("Kør model"):
            gbm_test = lgb.LGBMRegressor(objective='regression', n_estimators=boosting_rounds, verbosity=-1)

            gbm_test.fit(data_træning, sande_pris_træning, eval_set=[(data_test, sande_pris_test)], 
                        eval_metric='mse', callbacks=[early_stopping(15)])

            forudsagte_pris = gbm_test.predict(data_test, num_iteration=gbm_test.best_iteration_)
            plotting(sande_pris_test, forudsagte_pris)

            res = sklearn.inspection.permutation_importance(gbm_test, data_test, sande_pris_test, scoring="neg_mean_squared_error")
        
            st.write("Nu vil vi gerne inspicere hvor god vores model er til at forusige på data hvor den ikke kender prisen. Det venstre plot viser residualerne, altså sande værdi - forudsagte værdi. Det højre plot er sande værdi vs forudsagte værdi. Her er også konturer (de sorte linjer), der viser tætheden af punkterne.")
            st.subheader("Spørgsmål")
            st.markdown("""
                        - Prøv at ændre på hvor mange gange gange den må booste, ved at ændre boosting_rounds fra 1 til 10, 100 eller 1000. Kan du se en forbedring?
                        - Hvor har modellen sværest ved at forudsige prisen? Er det ved de billigste huse, de dyreste, eller dem i mellem? Hvad kan det være? Hvilke huse tror du der er mest data på?
                        - Leg rundt med andelen af data du bruger. Hvordan ændres resultatet alt efter hvor meget data den har. Hvor meget data skal du bruge for at have en rimelig model og forudsigelse?""")
            st.subheader("Hvilke varaibler er vigtigst?")
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

        #NN 
        st.subheader("Neurale Netværk")
        st.write("Neurale Netværk (NN) kommer fra at opbygningen af det, minder om den måde vores neuroner i hjernen snakker sammen på. På samme måde som et decision tree er der forskellige lag og vi kan styre hvor mange lag der er, men nu er det ikke kun sandt eller falsk, i stedet fungerer noderne som knapper der kan fintunes. ")
        st.write("Neurale netværk er mere følsomme overfor det data vi giver dem. Den fungerer bedst hvis resultatet er værdier mellem 0 og 1. Derfor bruger vi en funktion til at skalere vores data, kaldet StandardScaler.")
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


        st.write("""Nedenfor træner vi modellen. Vi kan også regne ud hvor mange parameter modellen bruger.
Herefter plotter vi for at se hvor godt modellen klarer sig.""")
        if st.button("Kør Neuralt Netværk"):
            # Her definerer og træner vi modellen
            mlp = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(layer_one, layer_two, layer_three, layer_four, layer_five, layer_six), 
            max_iter=2000, early_stopping=True, random_state=42)
            mlp.fit(data_træning, sande_pris_træning) 
            # Her giver vi den trænede model test data som den ikke har set før, og beder om at forudsige prisen
            forudsagte_pris = mlp.predict(data_test)  
            # Beregn antal parametre i modellen
            # Coef er vægtene er intercept er bias. Den henter antallet directe fra modellen.
            n_params = sum(coef.size + intercept.size for coef, intercept in zip(mlp.coefs_, mlp.intercepts_))
            st.write(f"Antal parametre i NN: {n_params}")
            plotting(sande_pris_test, forudsagte_pris)
            st.subheader("Spørgsmål:")
            st.markdown("""- Prøv at justere på antal neuroner i det neurale netværk - Kan du mindske usikkerheden?
- Får du det samme antal parameter når du regner efter?
- Hvilken algoritme klarer sig bedst? Boosted decision tree eller neutralt netværk?
- Leg rundt med andelen af data du bruger. Hvordan ændres resultatet alt efter hvor meget data den har. Hvor meget data skal du bruge for at have en rimelig model og forudsigelse?""")




    #DIABETES .ipynb
    elif dataset == "Diabetes":
        st.subheader("Standard Niveau - Diabetes")
        st.write("Algoritmen forudsiger om en person har diabetes ud fra de resterende variable.")

        # Automatically set the target column to 'Diabetes'
        target_column = 'Diabetes'

        # Button to run the model
        if st.button("Kør klassifikations model"):
            # Prepare data
            X = DS2.drop(columns=[target_column])
            y = DS2[target_column]

            # Handle missing values
            X = X.fillna(0)
            y = y.fillna(0)

            # Import necessary libraries

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = LGBMClassifier(random_state=42)
            model.fit(X_train, y_train)

            # Predict and evaluate
            y_pred = model.predict(X_test)

            # Add predictions to the dataset
            predictions_df = X_test.copy()
            predictions_df[target_column] = y_test.values
            predictions_df['Forudsigelse'] = y_pred

            # Display the dataset with predictions in a scrollable window
            st.write("Datasæt med forudsigelser:")
            st.dataframe(predictions_df, height=200, use_container_width=True)

            # Show accuracy score
            score = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy for klassifikation på {target_column}: {score:.4f}")

            # ROC Curve visualization
            st.subheader("📈 ROC Curve")
            
            # Get prediction probabilities
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Create ROC plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve for Diabetes Classification')
            ax.legend(loc="lower right")
            ax.grid(True)
            
            st.pyplot(fig)
            st.write(f"AUC Score: {roc_auc:.4f}")

            # Histogram of predicted probabilities by actual class
            st.subheader("Fordeling af syge og raske patienter som funktion af modellens forudsigelse")
            
            # Split probabilities by actual class (ground truth)
            prob_no_diabetes = y_pred_proba[y_test == 0]  # Actually no diabetes
            prob_diabetes = y_pred_proba[y_test == 1]     # Actually diabetes
            
            # Create histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(prob_no_diabetes, bins=30, alpha=0.6, color='lightblue', 
                   label=f'Ingen diabetes (n={len(prob_no_diabetes)})', edgecolor='black')
            ax.hist(prob_diabetes, bins=30, alpha=0.6, color='lightcoral', 
                   label=f'Diabetes (n={len(prob_diabetes)})', edgecolor='black')
            ax.set_xlabel('Forudsagt sandsynlighed for diabetes')
            ax.set_ylabel('Antal')
            ax.set_title('Fordeling af forudsagte sandsynligheder efter faktisk diagnose')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Beslutningsgrænse (0.5)')
            ax.legend()
            
            st.pyplot(fig)
            
            # Show some statistics
            col1, col2 = st.columns(2)
            with col1:
#                st.metric("Gennemsnitlig sandsynlighed - Ingen diabetes", f"{np.mean(prob_no_diabetes):.3f}")
                st.metric("Antal korrekt klassificeret - Ingen diabetes", f"{len(prob_no_diabetes[prob_no_diabetes < 0.5])}")
            with col2:
#                st.metric("Gennemsnitlig sandsynlighed - Diabetes", f"{np.mean(prob_diabetes):.3f}")
                st.metric("Antal korrekt klassificeret - Diabetes", f"{len(prob_diabetes[prob_diabetes >= 0.5])}")

        # Option to display the code
        if st.checkbox("Vis kode for klassifikations model"):
            code = f"""
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Prepare data
X = DS2.drop(columns=['{target_column}'])
y = DS2['{target_column}']

# Handle missing values
X = X.fillna(0)
y = y.fillna(0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {{score}}")
"""
            st.code(code, language="python")
        
    elif dataset == "Gletsjer":
        st.subheader("Standard Niveau - Gletsjer")
        st.write("Algoritmen forudsiger dybden af gletsjeren ud fra de resterende variable.")

        # Automatically set the target column to the last column in the dataset
        target_column = DS3.columns[-1]

        # Dropdown menu to select error metric
        error_metric = st.selectbox("Hvordan vurderer modellen hvad der er et godt spørgsmål at stille?", ["Kvadreret fejl", "Absolut fejl"])

        # Button to run the model
        if st.button("Kør regression model"):
            if target_column:
                # Prepare data
                X = DS3.drop(columns=[target_column])
                y = DS3[target_column]

                # Handle missing values
                X = X.fillna(0)
                y = y.fillna(0)

                # Import train_test_split before usage
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train model with selected loss function
                if error_metric == "Kvadreret fejl":
                    model = LGBMRegressor(objective='regression', random_state=42)  # L2 loss (MSE)
                elif error_metric == "Absolut fejl":
                    model = LGBMRegressor(objective='regression_l1', random_state=42)  # L1 loss (MAE)
                    
                model.fit(X_train, y_train)

                # Predict and evaluate
                y_pred = model.predict(X_test)

                # Add predictions to the dataset
                predictions_df = X_test.copy()
                predictions_df[target_column] = y_test.values
                predictions_df['Forudsigelse'] = y_pred

                # Display the dataset with predictions in a scrollable window
                st.write("Datasæt med forudsigelser:")
                st.dataframe(predictions_df, height=200, use_container_width=True)

                if error_metric == "Kvadreret fejl":
                    error = mean_squared_error(y_test, y_pred)
                    st.write(f"Kvadreret fejl for regression på {target_column}: {error} kr.")
                elif error_metric == "Absolut fejl":
                    error = mean_absolute_error(y_test, y_pred)
                    st.write(f"Absolut fejl for regression på {target_column}: {error} kr.")

                # True vs Predicted plot
                st.subheader("Rigtig vs forudsagt plot")
                
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_test, y_pred, alpha=0.6, color='blue')
                
                # Perfect prediction line
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfekt foudsigelse')
                
                ax.set_xlabel('Rigtige værdier')
                ax.set_ylabel('Forudsagte værdier')
                ax.set_title('Rigtige vs forudsagte værdier')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)

                # Histogram of residuals
                st.subheader("📊 Histogram over residualer")
                residuals = y_test - y_pred
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(residuals, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
                ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Residual')
                ax.set_xlabel('Residualer (Rigtig - Forudsagt)')
                ax.set_ylabel('Hyppighed')
                ax.set_title('Fordeling af residualer')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Residual statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Gennemsnit af residualer", f"{np.mean(residuals):.3f}")
                with col2:
                    st.metric("Usikkerhed på residualer", f"{np.std(residuals):.3f}")
                with col3:
                    st.metric("Max |residualer|", f"{np.max(np.abs(residuals)):.3f}")

        # Option to display the code
        if st.checkbox("Vis kode for regression model"):
            if error_metric == "Kvadreret fejl":
                objective_code = "model = LGBMRegressor(objective='regression', random_state=42)  # L2 loss (MSE)"
            else:
                objective_code = "model = LGBMRegressor(objective='regression_l1', random_state=42)  # L1 loss (MAE)"
            
            code = f"""
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Prepare data
X = DS3.drop(columns=['{target_column}'])
y = DS3['{target_column}']

# Handle missing values
X = X.fillna(0)
y = y.fillna(0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with selected objective
{objective_code}
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
if error_metric == "MSE":
    error = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {{error}}")
elif error_metric == "MAE":
    error = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {{error}}")
"""
            st.code(code, language="python")
        
    # elif dataset == "Upload dit eget datasæt":
    #     st.subheader("Standard Niveau - Upload dit eget datasæt")
    #     st.write("Indhold for Standard Niveau og Upload dit eget datasæt.")
    #     # Add standard-level content for uploaded dataset here

if __name__ == "__main__":
    main()