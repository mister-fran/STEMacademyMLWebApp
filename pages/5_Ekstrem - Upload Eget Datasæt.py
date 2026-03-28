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
st.set_page_config(page_title="Avanceret Niveau")

#Reset session state
PAGE_ID = "Ekstrem"  # change per page
# If we arrived here from another page, reset this page's state
if st.session_state.get("_active_page") != PAGE_ID:
    st.write(st.session_state.get("_active_page"))
    st.session_state.clear()
    st.session_state["_active_page"] = PAGE_ID
    st.rerun()

def main():
    st.title("Ekstrem - Upload Eget Datasæt")

    # Load datasets using cached functions
    DS3 = load_gletsjer_dataset()
    DS4 = load_partikel_dataset()

    # Dataset selection
    st.sidebar.header("Metode")
    dataset = st.sidebar.radio("Vælg metode:", ["Regression", "Classification"])

    # Add description
    st.write('En stor del af kusten ved at lave ML analyser er at forberede et "nydeligt datasæt." '\
    'Opgaven gør ud på at tilpasse et datasæt til brug modellen og ellers lege rundt med den. '\
    'Siden indeholder samme funktionalitet som under "Avanceret".')
    st.write("Ik' giv op hvis dette ikke virker første gang. Du kan også se filen 'Dataset_preprocessing.ipynb' på GitHub for at gøre dette nemmere.")

    # Add a download link for guidance PDF in the sidebar
    pdf_path = 'data/vejledning.pdf'  # Put your PDF file here
    
    st.sidebar.write("") # Add vertical space above button

    
    if dataset == "Regression":
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
            st.write('Fjern et lag fra modellen ved at sætte det til 0.')
            layer_one   = st.slider("Antal noder i lag 1", min_value=0, max_value=32, value=4, step=1)             
            layer_two   = st.slider("Antal noder i lag 2", min_value=0, max_value=32, value=0, step=1)            
            layer_three = st.slider("Antal noder i lag 3", min_value=0, max_value=32, value=0, step=1)
            layer_four  = st.slider("Antal noder i lag 4", min_value=0, max_value=32, value=0, step=1)
            layer_five  = st.slider("Antal noder i lag 5", min_value=0, max_value=32, value=0, step=1)
            layer_six   = st.slider("Antal noder i lag 6", min_value=0, max_value=32, value=0, step=1)

            hidden_layer_sizes = (layer_one, layer_two, layer_three, layer_four, layer_five, layer_six)
            #choose the entries of hidden_layer_sizes that are !=0
            hidden_layer_sizes = tuple(l for l in hidden_layer_sizes if l > 0)

            st.markdown("**VALGT ARKITEKTUR:** " + str(hidden_layer_sizes))    

            activation = st.selectbox("Hvilken activation function skal vi bruge?", options=['relu', 'tanh', 'logistic'], index=0)
            learning_rate_nn = st.selectbox("Hvilken slags learning rate skal vi bruge?", options=['constant', 'invscaling', 'adaptive'], index=0)
            max_iter = st.slider("Maksimalt antal iterationer (svarer til boosting_rounds for BDT)", min_value=1, max_value=2000, value=200, step=1)
            alpha = st.slider("Intern regulariseringsparameter for at forhindre overfitting", min_value=0.0001, max_value=0.1, value=0.0001, step=0.0001, format="%.4f")
            early_stopping_nn= st.checkbox("Brug early stopping?", value=True)

            if st.button("Kør Neuralt Netværk"):
                # Her definerer og træner vi modellen
                mlp = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, 
                max_iter=max_iter, activation=activation,learning_rate=learning_rate_nn, alpha = alpha, early_stopping=early_stopping_nn, random_state=42)
                mlp.fit(data_træning, sand_dybde_træning)
                # Her giver vi den trænede model test data som den ikke har set før, og beder om at forudsige dybden
                forudsagt_dybde = mlp.predict(data_test)  

                # Beregn antal parametre i modellen
                # Coef er vægtene er intercept er bias. Den henter antallet directe fra modellen.
                n_params = sum(coef.size + intercept.size for coef, intercept in zip(mlp.coefs_, mlp.intercepts_))
                st.write(f"Antal parametre i NN: {n_params}")
                plotting_reg_own(sand_dybde_test, forudsagt_dybde)

    elif dataset == "Classification":
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
            st.write('Fjern et lag fra modellen ved at sætte det til 0.')
            layer_one   = st.slider("Antal noder i lag 1", min_value=0, max_value=32, value=4, step=1)             
            layer_two   = st.slider("Antal noder i lag 2", min_value=0, max_value=32, value=0, step=1)            
            layer_three = st.slider("Antal noder i lag 3", min_value=0, max_value=32, value=0, step=1)
            layer_four  = st.slider("Antal noder i lag 4", min_value=0, max_value=32, value=0, step=1)
            layer_five  = st.slider("Antal noder i lag 5", min_value=0, max_value=32, value=0, step=1)
            layer_six   = st.slider("Antal noder i lag 6", min_value=0, max_value=32, value=0, step=1)

            hidden_layer_sizes = (layer_one, layer_two, layer_three, layer_four, layer_five, layer_six)
            #choose the entries of hidden_layer_sizes that are !=0
            hidden_layer_sizes = tuple(l for l in hidden_layer_sizes if l > 0)

            st.markdown("**VALGT ARKITEKTUR:** " + str(hidden_layer_sizes))    

            activation = st.selectbox("Hvilken activation function skal vi bruge?", options=['relu', 'tanh', 'logistic'], index=0)
            learning_rate_nn = st.selectbox("Hvilken slags learning rate skal vi bruge?", options=['constant', 'invscaling', 'adaptive'], index=0)
            max_iter = st.slider("Maksimalt antal iterationer (svarer til boosting_rounds for BDT)", min_value=1, max_value=2000, value=200, step=1)
            alpha = st.slider("Intern regulariseringsparameter for at forhindre overfitting", min_value=0.0001, max_value=0.1, value=0.0001, step=0.0001, format="%.4f")
            early_stopping_nn= st.checkbox("Brug early stopping?", value=True)
            beslutningsgrænse_nn = st.slider("Beslutningsgrænse ", min_value=0.0, max_value=1.0, value=0.5, step=0.01)        
            if st.button("Kør Neuralt Netværk"):
                # Her definerer og træner vi modellen
                mlp = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, 
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