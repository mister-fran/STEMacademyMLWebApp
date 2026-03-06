import streamlit as st
import pandas as pd
#Load data from dataloader
from utils.data_loader import load_pendul_dataset
import os
from utils.config import DATA_PATHS
#from utils.plots import plotting, plotting_glet, plotting_partikel, Plotting_class, plotting_reg_own, plotting_class_own

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

#Kompliceret ligning
from scipy.special import ellipk

st.set_page_config(page_title="Standard Niveau", page_icon="🎯")

#Reset session state
PAGE_ID = "pendul"  # change per page
# If we arrived here from another page, reset this page's state
if st.session_state.get("_active_page") != PAGE_ID:
    st.write(st.session_state.get("_active_page"))
    st.session_state.clear()
    st.session_state["_active_page"] = PAGE_ID
    st.rerun()
    
def main():
    st.title("Modul Pendul - Pendulets periode med ML")

    # Load datasets using cached functions
    #DSPendul = load_pendul_dataset()

    # Add description
    st.write('Alternativ til at køre .ipynb filen lokalt på din computer. Indeholder samme funktionaliteter som .ipynb filerne med uden at man skal skrive/se kode selv.')
    st.write("Vælg et datasæt for at begynde.")    

    
    st.sidebar.write("") # Add vertical space above button

    # Download button for PDF PARTIKEL
    if os.path.exists(DATA_PATHS['VejledningPENDUL']):
        try:
            with open(DATA_PATHS['VejledningPENDUL'], "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
            
            st.sidebar.download_button(
                label="📥 Hent vejledning til Pendul",
                data=pdf_bytes,
                file_name="vejledningPENDUL.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.sidebar.error(f"Fejl ved indlæsning af PDF: {e}")
    else:
        st.sidebar.warning("⚠️ Vejledning PDF ikke fundet.")
    # Pendul

    ## Data
    st.subheader("Inspicer dataen")
    st.write("Først vil vi gerne undersøge hvilken data vi har med at gøre. ")

    #Load data
    data = load_pendul_dataset()

    #Vis data
    st.dataframe(data, height=200, use_container_width=True)

    #Redefinitions for later use
    variabler = data.columns
    input_variabler = [v for v in variabler if v != 'Period']
    input_data = data[input_variabler].to_numpy()
    truth_data = data['Period'].to_numpy()

    #Andel af data
    st.write("Nedenfor kan du vælge hvor stor en andel af data du vil bruge til at træne modellen.")
    # Vælg hvor stor andel. Tallet skal være mellem 0 og 1.0.
    andel_af_data = st.slider("Andel af data til træning", min_value=0.01, max_value=1.0, value=1.0, step=0.01)
    #Vi omdefinerer vores input og truth data til kun at indeholde en del af dataene.
    input_data_justeret, truth_data_justeret = sklearn.utils.resample(
    input_data, truth_data, 
    n_samples=int(andel_af_data * len(input_data)), 
    random_state=42, 
    replace=False
    )

    # Tilrettelæg til brug i model
    st.write("""Vi splitter datasættet i et træningssæt og et testsæt.
    Træningssættet bruges til at træne modellen, hvor modellen får salgspriserne at vide.
    Testsættet bruges til at give den trænede model data uden salgspriser, som den så skal forudsige, men hvor vi stadig kender svaret.""")
    data_træning, data_test, sand_periode_træning, sand_periode_test = \
        sklearn.model_selection.train_test_split(input_data_justeret, truth_data_justeret, test_size=0.25, random_state=42)
    ## Det neurale netværk

    st.write("For at data kan bruges i det neurale netværk er vi nødt til at skalere det således at det ligger fordelt omkring 0 (der er ~lige mange værdier over og under 0) og dets spredning er 1 (værdierne ligger tæt på 0).")
    scaler = sklearn.preprocessing.StandardScaler()
    data_træning_scaled = scaler.fit_transform(data_træning)
    data_test_scaled = scaler.transform(data_test)

    st.write("I et neuralt netværk kan vi justere på hvor mange lag og hvor mange noder hvert lag skal have:")
    layer_one   = st.slider("Antal noder i lag 1", min_value=1, max_value=128, value=2, step=1)             
    layer_two   = st.slider("Antal noder i lag 2", min_value=1, max_value=128, value=2, step=1)            
    layer_three = st.slider("Antal noder i lag 3", min_value=1, max_value=128, value=2, step=1)
    layer_four  = st.slider("Antal noder i lag 4", min_value=1, max_value=128, value=2, step=1)
    layer_five  = st.slider("Antal noder i lag 5", min_value=1, max_value=128, value=2, step=1)
    layer_six   = st.slider("Antal noder i lag 6", min_value=1, max_value=128, value=2, step=1)
    st.write("Nedenfor træner vi modellen. Vi kan også regne ud hvor mange parametre modellen bruger.")
    st.write("Herefter plotter vi for at se hvor godt modellen klarer sig.")
    # Her definerer og træner vi modellen

    if "mlp" not in st.session_state:
        st.session_state.mlp = None
        st.session_state.scaler = None
        st.session_state.eval = None  # store plots/data you need

    def plotting(sand, forudsagt):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(sand, forudsagt, alpha=0.5)
            ax.plot([min(sand), max(sand)], [min(sand), max(sand)], color='red', linestyle='--')
            ax.set_xlabel('Sand værdi')
            ax.set_ylabel('Forudsagt værdi')
            ax.set_title('Sand vs Forudsagt')
            ax.grid()
            fig.tight_layout()
            return fig

    # if st.button("Kør Neuralt Netværk"):
    #     mlp = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(layer_one, layer_two, layer_three, layer_four, layer_five, layer_six), 
    #     max_iter=500, early_stopping=True, random_state=42)
    #     mlp.fit(data_træning_scaled, sand_periode_træning) 

    #     # Her giver vi den trænede model test data som den ikke har set før, og beder om at forudsige dybden
    #     forudsagt_periode = mlp.predict(data_test_scaled)  

    #     # Beregn antal parametre i modellen
    #     # Coef er vægtene er intercept er bias. Den henter antallet direkte fra modellen.
    #     n_params = sum(coef.size + intercept.size for coef, intercept in zip(mlp.coefs_, mlp.intercepts_))
    #     #Print info omkring træningens forløb
    #     st.write(f"Antal parametre i NN: " + str(n_params) + "--------" + "Antal Iterations:" + str(mlp.n_iter_) + "--------" + "Maximalt antal interations:" +  str(mlp.max_iter))
        
    #     st.write("Evaluer modellens performance ved at plotte modellens forudsagte perioder mod de sande perioder.")
    #     #Make a simple true vs predicted plot
        
    #     plotting(sand_periode_test, forudsagt_periode)
    #     fig = plt.gcf()
    #     #Scale down the current st plot
    #     fig.set_size_inches(4 , 4)
    #     st.pyplot(fig, use_container_width=False)

    #     #Feature importance
    #     st.write("Vi undersøger hvilke parametre der har størst betydning for modellens forudsigelse ved at udføre permutation importance.")
    #     import warnings
    #     warnings.filterwarnings("ignore", message="X does not have valid feature names, but .* was fitted with feature names")

    #     res = sklearn.inspection.permutation_importance(mlp, data_test_scaled, sand_periode_test, scoring="neg_mean_squared_error")

    #     imp_mse = res.importances_mean                
    #     order = np.argsort(imp_mse)[::-1]
    #     labels = np.asarray(variabler[:-1])[order]
    #     vals = imp_mse[order]

    #     fig, ax = plt.subplots(figsize=(8, 4))
    #     y = np.arange(len(vals))
    #     ax.barh(y, vals)
    #     ax.set_yticks(y)
    #     ax.set_yticklabels(labels)
    #     ax.set_xlabel("Increase in MSE (permutation)")
    #     ax.set_ylabel("Feature")
    #     ax.set_title("Permutation Importance")
    #     ax.invert_yaxis()  
    #     fig.tight_layout()
    #     st.pyplot(fig, use_container_width=False)

    #     st.subheader("Sammenlign med egne data")
    #     st.write("Lad modellen forudsige perioder ud fra dine egne værdier. ")
    #     st.write("Nedenfor kan du lave en forudsigelse af perioden på dine data og plotte/" \
    #     " dine målinger mod dens forudsigelser (også på dine data). /" \
    #     "Linjen viser hvor data burde ligger hvis de stemte overens (den er ikke lavet ud fra data men bare tegnet oveni). ")

    #     st.write("Vælg om du vil tegne perioden udregnet fra dine data med den tilnærmede formel (small angle approximation).")
    #     # For at sammenligne med beregning med small angle approximation 
    #     # Sæt nedenstående til 1
    #     Vis_Beregning_med_small_angle_approximation = 1
    # --- train button: ONLY trains + saves to session ---
    if st.button("Kør Neuralt Netværk"):
        mlp = sklearn.neural_network.MLPRegressor(
            hidden_layer_sizes=(layer_one, layer_two, layer_three, layer_four, layer_five, layer_six),
            max_iter=500, early_stopping=True, random_state=42
        )
        mlp.fit(data_træning_scaled, sand_periode_træning)
        forudsagt_periode = mlp.predict(data_test_scaled)

        # permutation importance
        res = sklearn.inspection.permutation_importance(
            mlp, data_test_scaled, sand_periode_test, scoring="neg_mean_squared_error"
        )
        imp_mse = res.importances_mean
        order = np.argsort(imp_mse)[::-1]
        labels = np.asarray(variabler[:-1])[order]
        vals = imp_mse[order]

        st.session_state.mlp = mlp
        st.session_state.scaler = scaler
        st.session_state.eval = {
            "sand_periode_test": sand_periode_test,
            "forudsagt_periode": forudsagt_periode,
            "perm_labels": labels,
            "perm_vals": vals,
            "n_params": sum(c.size + b.size for c, b in zip(mlp.coefs_, mlp.intercepts_)),
            "n_iter": mlp.n_iter_,
            "max_iter": mlp.max_iter,
        }

        # --- results section: shows on every rerun once trained ---
    if st.session_state.mlp is not None:
        mlp = st.session_state.mlp
        scaler = st.session_state.scaler
        ev = st.session_state.eval

        st.write(
            f"Antal parametre i NN: {ev['n_params']} -------- "
            f"Antal Iterations: {ev['n_iter']} -------- "
            f"Maximalt antal iterations: {ev['max_iter']}"
        )

        st.write("Evaluer modellens performance:")
        col1, col2 = st.columns(2)
        with col1:
            fig1 = plotting(ev["sand_periode_test"], ev["forudsagt_periode"])
            st.pyplot(fig1, use_container_width=True)

        with col2:
            # Residual histogram
            residuals = ev["sand_periode_test"] - ev["forudsagt_periode"]
            fig3, ax3 = plt.subplots(figsize=(5, 5))
            ax3.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
            ax3.axvline(x=0, color='red', linestyle='--', label='Perfekt forudsigelse')
            ax3.set_xlabel("Sand periode − Forudsagt periode")
            ax3.set_ylabel("Antal")
            ax3.set_title("Histogram over residualer")
            ax3.legend()
            ax3.grid(True)
            fig3.tight_layout()
            st.pyplot(fig3, use_container_width=True)

        #Permutation Importance
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        y = np.arange(len(ev["perm_vals"]))
        ax2.barh(y, ev["perm_vals"])
        ax2.set_yticks(y)
        ax2.set_yticklabels(ev["perm_labels"])
        ax2.set_xlabel("Increase in MSE (permutation)")
        ax2.set_ylabel("Feature")
        ax2.set_title("Permutation Importance")
        ax2.invert_yaxis()
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=False)

        #Cellen tager excel-skabelonen som input.

        #Indsæt din egen fil nedenfor
        uploaded_file = st.file_uploader("Upload din egen Excel fil med data her", type=["xlsx"])
        dataset_path = uploaded_file
        if uploaded_file is None:
            st.info("Upload en Excel-fil for at lave forudsigelser.")
        else:
            Vis_Beregning_med_small_angle_approximation = st.checkbox("Vis beregning med small angle approximation", value=False)
            Vis_Beregning_med_Kompliceret_Ligning = st.checkbox("Vis beregning med kompliceret ligning", value=False)
            df = pd.read_excel(dataset_path, skiprows=2, engine='openpyxl') #Understøtter nyere excelformater

            # Number of features the model expects (no need for matching column names)
            n_features = getattr(mlp, "n_features_in_", None)

            #Del data op i input og periode
            X_new = df.iloc[:, :n_features].apply(pd.to_numeric, errors="coerce").to_numpy()
            målt_periode = pd.to_numeric(df.iloc[:, n_features], errors="coerce").to_numpy()

            #Beregn periode med small angle approximation for målt længde.
            g = 9.81  # m/s^2
            L = X_new[:, 0]
            T_simple = 2 * np.pi * np.sqrt(L / g)

            #Bergen periode med kompliceret ligning
            rho_air = 1.225        # kg/m^3
            C_d = 1.1              # cylinder, transverse flow
            g = 9.81

            L, alpha, theta0, crosssection, m_total = X_new[:, 0], X_new[:, 1], X_new[:, 2], X_new[:, 3], X_new[:, 4]
            shape = ((1/3)*alpha + (1 - alpha)) / ((1/2)*alpha + (1 - alpha))
            T0 = 2 * np.pi * np.sqrt(L / g * shape)
            k2 = np.sin(theta0 / 2)**2
            angle_factor = (2 / np.pi) * ellipk(k2)
            A = crosssection
            omega0 = 2 * np.pi / T0

            gamma = (rho_air * C_d * A * L) / m_total
            drag_factor = 1 + (gamma / omega0)**2 / 8

            T_komp =  T0 * angle_factor * drag_factor

            #Skaler input så vi kan bruge det i vores NN model
            X_new_scaled = scaler.transform(X_new)
            #Forudsig med modellen på vores egen input data 
            forudsagt_periode_egen = mlp.predict(X_new_scaled)

            #Regn usikkerheder
            mae_model = np.mean(np.abs(forudsagt_periode_egen - målt_periode))
            mae_simple = np.mean(np.abs(T_simple - målt_periode))
            mae_komp = np.mean(np.abs(T_komp - målt_periode))

            col1, col2 = st.columns([1.5,1])
            with col1:
                #Plot
                plt.figure(figsize=(6,6))
                plt.scatter(målt_periode, forudsagt_periode_egen, alpha=0.5, label=f"Model (MAE={mae_model:.3f})", color='blue')
                mn = float(np.min([målt_periode.min(), forudsagt_periode_egen.min(), T_simple.min(), T_komp.min()]))
                mx = float(np.max([målt_periode.max(), forudsagt_periode_egen.max(), T_simple.max(), T_komp.max()]))
                plt.ylabel("Forudsagt Periode")

                if Vis_Beregning_med_small_angle_approximation == 1:
                    plt.scatter(målt_periode, T_simple, alpha=0.5, label=f"T=2π√(L/g) (MAE={mae_simple:.3f})", color='orange')
                    plt.ylabel("Forudsagt / Beregnet Periode")

                if Vis_Beregning_med_Kompliceret_Ligning == 1:
                    plt.scatter(målt_periode, T_komp, alpha=0.5, label=f"Kompliceret ligning (MAE={mae_komp:.3f})", color='green')
                    plt.ylabel("Forudsagt / Beregnet Periode")

                plt.plot([mn, mx], [mn, mx], "r--", label="Perfekt overensstemmelse")
                plt.xlabel("Målt Periode")
                plt.title("Målt vs Forudsagt Periode")
                plt.grid(True)
                plt.legend()
                st.pyplot(plt.gcf(), use_container_width=True)

if __name__ == "__main__":
    main()