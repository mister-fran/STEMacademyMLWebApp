import streamlit as st
import pandas as pd
#Load data from dataloader
from utils.data_loader import load_pendul_dataset_short, load_pendul_dataset_long

import os
from utils.config import DATA_PATHS
#from utils.plots import plotting, plotting_glet, plotting_partikel, Plotting_class, plotting_reg_own, plotting_class_own

#Importer pakker
# Data
import numpy as np
import scipy as scipy

# Plotting
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Sklearn: et librabry med en masse funtioner vi bruger i Machine Learning
import sklearn as sklearn
from sklearn.inspection import PartialDependenceDisplay
from sklearn.pipeline import Pipeline

# LightGBM - pakke til at køre decision tree
import lightgbm as lgb
from lightgbm import early_stopping

#Kompliceret ligning
from scipy.special import ellipk

st.set_page_config(page_title="Standard Niveau")

#Reset session state
PAGE_ID = "pendul"  # change per page
# If we arrived here from another page, reset this page's state
if st.session_state.get("_active_page") != PAGE_ID:
    st.write(st.session_state.get("_active_page"))
    st.session_state.clear()
    st.session_state["_active_page"] = PAGE_ID
    st.rerun()
    
def main():
    st.header("Modul Pendul - Pendulets periode med ML")

    # Load datasets using cached functions
    #DSPendul = load_pendul_dataset()

    # Add description
    st.write('Hvis du aldrig har arbejdet med Machine Learning før er du havnet det rigtige sted. ' \
    'Modul Pendul er en basal introdution til det neurale netværk forklaret i kontekst af det simple pendul. ' \
    'Læs vejledningen (som findes i venstre side) sideløbende med denne side. '\
    'Hvis du aldrig har arbejdet med ML før, er det en god ide at se [denne video](https://www.youtube.com/watch?v=CqOfi41LfDw) sideløbende med vejledningen for at få et bedre overblik over hvordan ML virker.')
    st.write("God arbejdslyst.")    

    
    st.sidebar.write("") # Add vertical space above button

    # Download button for PDF PARTIKEL
    if os.path.exists(DATA_PATHS['VejledningPENDUL']):
        try:
            with open(DATA_PATHS['VejledningPENDUL'], "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
            
            st.sidebar.download_button(
                label="Hent vejledning til Modul Pendul",
                data=pdf_bytes,
                file_name="vejledningPENDUL.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.sidebar.error(f"Fejl ved indlæsning af PDF: {e}")
    else:
        st.sidebar.warning("Vejledning PDF ikke fundet.")
    
    #Download template for xl
    if os.path.exists(DATA_PATHS['TemplatePENDUL']):
        try:
            with open(DATA_PATHS['TemplatePENDUL'], "rb") as excel_file:
                excel_bytes = excel_file.read()
            
            st.sidebar.download_button(
                label="Hent skabelon til datatagning til Modul Pendul",
                data=excel_bytes,
                file_name="Penduldata_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.sidebar.error(f"Fejl ved indlæsning af Excel-fil: {e}")
    else:
        st.sidebar.warning("Fil ikke fundet.")

    #Download for Example dataset
    if os.path.exists(DATA_PATHS['EksempelPENDUL']):
        try:
            with open(DATA_PATHS['EksempelPENDUL'], "rb") as excel_file:
                excel_bytes = excel_file.read()
            
            st.sidebar.download_button(
                label="Hent Eksempel på datasæt til Modul Pendul",
                data=excel_bytes,
                file_name="Eksempel_Data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.sidebar.error(f"Fejl ved indlæsning af Excel-fil: {e}")
    else:
        st.sidebar.warning("Fil ikke fundet.")
    # Pendul

    ## Data
    st.markdown("---")
    st.markdown("##### Modul 1")
    st.markdown("**Træningsdata**")
    st.write("Nedenfor er visualiseret det data vi bruger til at træne vores model på."\
             " Desto mere og bedre data vi har, desto bedre bliver modellen til at forudsige perioder på baggrund af de øvrige parametre. ")
    st.write("Du kan vælge mellem to datasæt. Forskellen på dem er i hvilket interval værdierne for længden ligger i. Dette betyder modellen kun er god til at forudsige perioder for penduler med en længde inden for det givne interval. For intervallet (0.2m - 1.0m) er modellen f.eks. dårlig til at forudsige svingningstid for penduler med længde >1m og vice versa.")
    
    st.selectbox("Vælg datasæt", options=[ "L_measured i intervallet (0.2m - 2m)","L_measured i intervallet: (0.2m - 1m)"], key="datasæt_pendul")

    #Load data
    if st.session_state.datasæt_pendul == "L_measured i intervallet: (0.2m - 1m)":
        data = load_pendul_dataset_short()
    elif st.session_state.datasæt_pendul == "L_measured i intervallet (0.2m - 2m)":
        data = load_pendul_dataset_long()

    #Vis data
    st.dataframe(data, height=200, use_container_width=True)

    #Redefinitions for later use
    variabler = data.columns
    input_variabler = [v for v in variabler if v != 'Period']
    input_data = data[input_variabler].to_numpy()
    truth_data = data['Period'].to_numpy()

    st.markdown("**Træning**")
    #Andel af data
    st.write("Nedenfor kan du vælge hvor stor en andel af data du vil bruge til at træne modellen.")
    # Vælg hvor stor andel. Tallet skal være mellem 0 og 1.0.
    andel_af_data = st.slider("Andel af data til træning", min_value=0.01, max_value=1.0, value=0.05, step=0.01)
    #Vi omdefinerer vores input og truth data til kun at indeholde en del af dataene.
    input_data_justeret, truth_data_justeret = sklearn.utils.resample(
    input_data, truth_data, 
    n_samples=int(andel_af_data * len(input_data)), 
    random_state=42, 
    replace=False
    )

    # Tilrettelæg data til brug i model    
    data_træning, data_test, sand_periode_træning, sand_periode_test = \
        sklearn.model_selection.train_test_split(input_data_justeret, truth_data_justeret, test_size=0.25, random_state=42)
    #Skaler
    scaler = sklearn.preprocessing.StandardScaler()
    data_træning_scaled = scaler.fit_transform(data_træning)
    data_test_scaled = scaler.transform(data_test)
    
    ## Det neurale netværk
    st.write("I et neuralt netværk kan vi justere på hvor mange lag og hvor mange noder hvert lag skal have:")
    
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
    
    
    st.write("Nedenfor træner vi modellen. Der vises hvor mange parametre der er i modellen, samt antallet af interationer den har trænet over. Herefter plotter vi for at se hvor godt modellen klarer sig.")
    # Her definerer og træner vi modellen
    if "mlp" not in st.session_state:
        st.session_state.mlp = None
        st.session_state.scaler = None
        st.session_state.eval = None  # store plots/data you need

    def plotting(sand, forudsagt):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(sand, forudsagt, alpha=0.5)
            #Add MSE as label
            mse = sklearn.metrics.mean_squared_error(sand, forudsagt)
            ax.text(0.05, 0.95, f'MSE: {mse:.3f}', transform=ax.transAxes, verticalalignment='top')
            ax.plot([min(sand), max(sand)], [min(sand), max(sand)], color='red', linestyle='--')
            ax.set_xlabel('Sand værdi')
            ax.set_ylabel('Forudsagt værdi')
            ax.set_title('Sand vs Forudsagt')
            ax.grid()
            fig.tight_layout()
            return fig

    #Add plot of loss vs iterations
    


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
            hidden_layer_sizes=hidden_layer_sizes,
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
            f"Antal Iterations: {ev['n_iter']}"
            #f"Maximalt antal iterations: {ev['max_iter']}"
        )
        #Print a message saying ive reached max iterations
        if ev['n_iter'] >= ev['max_iter']:
            st.warning("Modellen nåede det maksimale antal iterationer (500) før den konvergerede. Juster på modellens parametre for at se om det forbedrer træningen.")

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

        col1, col2 = st.columns(2)
        with col1:
            # Indsæt Loss plot her:
            fig_loss, ax_loss = plt.subplots(figsize=(5, 5))
            ax_loss.plot(mlp.loss_curve_, color='tab:blue', linewidth=2)
            ax_loss.set_xlabel("Iterations")
            ax_loss.set_ylabel("MSE Loss")
            ax_loss.set_title("MSE Loss vs Iterations")
            ax_loss.grid(True)
            st.pyplot(fig_loss, use_container_width=True)

        with col2:
            #Permutation Importance
            fig2, ax2 = plt.subplots(figsize=(5, 5))
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
        st.markdown("---")
        st.markdown("##### Udfør Modul 2 før du går videre")
        st.markdown("---")
        st.markdown("##### Modul 3")

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

            mse_model = np.mean((forudsagt_periode_egen - målt_periode)**2)
            mse_simple = np.mean((T_simple - målt_periode)**2)
            mse_komp = np.mean((T_komp - målt_periode)**2)
            
            col1, col2 = st.columns([1.5,1])
            with col1:
                #Plot
                plt.figure(figsize=(6,6))
                plt.scatter(målt_periode, forudsagt_periode_egen, alpha=0.5, label=f"Model (MSE={mse_model:.3f})", color='blue')
                mn = float(np.min([målt_periode.min(), forudsagt_periode_egen.min(), T_simple.min(), T_komp.min()]))
                mx = float(np.max([målt_periode.max(), forudsagt_periode_egen.max(), T_simple.max(), T_komp.max()]))
                plt.ylabel("Forudsagt Periode")

                if Vis_Beregning_med_small_angle_approximation == 1:
                    plt.scatter(målt_periode, T_simple, alpha=0.5, label=f"T=2π√(L/g) (MSE={mse_simple:.3f})", color='orange')
                    plt.ylabel("Forudsagt / Beregnet Periode")

                if Vis_Beregning_med_Kompliceret_Ligning == 1:
                    plt.scatter(målt_periode, T_komp, alpha=0.5, label=f"Kompliceret ligning (MSE={mse_komp:.3f})", color='green')
                    plt.ylabel("Forudsagt / Beregnet Periode")

                plt.plot([mn, mx], [mn, mx], "r--", label="Perfekt overensstemmelse")
                plt.xlabel("Målt Periode")
                plt.title("Målt vs Forudsagt Periode")
                plt.grid(True)
                plt.legend()
                st.pyplot(plt.gcf(), use_container_width=True)

        if "PDP" not in st.session_state:
                st.session_state.PDP = None
                
        # Create the button
        if st.button("Generer Partial Dependence Plot"):
    
            # 1. Saml den allerede trænede scaler og model i en pipeline
            pipeline = Pipeline([
                ('scaler', scaler),
                ('mlp', mlp)
            ])
            features_to_plot = list(range(len(input_variabler)))
            # 2. Generer Partial Dependence Plots
            fig, ax = plt.subplots(figsize=(12, 8))
            display = PartialDependenceDisplay.from_estimator(
                estimator=pipeline,       
                X=data_test,          
                features=features_to_plot,
                feature_names=input_variabler,
                kind="both",
                grid_resolution=50,
                ax=ax,
                subsample=1000,
                pd_line_kw={"color":"tab:blue","linestyle":"-","linewidth":2,"label":"."},
                ice_lines_kw = {"color": "lightblue", "alpha": 0.12, "linewidth": 0.5,"label":"."}
            )
            for ax in display.axes_.flat:
                # ax kan være None hvis grid'et ikke er fuldt udfyldt (f.eks. 5 plots i et 2x3 grid)
                if ax is not None:
                    legend = ax.get_legend()
                    if legend is not None:
                        legend.remove()
            # Create custom legend items
            ice_line = mlines.Line2D([], [], color='tab:blue', alpha=1,linewidth=2, label='Individuelle Partial Dependence')
            pdp_line = mlines.Line2D([], [], color='lightblue',alpha=1, linewidth=1, label='Gennemsnitlig Partial Dependence')
            # Place a single legend on the figure itself
            fig.legend(handles=[ice_line, pdp_line], loc='lower right', bbox_to_anchor=(0.98, 0.1), ncol=1, fontsize=12)
            # Adjust layout so subplots don't overlap with super title and bottom legend isn't cut off
            fig.subplots_adjust(top=0.92, bottom=0.12, wspace=0.3, hspace=0.4)

            st.session_state.PDP = fig
            
        if st.session_state.PDP is not None:
            st.pyplot(st.session_state.PDP, use_container_width=True)
        
if __name__ == "__main__":
    main()