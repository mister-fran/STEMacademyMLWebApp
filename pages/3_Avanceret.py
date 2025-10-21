import streamlit as st
import pandas as pd
#Load data from dataloader
from utils.data_loader import load_huspriser_dataset, load_diabetes_dataset, load_gletsjer_dataset
import os
from utils.config import DATA_PATHS

st.set_page_config(page_title="Avanceret Niveau", page_icon="🚀")


def main():

    # Load datasets using cached functions
    DS1 = load_huspriser_dataset()
    DS2 = load_diabetes_dataset()
    DS3 = load_gletsjer_dataset()

    # Dataset selection
    st.sidebar.header("Datasæt")
    dataset = st.sidebar.radio("Vælg et datasæt:", ["Huspriser", "Diabetes", "Gletsjer"])

    st.title(f"🚀 Avanceret Niveau")

    # Add description
    st.write("Vælg et datasæt for at begynde.")
    st.write('Under "Avanceret" har du mere mulighed for at lege med modellen, skrue på flere parametre og bedre analysere dens performance. Prøv denne af hvis du føler dig komfortabel med "Standard".')

    st.subheader(f"Visualisering af {dataset}")
    # Display the selected dataset with scrolling enabled and limited to 5 rows tall
    
    if dataset == "Huspriser":
        st.dataframe(DS1, height=200, use_container_width=True)
    elif dataset == "Diabetes":
        st.dataframe(DS2, height=200, use_container_width=True)
    elif dataset == "Gletsjer":
        st.dataframe(DS3, height=200, use_container_width=True)

    # Add a download link for guidance PDF in the sidebar
    pdf_path = 'data/vejledning.pdf'  # Put your PDF file here
    
    st.sidebar.write("") # Add vertical space above button

    # Download button for PDF
    if os.path.exists(DATA_PATHS['Vejledning']):
        try:
            with open(DATA_PATHS['Vejledning'], "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
            
            st.sidebar.download_button(
                label="📥 Hent vejledning",
                data=pdf_bytes,
                file_name="vejledning.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.sidebar.error(f"Fejl ved indlæsning af PDF: {e}")
    else:
        st.sidebar.warning("⚠️ Vejledning PDF ikke fundet.")

    # Content based on dataset - Avanceret level
    if dataset == "Huspriser":
        st.subheader("Avanceret Niveau - Huspriser")
        st.write("Algoritmen forudsiger prisen af et hus ud fra de resterende variable som du kan se øverst på siden.  \n Nedenfor kan du selv vælge hvordan algoritmen skal vurdere om et spørgsmål den stiller er godt eller dårligt (Se forklaring i PDF).")
        
        target_column = st.selectbox("Hvilken variabel vil du forudsige?", DS1.columns, index=len(DS1.columns) - 1)

        # Allow user to select columns for regression (excluding the target column)
        available_columns = [col for col in DS1.columns if col != target_column]
        selected_columns = st.multiselect(
            "Hvilke variable vil du bruge til at forudsige den med?", 
            available_columns, 
            default=available_columns
        )

        # Error metric selection
        error_metric = st.selectbox("Hvordan vurderer modellen hvad der er et godt spørgsmål at stille?", ["Kvadreret fejl", "Absolut fejl"])

        # Button to run the model
        if st.button("Kør regression model"):
            # Ensure at least one column is selected
            if not selected_columns:
                st.error("Vælg mindst én kolonne til regression.")
            else:
                # Prepare data using only selected columns
                X = DS1[selected_columns]
                y = DS1[target_column]

                # Handle missing values
                X = X.fillna(0)
                y = y.fillna(0)

                # Train-test split
                from sklearn.model_selection import train_test_split
                from lightgbm import LGBMRegressor
                from sklearn.metrics import mean_squared_error

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train a regression model using LGBMRegressor with selected objective
                if error_metric == "Kvadreret fejl":
                    model = LGBMRegressor(objective='regression', random_state=42)  # L2 loss (MSE)
                elif error_metric == "Absolut fejl":
                    model = LGBMRegressor(objective='regression_l1', random_state=42)  # L1 loss (MAE)
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

                # Add predictions to the dataset
                predictions_df = X_test.copy()
                predictions_df[target_column] = y_test.values
                predictions_df['Forudsigelse'] = y_pred

                # Display the dataset with predictions
                st.write("Datasæt med forudsigelser:")
                st.dataframe(predictions_df, height=200, use_container_width=True)

                if error_metric == "Kvadreret fejl":
                    error = mean_squared_error(y_test, y_pred)
                    st.write(f"Kvadreret fejl for regression på {target_column}: {error} kr.")
                elif error_metric == "Absolut fejl":
                    from sklearn.metrics import mean_absolute_error
                    error = mean_absolute_error(y_test, y_pred)
                    st.write(f"Absolut fejl for regression på {target_column}: {error} kr.")

                # Rigtig vs forudsagt plot
                st.subheader("Rigtige vs forudsagte værdier værdier")
                import matplotlib.pyplot as plt
                import numpy as np
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_test, y_pred, alpha=0.6, color='blue')
                
                # Perfect prediction line
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfekt foudsigelse')
                
                ax.set_xlabel('Rigtige værdier')
                ax.set_ylabel('Foroudsagte værdier')
                ax.set_title('Rigtige vs forudsagte værdier')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)

                # Histogram of residuals
                st.subheader("Histogram over residualer")
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

    elif dataset == "Diabetes":
        st.subheader("Avanceret Niveau - Diabetes")
        st.write("Nedenfor kan du vælge hvilke variable du vil bruge til at forudsige diabetes. Ikke alle af disse er nødvendigvis lige relevante for at forudsige resultatet.")
        
        # Target column is fixed for diabetes
        target_column = 'Diabetes'

        # Allow user to select columns for classification (excluding the target column)
        available_columns = [col for col in DS2.columns if col != target_column]
        selected_columns = st.multiselect(
            "Vælg kolonner til klassifikation:", 
            available_columns, 
            default=available_columns
        )

        # Button to run the model
        if st.button("Kør klassifikations model"):
            # Ensure at least one column is selected
            if not selected_columns:
                st.error("Vælg mindst én kolonne til klassifikation.")
            else:
                # Prepare data using only selected columns
                X = DS2[selected_columns]
                y = DS2[target_column]

                # Handle missing values
                X = X.fillna(0)
                y = y.fillna(0)

                # Train-test split
                from sklearn.model_selection import train_test_split
                from lightgbm import LGBMClassifier
                from sklearn.metrics import accuracy_score

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train a classification model using LGBMClassifier
                model = LGBMClassifier(random_state=42)
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

                # Add predictions to the dataset
                predictions_df = X_test.copy()
                predictions_df[target_column] = y_test.values
                predictions_df['Forudsigelse'] = y_pred

                # Display the dataset with predictions
                st.write("Datasæt med forudsigelser:")
                st.dataframe(predictions_df, height=200, use_container_width=True)

                # Show accuracy score
                score = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy for klassifikation på {target_column}: {score:.4f}")

                # ROC Curve visualization
                st.subheader("📈 ROC Curve")
                from sklearn.metrics import roc_curve, auc
                import matplotlib.pyplot as plt
    
                
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
                import numpy as np
                
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
                    st.metric("Antal korrekt klassificeret - Ingen diabetes", f"{len(prob_no_diabetes[prob_no_diabetes < 0.5])}")
                with col2:
                    st.metric("Antal korrekt klassificeret - Diabetes", f"{len(prob_diabetes[prob_diabetes >= 0.5])}")
        
    elif dataset == "Gletsjer":
        st.subheader("Avanceret Niveau - Gletsjer")
        st.write("Algoritmen forudsiger dybden af gletsjeren ud fra de resterende variable som du kan se øverst på siden.  \n Nedenfor kan du selv vælge hvordan algoritmen skal vurdere om et spørgsmål den stiller er godt eller dårligt (Se forklaring i PDF).")
        
        target_column = st.selectbox("Hvilken variabel vil du forudsige?", DS3.columns, index=len(DS3.columns) - 1)

        # Allow user to select columns for regression (excluding the target column)
        available_columns = [col for col in DS3.columns if col != target_column]
        selected_columns = st.multiselect(
            "Hvilke variable vil du bruge til at forudsige den med?", 
            available_columns, 
            default=available_columns
        )

        # Error metric selection
        error_metric = st.selectbox("Hvordan vurderer modellen hvad der er et godt spørgsmål at stille?", ["Kvadreret fejl", "Absolut fejl"])

        # Button to run the model
        if st.button("Kør regression model"):
            # Ensure at least one column is selected
            if not selected_columns:
                st.error("Vælg mindst én kolonne til regression.")
            else:
                # Prepare data using only selected columns
                X = DS3[selected_columns]
                y = DS3[target_column]

                # Handle missing values
                X = X.fillna(0)
                y = y.fillna(0)

                # Train-test split
                from sklearn.model_selection import train_test_split
                from lightgbm import LGBMRegressor
                from sklearn.metrics import mean_squared_error

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train a regression model using LGBMRegressor with selected objective
                if error_metric == "Kvadreret fejl":
                    model = LGBMRegressor(objective='regression', random_state=42)  # L2 loss (MSE)
                elif error_metric == "Absolut fejl":
                    model = LGBMRegressor(objective='regression_l1', random_state=42)  # L1 loss (MAE)
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

                # Add predictions to the dataset
                predictions_df = X_test.copy()
                predictions_df[target_column] = y_test.values
                predictions_df['Forudsigelse'] = y_pred

                # Display the dataset with predictions
                st.write("Datasæt med forudsigelser:")
                st.dataframe(predictions_df, height=200, use_container_width=True)

                if error_metric == "Kvadreret fejl":
                    error = mean_squared_error(y_test, y_pred)
                    st.write(f"Kvadreret fejl for regression på {target_column}: {error} kr.")
                elif error_metric == "Absolut fejl":
                    from sklearn.metrics import mean_absolute_error
                    error = mean_absolute_error(y_test, y_pred)
                    st.write(f"Absolut fejl for regression på {target_column}: {error} kr.")

                # Rigtig vs forudsagt plot
                st.subheader("Rigtig vs forudsagt plot")
                import matplotlib.pyplot as plt
                import numpy as np
                
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
           
if __name__ == "__main__":
    main()