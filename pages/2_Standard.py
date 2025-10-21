import streamlit as st
import pandas as pd
#Load data from dataloader
from utils.data_loader import load_huspriser_dataset, load_diabetes_dataset, load_gletsjer_dataset
import os
from utils.config import DATA_PATHS
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

    # Content based on dataset - Standard level
    if dataset == "Huspriser":
        st.subheader("Standard Niveau - Huspriser")
        st.write("Algoritmen forudsiger prisen af et hus ud fra de resterende variable som du kan se øverst på siden.  \n Nedenfor kan du selv vælge hvordan algoritmen skal vurdere om et spørgsmål den stiller er godt eller dårligt (Se forklaring i PDF).")

        # Automatically set the target column to the last column in the dataset
        target_column = DS1.columns[-1]

        # Dropdown menu to select error metric
        error_metric = st.selectbox("Hvordan vurderer modellen hvad der er et godt spørgsmål at stille?", ["Kvadreret fejl", "Absolut fejl"])

        # Button to run the model
        if st.button("Kør regression model"):
            if target_column:
                # Prepare data
                X = DS1.drop(columns=[target_column])
                y = DS1[target_column]

                # Handle missing values
                X = X.fillna(0)
                y = y.fillna(0)

                # Import train_test_split before usage
                from sklearn.model_selection import train_test_split
                from lightgbm import LGBMRegressor
                from sklearn.metrics import mean_squared_error

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
X = DS1.drop(columns=['{target_column}'])
y = DS1['{target_column}']

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
            from sklearn.model_selection import train_test_split
            from lightgbm import LGBMClassifier
            from sklearn.metrics import accuracy_score, f1_score

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
                from sklearn.model_selection import train_test_split
                from lightgbm import LGBMRegressor
                from sklearn.metrics import mean_squared_error

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
                    from sklearn.metrics import mean_absolute_error
                    error = mean_absolute_error(y_test, y_pred)
                    st.write(f"Absolut fejl for regression på {target_column}: {error} kr.")

                # True vs Predicted plot
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