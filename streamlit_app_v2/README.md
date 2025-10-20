# Machine Learning - STEM Academy (Multi-Page App)

Dette er en struktureret multi-page Streamlit applikation til at lære machine learning koncepter.

## Struktur

```
streamlit_app/
├── app.py                    # Hovedside og entry point
├── pages/
│   ├── 1_📊_Datasæt.py      # Datasæt exploration 
│   ├── 2_🎯_Standard.py     # Standard niveau ML
│   └── 3_🚀_Avanceret.py    # Avanceret niveau ML
├── utils/
│   ├── __init__.py
│   ├── config.py            # Konfiguration
│   ├── data_loader.py       # Data loading utilities
│   └── ml_utils.py          # Machine learning utilities
├── data/                    # (tom - til fremtidige data filer)
├── requirements.txt         # Python dependencies
└── README.md               # Denne fil
```

## Sådan kører du appen

1. **Installer dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Kør appen:**
   ```bash
   streamlit run app.py
   ```

3. **Navigation:**
   - Appen åbner på hovedsiden (app.py)
   - Brug sidepanelet til venstre for at navigere mellem sider
   - Hver side har sin egen funktionalitet

## Sider

- **🏠 Hjem** (app.py): Oversigt og introduktion
- **📊 Datasæt**: Udforsk og analyser datasæt
- **🎯 Standard**: Grundlæggende machine learning
- **🚀 Avanceret**: Avancerede ML funktioner med mere kontrol

## Features

- **Dataset visualization**: Se og udforsk de forskellige datasæt
- **Machine Learning**: Træn regression modeller
- **Code examples**: Se Python kode for ML algoritmerne
- **Upload functionality**: Upload og arbejd med egne datasæt
- **Feature selection**: Vælg hvilke kolonner der skal bruges (Avanceret niveau)

## Baseret på

Denne strukturerede version er baseret på den originale `Website_0.py` fil, men opdelt i separate sider for bedre organisation og vedligeholdelse.