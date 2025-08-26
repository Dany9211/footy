import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib # Per salvare e caricare il modello

st.set_page_config(page_title="Analisi Campionati Next Gol e stats live", layout="wide")
st.title("Analisi Tabella 23agosto2023 - con Predizioni Gol!")

# --- Funzione per il caricamento del file CSV (come nel tuo codice) ---
@st.cache_data
def load_data(uploaded_file):
    """
    Carica i dati da un file CSV caricato dall'utente.
    Tenta diverse strategie di parsing per gestire potenziali errori.
    """
    if uploaded_file is not None:
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8', on_bad_lines='skip', header=0)
            if not df.empty and len(df.columns) > 1:
                st.success(f"File CSV caricato con successo (delimitatore ';', codifica utf-8). Colonne: {df.columns.tolist()}")
                return df
            uploaded_file.seek(0)
        except Exception as e:
            st.error(f"Errore di caricamento (';', utf-8): {e}. Tentativo successivo...")
            uploaded_file.seek(0)

        try:
            df = pd.read_csv(uploaded_file, sep=';', encoding='latin1', on_bad_lines='skip', header=0)
            if not df.empty and len(df.columns) > 1:
                st.success(f"File CSV caricato con successo (delimitatore ';', codifica latin1). Colonne: {df.columns.tolist()}")
                return df
            uploaded_file.seek(0)
        except Exception as e:
            st.error(f"Errore di caricamento (';', latin1): {e}. Tentativo successivo...")
            uploaded_file.seek(0)
        
        try:
            df = pd.read_csv(uploaded_file, sep=',', encoding='utf-8', on_bad_lines='skip', engine='python', header=0)
            if not df.empty and len(df.columns) > 1:
                st.success(f"File CSV caricato con successo (delimitatore ',', codifica utf-8, motore python). Colonne: {df.columns.tolist()}")
                return df
            uploaded_file.seek(0)
        except Exception as e:
            st.error(f"Errore di caricamento (',', utf-8, python engine): {e}. Tentativo successivo...")
            uploaded_file.seek(0)

        try:
            df = pd.read_csv(uploaded_file, sep=None, engine='python', on_bad_lines='skip', header=0)
            if not df.empty and len(df.columns) > 1:
                st.success(f"File CSV caricato con successo (delimitatore auto-rilevato, motore python). Colonne: {df.columns.tolist()}")
                return df
            uploaded_file.seek(0)
        except Exception as e:
            st.error(f"Errore di caricamento (auto-delimitatore, python engine): {e}. Tentativo successivo...")
            uploaded_file.seek(0)
            
        st.error("Impossibile leggere il file CSV con le strategie di parsing automatiche. Controlla attentamente il formato del file, il delimitatore (punto e virgola, virgola o altro) e la codifica.")
        return pd.DataFrame()
    return pd.DataFrame()

# --- Funzione per convertire stringhe con virgola in float (come nel tuo codice) ---
def convert_to_float(series):
    return pd.to_numeric(series.astype(str).str.replace(",", "."), errors="coerce")

# --- Caricamento dati iniziali tramite file upload ---
uploaded_file = st.sidebar.file_uploader("Carica il tuo file CSV", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df.empty:
        st.error("Il DataFrame caricato dal file è vuoto o c'è stato un errore di lettura. Controlla il formato del tuo CSV.")
        st.stop()
    st.write(f"**Righe iniziali nel dataset:** {len(df)}")
    st.write(f"**Colonne caricate:** {df.columns.tolist()}")
else:
    st.info("Per iniziare, carica un file CSV dal tuo computer.")
    st.stop()

# --- Pre-elaborazione e pulizia dati (come nel tuo codice) ---
if 'Data' in df.columns:
    df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y %H:%M', errors='coerce')
    df = df.dropna(subset=['Data'])
else:
    st.error("Colonna 'Data' non trovata. Assicurati che il nome della colonna sia corretto (sensibile alle maiuscole).")

if 'Anno' in df.columns:
    df['Anno'] = pd.to_numeric(df['Anno'], errors='coerce')
    df = df.dropna(subset=['Anno'])
else:
    st.error("Colonna 'Anno' non trovata. Assicurati che il nome della colonna sia corretto (sensibile alle maiuscole).")

all_numeric_cols_with_comma = [
    "Odd_Home", "Odd_Draw", "Odd__Away", "Odd_Over_0.5", "Odd_over_1.5", 
    "Odd_over_2.5", "Odd_Over_3.5", "Odd_Over_4.5", "Odd_Under_0.5", 
    "Odd_Under_1.5", "Odd_Under_2.5", "Odd_Under_3.5", "Odd_Under_4.5",
    "elohomeo", "eloawayo", "formah", "formaa", "suth", "suth1", "suth2",
    "suta", "suta1", "suta2", "sutht", "sutht1", "sutht2", "sutat", "sutat1", "sutat2",
    "corh", "corh1", "corh2", "cora", "cora1", "cora2", "yellowh", "yellowh1", "yellowh2",
    "yellowa", "yellowa1", "yellowa2", "ballph", "ballph1", "ballph2", "ballpa", "ballpa1", "ballpa2"
]

for col in all_numeric_cols_with_comma:
    if col in df.columns:
        df[col] = convert_to_float(df[col])

other_int_cols = ["Gol_Home_FT", "Gol_Away_FT", "Gol_Home_HT", "Gol_Away_HT", 
                  "Home_Pos_Tot", "Away_Pos_Tot", "Home_Pos_H", "Away_Pos_A", "Giornata", "BTTS_SI"]

for col in other_int_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

# --- Aggiunta colonne risultato_ft e risultato_ht (come nel tuo codice) ---
if "Gol_Home_FT" in df.columns and "Gol_Away_FT" in df.columns:
    df["risultato_ft"] = df["Gol_Home_FT"].astype(str) + "-" + df["Gol_Away_FT"].astype(str)
if "Gol_Home_HT" in df.columns and "Gol_Away_HT" in df.columns:
    df["risultato_ht"] = df["Gol_Home_HT"].astype(str) + "-" + df["Gol_Away_HT"].astype(str)

# --- INIZIO: Nuove funzioni e logica per la predizione del gol ---

# Funzione per creare la variabile target e le feature
def create_prediction_features_and_target(df_input):
    df_pred = df_input.copy()

    # Creiamo la variabile target: 'gol_nel_prossimo_15_min'
    # Questo è un esempio semplificato. In un caso reale, avresti bisogno di dati a intervalli di tempo.
    # Per questo esempio, simuliamo l'idea che un gol "imminente" sia legato a partite con molti gol FT.
    # In un'applicazione live, questa logica dovrebbe basarsi sui minutaggi dei gol effettivi.
    
    # Esempio Semplificato: Se ci sono più di 3 gol totali nella partita FT, assumiamo un'alta probabilità
    # che un gol fosse imminente ad un certo punto. Questo è SOLO un placeholder.
    # Per una vera predizione, avresti bisogno di un dataset che abbia lo stato della partita ad ogni minuto.
    # Supponiamo che la colonna "Gol_Home_FT" e "Gol_Away_FT" esistano e siano numeriche.
    
    if "Gol_Home_FT" in df_pred.columns and "Gol_Away_FT" in df_pred.columns:
        df_pred['total_goals_ft'] = df_pred['Gol_Home_FT'] + df_pred['Gol_Away_FT']
        df_pred['gol_nel_prossimo_15_min'] = (df_pred['total_goals_ft'] > 2).astype(int) # Esempio di target
        
        # Se 'Minutaggio_Gol_Home' e 'Minutaggio_gol_Away' sono disponibili e formattati come "min1;min2",
        # potremmo creare un target più realistico.
        # Per semplicità, in questo primo passo, useremo un target basato sul totale FT.
        # Per un target più avanzato, dovresti:
        # 1. Espandere il dataset per avere una riga per ogni minuto di ogni partita.
        # 2. Per ogni minuto, calcolare se un gol è stato segnato nei successivi X minuti.
        # Questo è un task complesso e va oltre una prima implementazione immediata, ma è la direzione giusta.
    else:
        st.warning("Colonne 'Gol_Home_FT' o 'Gol_Away_FT' mancanti, impossibile creare variabile target per la predizione.")
        return pd.DataFrame(), None # Ritorna DataFrame vuoto e target None

    # Seleziona le feature (variabili di input per il modello)
    # È fondamentale che queste feature siano disponibili *prima* che l'evento avvenga
    # Per una predizione "live", dovresti avere i valori di queste feature al minuto X.
    # Qui usiamo le quote pre-partita come proxy per la "situazione di partenza".
    features_cols = [
        "Odd_Home", "Odd_Draw", "Odd__Away", "Odd_Over_0.5", "Odd_over_1.5", 
        "Odd_over_2.5", "Odd_Over_3.5", "Odd_Under_0.5", "Odd_Under_1.5", 
        "Odd_Under_2.5", "BTTS_SI", "elohomeo", "eloawayo", "formah", "formaa"
    ]
    
    # Rimuoviamo le feature che non sono presenti nel DataFrame
    available_features = [col for col in features_cols if col in df_pred.columns]
    
    # Assicurati che le feature siano numeriche e non abbiano NaN
    for col in available_features:
        df_pred[col] = pd.to_numeric(df_pred[col], errors='coerce')
    
    # Riempiamo i valori mancanti con la media per semplicità.
    # Una strategia migliore potrebbe essere l'imputazione più sofisticata o la rimozione delle righe.
    df_pred = df_pred.fillna(df_pred.mean(numeric_only=True))

    X = df_pred[available_features]
    y = df_pred['gol_nel_prossimo_15_min']
    
    # Assicurati che X e y non siano vuoti dopo il cleaning
    if X.empty or y.empty:
        st.warning("DataFrame feature o target vuoto dopo la preparazione per la predizione.")
        return pd.DataFrame(), None

    return X, y, df_pred # Restituisci anche il DataFrame completo con le feature create

# Funzione per addestrare il modello
@st.cache_resource # Cache il modello per non addestrarlo ogni volta
def train_goal_prediction_model(X, y, model_type="Logistic Regression"):
    st.write(f"Addestramento del modello {model_type}...")
    
    # Suddivisione dei dati in training e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    if model_type == "Logistic Regression":
        model = LogisticRegression(random_state=42, solver='liblinear')
    elif model_type == "Random Forest":
        model = RandomForestClassifier(random_state=42, n_estimators=100)
    else:
        st.error("Tipo di modello non riconosciuto.")
        return None, None, None, None

    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probabilità della classe positiva (gol)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    st.success("Modello addestrato con successo!")
    st.write(f"**Metriche di valutazione del modello {model_type}:**")
    st.write(f"- Accuratezza: {accuracy:.2f}")
    st.write(f"- Precisione: {precision:.2f}")
    st.write(f"- Richiamo (Recall): {recall:.2f}")
    st.write(f"- F1-Score: {f1:.2f}")
    st.write(f"- AUC-ROC: {roc_auc:.2f}")
    
    st.markdown("---")
    st.markdown("📈 *Interpretazione delle metriche:*")
    st.markdown("- **Accuratezza**: Percentuale di previsioni corrette totali.")
    st.markdown("- **Precisione**: Dei gol previsti, quanti sono stati effettivamente gol. (Importante per non dare falsi allarmi)")
    st.markdown("- **Richiamo (Recall)**: Dei gol effettivi, quanti sono stati previsti. (Importante per non perdere opportunità)")
    st.markdown("- **F1-Score**: Media armonica di precisione e richiamo.")
    st.markdown("- **AUC-ROC**: Misura la capacità del modello di distinguere tra le classi.")
    st.markdown("---")

    return model, X_train.columns.tolist(), X_test, y_test # Ritorna anche le colonne usate e il set di test per la visualizzazione

# --- FINE: Nuove funzioni e logica per la predizione del gol ---

# Il resto del tuo codice Streamlit continua qui...
filters = {}

# --- FILTRI INIZIALI (come nel tuo codice) ---
st.sidebar.header("Filtri Dati")

if "League" in df.columns:
    leagues = ["Tutte"] + sorted(df["League"].dropna().unique())
    selected_league = st.sidebar.selectbox("Seleziona Campionato", leagues)
    if selected_league != "Tutte":
        filters["League"] = selected_league
    
    if selected_league != "Tutte":
        filtered_teams_df = df[df["League"] == selected_league]
    else:
        filtered_teams_df = df.copy()
else:
    filtered_teams_df = df.copy()
    selected_league = "Tutte"
    st.sidebar.error("Colonna 'League' non trovata. Il filtro per campionato non sarà disponibile.")

if "Anno" in df.columns:
    df_anni_numeric = df["Anno"].dropna()
    if not df_anni_numeric.empty:
        all_unique_years = sorted(df_anni_numeric.unique().astype(int), reverse=True)
        dynamic_range_options_labels = []
        dynamic_range_options_labels.append("Anno Corrente")
        for num_years in range(2, 11):
            label = f"Ultimi {num_years} anni"
            dynamic_range_options_labels.append(label)

        display_options = ["Tutti"] + dynamic_range_options_labels + [str(y) for y in all_unique_years]
        selected_anno_display = st.sidebar.selectbox("Seleziona Anno", display_options)

        if selected_anno_display == "Tutti":
            if "Anno" in filters:
                del filters["Anno"]
        elif selected_anno_display == "Anno Corrente":
            current_calendar_year = datetime.datetime.now().year
            if current_calendar_year in all_unique_years:
                filters["Anno"] = current_calendar_year
            else:
                st.sidebar.info(f"Nessun dato disponibile per l'Anno Corrente ({current_calendar_year}) nel dataset caricato.")
                if "Anno" in filters:
                    del filters["Anno"]
        elif selected_anno_display.startswith("Ultimi"):
            num_years_back = int(selected_anno_display.split(' ')[1])
            if len(all_unique_years) >= num_years_back:
                years_to_filter = all_unique_years[:num_years_back]
                min_year_to_filter = min(years_to_filter)
                max_year_to_filter = max(years_to_filter)
                filters["Anno"] = (min_year_to_filter, max_year_to_filter)
            elif all_unique_years:
                min_year_to_filter = min(all_unique_years)
                max_year_to_filter = max(all_unique_years)
                filters["Anno"] = (min_year_to_filter, max_year_to_filter)
                st.sidebar.info(f"Il dataset contiene solo {len(all_unique_years)} anni. Verranno utilizzati tutti gli anni disponibili per '{selected_anno_display}'.")
            else:
                st.sidebar.info(f"Nessun dato disponibile per '{selected_anno_display}' nel dataset caricato.")
                if "Anno" in filters:
                    del filters["Anno"]
        else:
            try:
                selected_year_int = int(selected_anno_display)
                filters["Anno"] = selected_year_int
            except ValueError:
                st.sidebar.error(f"Valore anno non valido: {selected_anno_display}. Ignorato.")
                if "Anno" in filters:
                    del filters["Anno"]
    else:
        st.sidebar.info("Nessun anno valido trovato nella colonna 'Anno'.")
else:
    st.sidebar.error("Colonna 'Anno' non trovata. Il filtro per anno non sarà disponibile.")


if "Giornata" in df.columns:
    giornata_min = int(df["Giornata"].min()) if not df["Giornata"].isnull().all() else 1
    giornata_max = int(df["Giornata"].max()) if not df["Giornata"].isnull().all() else 38
    giornata_range = st.sidebar.slider(
        "Seleziona Giornata",
        min_value=giornata_min,
        max_value=giornata_max,
        value=(giornata_min, giornata_max)
    )
    filters["Giornata"] = giornata_range
else:
    st.sidebar.error("Colonna 'Giornata' non trovata. Il filtro per giornata non sarà disponibile.")

if "Home_Team" in filtered_teams_df.columns:
    home_teams = ["Tutte"] + sorted(filtered_teams_df["Home_Team"].dropna().unique())
    selected_home = st.sidebar.selectbox("Seleziona Squadra Home", home_teams)
    if selected_home != "Tutte":
        filters["Home_Team"] = selected_home
else:
    st.sidebar.error("Colonna 'Home_Team' non trovata. Il filtro per squadra home non sarà disponibile.")

if "Away_Team" in filtered_teams_df.columns:
    away_teams = ["Tutte"] + sorted(filtered_teams_df["Away_Team"].dropna().unique())
    selected_away = st.sidebar.selectbox("Seleziona Squadra Away", away_teams)
    if selected_away != "Tutte":
        filters["Away_Team"] = selected_away
else:
    st.sidebar.error("Colonna 'Away_Team' non trovata. Il filtro per squadra away non sarà disponibile.")

if "risultato_ht" in df.columns:
    ht_results = sorted(df["risultato_ht"].dropna().unique())
    selected_ht_results = st.sidebar.multiselect("Seleziona Risultato HT", ht_results, default=None)
    if selected_ht_results:
        filters["risultato_ht"] = selected_ht_results
else:
    st.sidebar.error("Colonna 'risultato_ht' non trovata. Il filtro per risultato HT non sarà disponibile.")

def add_range_filter(col_name, label=None):
    if col_name in df.columns:
        numeric_col_series = convert_to_float(df[col_name])
        if not numeric_col_series.isnull().all():
            col_min = float(numeric_col_series.min(skipna=True))
            col_max = float(numeric_col_series.max(skipna=True))
            
            st.sidebar.write(f"Range attuale {label or col_name}: {col_min:.2f} - {col_max:.2f}")
            min_val_input = st.sidebar.text_input(f"Min {label or col_name}", key=f"min_{col_name}", value="")
            max_val_input = st.sidebar.text_input(f"Max {label or col_name}", key=f"max_{col_name}", value="")
            
            if min_val_input.strip() != "" and max_val_input.strip() != "":
                try:
                    filters[col_name] = (float(min_val_input), float(max_val_input))
                except ValueError:
                    st.sidebar.error(f"Valori non validi per {label or col_name}. Inserisci numeri.")
                    if col_name in filters:
                        del filters[col_name]
            else:
                if col_name in filters:
                    del filters[col_name]
        else:
            st.sidebar.info(f"Colonna '{label or col_name}' non contiene valori numerici validi per il filtro.")
            if col_name in filters:
                del filters[col_name]
    else:
        st.sidebar.error(f"Colonna '{label or col_name}' non trovata per il filtro.")
        if col_name in filters:
            del filters[col_name]

st.sidebar.header("Filtri Quote")
for col in ["Odd_Home", "Odd_Draw", "Odd__Away"]:
    add_range_filter(col)
for col in ["Odd_Over_0.5", "Odd_over_1.5", "Odd_over_2.5", "Odd_Over_3.5", "Odd_Over_4.5",
            "Odd_Under_0.5", "Odd_Under_1.5", "Odd_Under_2.5", "Odd_Under_3.5", "Odd_Under_4.5", "BTTS_SI"]:
    add_range_filter(col)


filtered_df = df.copy()
for col, val in filters.items():
    if col in ["Odd_Home", "Odd_Draw", "Odd__Away", "Odd_Over_0.5", "Odd_over_1.5", 
                "Odd_over_2.5", "Odd_Over_3.5", "Odd_Over_4.5", "Odd_Under_0.5", 
                "Odd_Under_1.5", "Odd_Under_2.5", "Odd_Under_3.5", "Odd_Under_4.5", 
                "BTTS_SI", "Giornata"]:
        
        if not isinstance(val, tuple) or len(val) != 2:
            st.error(f"Errore: il valore del filtro per la colonna '{col}' ({val}) non è un intervallo numerico valido. Ignoro il filtro.")
            continue

        series_to_filter = convert_to_float(filtered_df[col])
        try:
            lower_bound = float(val[0])
            upper_bound = float(val[1])
        except (ValueError, TypeError) as e:
            st.error(f"Errore: i valori del filtro per la colonna '{col}' ({val[0]}, {val[1]}) non sono convertibili in numeri. Dettagli: {e}. Ignoro il filtro.")
            continue

        mask = series_to_filter.between(lower_bound, upper_bound)
        filtered_df = filtered_df[mask.fillna(True)]
    elif col == "risultato_ht":
        if isinstance(val, list):
            filtered_df = filtered_df[filtered_df[col].isin(val)]
        else:
            st.error(f"Errore: il valore del filtro per la colonna '{col}' non è una lista come previsto. Ignoro il filtro.")
            continue
    elif col == "Anno":
        if isinstance(val, tuple) and len(val) == 2:
            lower_bound, upper_bound = val
            series_to_filter = pd.to_numeric(filtered_df[col], errors='coerce')
            mask = series_to_filter.between(lower_bound, upper_bound)
            filtered_df = filtered_df[mask.fillna(True)]
        else:
            filtered_df = filtered_df[filtered_df[col] == val]
    else:
        filtered_df = filtered_df[filtered_df[col] == val]

st.subheader("Dati Filtrati")
st.write(f"**Righe visualizzate:** {len(filtered_df)}")

st.markdown("---")
st.subheader("Riepilogo partite per Anno")
if not filtered_df.empty and "Anno" in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df["Anno"]):
    partite_per_anno = filtered_df["Anno"].value_counts().sort_index()
    riepilogo_df = pd.DataFrame(partite_per_anno).reset_index()
    riepilogo_df.columns = ["Anno", "Partite Trovate"]
    st.table(riepilogo_df)
else:
    st.info("Nessuna partita trovata o la colonna 'Anno' non è disponibile/numerica nel dataset filtrato.")
st.markdown("---")

st.dataframe(filtered_df.head(50))


# --- SEZIONE AGGIUNTA: Previsione Gol (con interazione UI) ---
st.header("7. Predizione Gol")
st.markdown("Qui puoi addestrare un modello per prevedere la probabilità di un gol imminente basandosi sui dati filtrati.")
st.markdown("⚠️ **Nota:** Per questa implementazione immediata, la variabile target `gol_nel_prossimo_15_min` è un proxy semplificato (basato su `total_goals_ft > 2`). Per una vera predizione 'live', sarebbe necessario un dataset con lo stato della partita ad ogni minuto.")

# Prepara i dati per la predizione
X_pred, y_pred, df_with_features = create_prediction_features_and_target(df) # Usa il DataFrame originale 'df' per l'addestramento
if not X_pred.empty and y_pred is not None:
    model_choice = st.selectbox(
        "Scegli il tipo di modello per la predizione del gol",
        ["Logistic Regression", "Random Forest"]
    )

    if st.button("Addestra Modello di Predizione Gol"):
        trained_model, feature_names, X_test_for_display, y_test_for_display = train_goal_prediction_model(X_pred, y_pred, model_choice)
        
        if trained_model:
            st.session_state['trained_model'] = trained_model
            st.session_state['model_features'] = feature_names
            st.session_state['X_test_for_display'] = X_test_for_display
            st.session_state['y_test_for_display'] = y_test_for_display
            
            st.success("Modello pronto per fare previsioni!")

    if 'trained_model' in st.session_state:
        st.subheader("Fai una Previsione su una Partita:")
        st.info("Utilizzeremo la prima partita nel tuo dataset filtrato come esempio per la previsione.")
        
        if not filtered_df.empty:
            sample_match = filtered_df.iloc[[0]] # Prendi la prima riga del DF filtrato
            
            # Prepara le feature per la singola partita (deve avere le stesse colonne del training)
            sample_X, _, _ = create_prediction_features_and_target(sample_match)
            
            if not sample_X.empty:
                # Assicurati che le colonne siano nello stesso ordine di quelle usate per l'addestramento
                # E che abbia tutte le feature richieste dal modello.
                # Se mancano feature in sample_X, riempile con la media o 0 come fatto nel training.
                for col in st.session_state['model_features']:
                    if col not in sample_X.columns:
                        # Qui è essenziale che i valori mancanti siano gestiti in modo consistente
                        # con l'addestramento. Per semplicità, usiamo la media del training set.
                        if col in X_pred.columns:
                            sample_X[col] = X_pred[col].mean()
                        else:
                            sample_X[col] = 0 # Fallback se la colonna non era nemmeno nel dataset originale
                sample_X = sample_X[st.session_state['model_features']] # Riordina e seleziona le colonne

                st.write("**Dettagli della partita di esempio (prima riga filtrata):**")
                st.dataframe(sample_match[['Home_Team', 'Away_Team', 'League', 'Anno', 'Odd_Home', 'Odd_Draw', 'Odd__Away', 'Odd_over_2.5']].head())
                
                prediction = st.session_state['trained_model'].predict(sample_X)
                prediction_proba = st.session_state['trained_model'].predict_proba(sample_X)[:, 1]

                st.write(f"**Previsione del modello:**")
                if prediction[0] == 1:
                    st.success(f"È previsto un gol imminente in questa partita con una probabilità del **{prediction_proba[0]*100:.2f}%**.")
                else:
                    st.warning(f"Non è previsto un gol imminente in questa partita (probabilità: {prediction_proba[0]*100:.2f}%).")
            else:
                st.warning("Impossibile preparare le feature per la partita di esempio. Controlla il formato dei dati.")
        else:
            st.warning("Il DataFrame filtrato è vuoto, non ci sono partite di esempio per la previsione.")
else:
    st.info("Per iniziare la predizione, carica un file CSV e assicurati che contenga le colonne 'Gol_Home_FT' e 'Gol_Away_FT'.")

st.markdown("---")
st.markdown("---")

# Il resto delle tue funzioni di analisi (come calcola_first_to_score, calcola_double_chance, etc.) vanno qui sotto
# ...
# --- Funzione per calcolare le probabilità di Vittoria/Sconfitta dopo il primo gol ---
def calcola_first_to_score_outcome(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    
    risultati = {
        "Casa Segna Primo e Vince": 0,
        "Casa Segna Primo e Non Vince": 0,
        "Trasferta Segna Prima e Vince": 0,
        "Trasferta Segna Prima e Non Vince": 0,
        "Nessun Gol": 0
    }
    
    total_matches = len(df_to_analyze)

    for _, row in df_to_analyze.iterrows():
        gol_home_str = str(row.get("Minutaggio_Gol_Home", ""))
        gol_away_str = str(row.get("Minutaggio_gol_Away", ""))

        gol_home = [int(x) for x in gol_home_str.split(";") if x.isdigit()]
        gol_away = [int(x) for x in gol_away_str.split(";") if x.isdigit()]

        min_home_goal = min(gol_home) if gol_home else float('inf')
        min_away_goal = min(gol_away) if gol_away else float('inf')
        
        home_vince = row["Gol_Home_FT"] > row["Gol_Away_FT"]
        away_vince = row["Gol_Away_FT"] > row["Gol_Home_FT"]
        
        if min_home_goal < min_away_goal:
            if home_vince:
                risultati["Casa Segna Primo e Vince"] += 1
            else:
                risultati["Casa Segna Primo e Non Vince"] += 1
        elif min_away_goal < min_home_goal:
            if away_vince:
                risultati["Trasferta Segna Prima e Vince"] += 1
            else:
                risultati["Trasferta Segna Prima e Non Vince"] += 1
        else:
            risultati["Nessun Gol"] += 1

    stats = []
    for esito, count in risultati.items():
        perc = round((count / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats.append((esito, count, perc, odd_min))
    
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

# --- Nuova funzione per analizzare l'esito del secondo gol dopo il primo ---
def calcola_first_to_score_next_goal_outcome(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    
    risultati = {
        "Casa Segna Primo e Segna di Nuovo": 0,
        "Casa Segna Primo e Subisce Gol": 0,
        "Trasferta Segna Prima e Segna di Nuovo": 0,
        "Trasferta Segna Prima e Subisce Gol": 0,
        "Solo un gol o nessuno": 0
    }
    
    total_matches = len(df_to_analyze)

    for _, row in df_to_analyze.iterrows():
        gol_home_str = str(row.get("Minutaggio_Gol_Home", ""))
        gol_away_str = str(row.get("Minutaggio_gol_Away", ""))

        gol_home = sorted([int(x) for x in gol_home_str.split(";") if x.isdigit()])
        gol_away = sorted([int(x) for x in gol_away_str.split(";") if x.isdigit()])

        all_goals = []
        if gol_home:
            all_goals.extend([ (t, 'home') for t in gol_home ])
        if gol_away:
            all_goals.extend([ (t, 'away') for t in gol_away ])
        
        if len(all_goals) < 2:
            risultati["Solo un gol o nessuno"] += 1
            continue
            
        all_goals.sort()
        
        first_goal = all_goals[0]
        second_goal = all_goals[1]
        
        first_scorer = first_goal[1]
        second_scorer = second_goal[1]
        
        if first_scorer == 'home':
            if second_scorer == 'home':
                risultati["Casa Segna Primo e Segna di Nuovo"] += 1
            else:
                risultati["Casa Segna Primo e Subisce Gol"] += 1
        elif first_scorer == 'away':
            if second_scorer == 'away':
                risultati["Trasferta Segna Prima e Segna di Nuovo"] += 1
            else:
                risultati["Trasferta Segna Prima e Subisce Gol"] += 1

    stats = []
    for esito, count in risultati.items():
        perc = round((count / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats.append((esito, count, perc, odd_min))
    
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

# --- Funzione per calcolare i mercati di Doppia Chance ---
def calcola_double_chance(df_to_analyze, period):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
    
    df_double_chance = df_to_analyze.copy()
    
    if period == 'ft':
        df_double_chance["gol_home"] = df_double_chance["Gol_Home_FT"]
        df_double_chance["gol_away"] = df_double_chance["Gol_Away_FT"]
    elif period == 'ht':
        df_double_chance["gol_home"] = df_double_chance["Gol_Home_HT"]
        df_double_chance["gol_away"] = df_double_chance["Gol_Away_HT"]
    elif period == 'sh':
        df_double_chance["gol_home_sh"] = df_double_chance["Gol_Home_FT"] - df_double_chance["Gol_Home_HT"]
        df_double_chance["gol_away_sh"] = df_double_chance["Gol_Away_FT"] - df_double_chance["Gol_Away_HT"]
        df_double_chance["gol_home"] = df_double_chance["gol_home_sh"]
        df_double_chance["gol_away"] = df_double_chance["gol_away_sh"]
    else:
        st.error("Periodo non valido per il calcolo della doppia chance.")
        return pd.DataFrame()
        
    total_matches = len(df_double_chance)
    
    # 1X (Home Win or Draw)
    count_1x = ((df_double_chance["gol_home"] >= df_double_chance["gol_away"])).sum()
    
    # 12 (Home Win or Away Win)
    count_12 = ((df_double_chance["gol_home"] != df_double_chance["gol_away"])).sum()
    
    # X2 (Draw or Away Win)
    count_x2 = ((df_double_chance["gol_away"] >= df_double_chance["gol_home"])).sum()
    
    data = [
        ["1X", count_1x, round((count_1x / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["12", count_12, round((count_12 / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["X2", count_x2, round((count_x2 / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]
    
    df_stats = pd.DataFrame(data, columns=["Mercato", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_stats


# --- Funzione per calcolare le stats SH ---
def calcola_stats_sh(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    df_sh = df_to_analyze.copy()
    
    # Calcolo dei gol nel secondo tempo
    df_sh["gol_home_sh"] = df_sh["Gol_Home_FT"] - df_sh["Gol_Home_HT"]
    df_sh["gol_away_sh"] = df_sh["Gol_Away_FT"] - df_sh["Gol_Away_HT"]
    
    # Winrate SH
    risultati_sh = {"1 (Casa)": 0, "X (Pareggio)": 0, "2 (Trasferta)": 0}
    for _, row in df_sh.iterrows():
        if row["gol_home_sh"] > row["gol_away_sh"]:
            risultati_sh["1 (Casa)"] += 1
        elif row["gol_home_sh"] < row["gol_away_sh"]:
            risultati_sh["2 (Trasferta)"] += 1
        else:
            risultati_sh["X (Pareggio)"] += 1
    
    total_sh_matches = len(df_sh)
    stats_sh_winrate = []
    for esito, count in risultati_sh.items():
        perc = round((count / total_sh_matches) * 100, 2) if total_sh_matches > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats_sh_winrate.append((esito, count, perc, odd_min))
    df_winrate_sh = pd.DataFrame(stats_sh_winrate, columns=["Esito", "Conteggio", "WinRate %", "Odd Minima"])
    
    # Over Goals SH
    over_sh_data = []
    df_sh["tot_goals_sh"] = df_sh["gol_home_sh"] + df_sh["gol_away_sh"]
    for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
        count = (df_sh["tot_goals_sh"] > t).sum()
        perc = round((count / len(df_sh)) * 100, 2)
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        over_sh_data.append([f"Over {t} SH", count, perc, odd_min])
    df_over_sh = pd.DataFrame(over_sh_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])

    # BTTS SH
    btts_sh_count = ((df_sh["gol_home_sh"] > 0) & (df_sh["gol_away_sh"] > 0)).sum()
    no_btts_sh_count = len(df_sh) - btts_sh_count
    btts_sh_data = [
        ["BTTS SI SH", btts_sh_count, round((btts_sh_count / total_sh_matches) * 100, 2) if total_sh_matches > 0 else 0],
        ["BTTS NO SH", no_btts_sh_count, round((no_btts_sh_count / total_sh_matches) * 100, 2) if total_sh_matches > 0 else 0]
    ]
    df_btts_sh = pd.DataFrame(btts_sh_data, columns=["Mercato", "Conteggio", "Percentuale %"])
    df_btts_sh["Odd Minima"] = df_btts_sh["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_winrate_sh, df_over_sh, df_btts_sh


# --- Nuova funzione per calcolare le stats SH complete ---
def calcola_first_to_score_sh(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    
    risultati = {"Home Team": 0, "Away Team": 0, "No Goals SH": 0}
    total_matches = len(df_to_analyze)

    for _, row in df_to_analyze.iterrows():
        gol_home_str = str(row.get("Minutaggio_Gol_Home", ""))
        gol_away_str = str(row.get("Minutaggio_gol_Away", ""))

        gol_home = [int(x) for x in gol_home_str.split(";") if x.isdigit() and int(x) > 45]
        gol_away = [int(x) for x in gol_away_str.split(";") if x.isdigit() and int(x) > 45]

        min_home_goal = min(gol_home) if gol_home else float('inf')
        min_away_goal = min(gol_away) if gol_away else float('inf')
        
        if min_home_goal < min_away_goal:
            risultati["Home Team"] += 1
        elif min_away_goal < min_home_goal:
            risultati["Away Team"] += 1
        else:
            if min_home_goal == float('inf'):
                risultati["No Goals SH"] += 1

    stats = []
    for esito, count in risultati.items():
        perc = round((count / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats.append((esito, count, perc, odd_min))
    
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

def calcola_first_to_score_outcome_sh(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    
    risultati = {
        "Casa Segna Primo SH e Vince": 0,
        "Casa Segna Primo SH e Non Vince": 0,
        "Trasferta Segna Prima SH e Vince": 0,
        "Trasferta Segna Prima SH e Non Vince": 0,
        "Nessun Gol SH": 0
    }
    
    total_matches = len(df_to_analyze)

    for _, row in df_to_analyze.iterrows():
        gol_home_str = str(row.get("Minutaggio_Gol_Home", ""))
        gol_away_str = str(row.get("Minutaggio_gol_Away", ""))

        gol_home_sh = [int(x) for x in gol_home_str.split(";") if x.isdigit() and int(x) > 45]
        gol_away_sh = [int(x) for x in gol_away_str.split(";") if x.isdigit() and int(x) > 45]

        min_home_goal = min(gol_home_sh) if gol_home_sh else float('inf')
        min_away_goal = min(gol_away_sh) if gol_away_sh else float('inf')
        
        home_vince = row["Gol_Home_FT"] > row["Gol_Away_FT"]
        away_vince = row["Gol_Away_FT"] > row["Gol_Home_FT"]
        
        if min_home_goal < min_away_goal:
            if home_vince:
                risultati["Casa Segna Primo SH e Vince"] += 1
            else:
                risultati["Casa Segna Primo SH e Non Vince"] += 1
        elif min_away_goal < min_home_goal:
            if away_vince:
                risultati["Trasferta Segna Prima SH e Vince"] += 1
            else:
                risultati["Trasferta Segna Prima SH e Non Vince"] += 1
        else:
            risultati["Nessun Gol SH"] += 1

    stats = []
    for esito, count in risultati.items():
        perc = round((count / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats.append((esito, count, perc, odd_min))
    
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

def calcola_first_to_score_next_goal_outcome_sh(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    
    risultati = {
        "Casa Segna Primo SH e Segna di Nuovo SH": 0,
        "Casa Segna Primo SH e Subisce Gol SH": 0,
        "Trasferta Segna Prima SH e Segna di Nuovo SH": 0,
        "Trasferta Segna Prima SH e Subisce Gol SH": 0,
        "Solo un gol SH o nessuno": 0
    }
    
    total_matches = len(df_to_analyze)

    for _, row in df_to_analyze.iterrows():
        gol_home_str = str(row.get("Minutaggio_Gol_Home", ""))
        gol_away_str = str(row.get("Minutaggio_gol_Away", ""))

        gol_home_sh = sorted([int(x) for x in gol_home_str.split(";") if x.isdigit() and int(x) > 45])
        gol_away_sh = sorted([int(x) for x in gol_away_str.split(";") if x.isdigit() and int(x) > 45])

        all_goals = []
        if gol_home_sh:
            all_goals.extend([ (t, 'home') for t in gol_home_sh ])
        if gol_away_sh:
            all_goals.extend([ (t, 'away') for t in gol_away_sh ])
        
        if len(all_goals) < 2:
            risultati["Solo un gol SH o nessuno"] += 1
            continue
            
        all_goals.sort()
        
        first_goal = all_goals[0]
        second_goal = all_goals[1]
        
        first_scorer = first_goal[1]
        second_scorer = second_goal[1]
        
        if first_scorer == 'home':
            if second_scorer == 'home':
                risultati["Casa Segna Primo SH e Segna di Nuovo SH"] += 1
            else:
                risultati["Casa Segna Primo SH e Subisce Gol SH"] += 1
        elif first_scorer == 'away':
            if second_scorer == 'away':
                risultati["Trasferta Segna Prima SH e Segna di Nuovo SH"] += 1
            else:
                risultati["Trasferta Segna Prima SH e Subisce Gol SH"] += 1

    stats = []
    for esito, count in risultati.items():
        perc = round((count / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats.append((esito, count, perc, odd_min))
    
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

def calcola_to_score_sh(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

    df_to_score = df_to_analyze.copy()

    df_to_score["gol_home_sh"] = df_to_analyze["Gol_Home_FT"] - df_to_analyze["Gol_Home_HT"]
    df_to_score["gol_away_sh"] = df_to_analyze["Gol_Away_FT"] - df_to_analyze["Gol_Away_HT"]

    home_to_score_count = (df_to_score["gol_home_sh"] > 0).sum()
    away_to_score_count = (df_to_score["gol_away_sh"] > 0).sum()
    
    total_matches = len(df_to_score)
    
    stats = [
        ["Home Team to Score SH", home_to_score_count, round((home_to_score_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["Away Team to Score SH", away_to_score_count, round((away_to_score_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]
    
    df_stats = pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_stats

def calcola_clean_sheet_sh(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    
    df_clean_sheet = df_to_analyze.copy()
    
    df_clean_sheet["gol_home_sh"] = df_clean_sheet["Gol_Home_FT"] - df_clean_sheet["Gol_Home_HT"]
    df_clean_sheet["gol_away_sh"] = df_clean_sheet["Gol_Away_FT"] - df_clean_sheet["Gol_Away_HT"]
    
    home_clean_sheet_count = (df_clean_sheet["gol_away_sh"] == 0).sum()
    away_clean_sheet_count = (df_clean_sheet["gol_home_sh"] == 0).sum()
    
    total_matches = len(df_clean_sheet)
    
    data = [
        ["Clean Sheet SH (Casa)", home_clean_sheet_count, round((home_clean_sheet_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["Clean Sheet SH (Trasferta)", away_clean_sheet_count, round((away_clean_sheet_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]
    
    df_stats = pd.DataFrame(data, columns=["Esito", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_stats

# --- Funzione per calcolare le percentuali di gol fatti/subiti per squadra/periodo ---
def calcola_goals_per_team_period(df_to_analyze, team_type, action_type, period):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=[f"Mercato (Over {period})", "Conteggio", "Percentuale %", "Odd Minima"])
    
    df_temp = df_to_analyze.copy()
    
    if period == 'ft':
        scored_col = "Gol_Home_FT" if team_type == 'home' else "Gol_Away_FT"
        conceded_col = "Gol_Away_FT" if team_type == 'home' else "Gol_Home_FT"
    elif period == 'ht':
        scored_col = "Gol_Home_HT" if team_type == 'home' else "Gol_Away_HT"
        conceded_col = "Gol_Away_HT" if team_type == 'home' else "Gol_Home_HT"
    elif period == 'sh':
        df_temp["gol_home_sh"] = df_temp["Gol_Home_FT"] - df_temp["Gol_Home_HT"]
        df_temp["gol_away_sh"] = df_temp["Gol_Away_FT"] - df_temp["Gol_Away_HT"]
        scored_col = "gol_home_sh" if team_type == 'home' else "gol_away_sh"
        conceded_col = "gol_away_sh" if team_type == 'home' else "gol_home_sh"
    else:
        st.error("Periodo non valido per il calcolo della doppia chance.")
        return pd.DataFrame()
    
    col_to_analyze = scored_col if action_type == 'fatti' else conceded_col
    
    total_matches = len(df_temp)
    
    ranges = [0.5, 1.5]
    data = []
    
    for r in ranges:
        count = (df_temp[col_to_analyze] > r).sum()
        perc = round((count / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        data.append([f"Over {r}", count, perc, odd_min])
        
    df_results = pd.DataFrame(data, columns=[f"Mercato (Over {period})", "Conteggio", "Percentuale %", "Odd Minima"])
    return df_results


# --- FUNZIONE WINRATE ---
def calcola_winrate(df, col_risultato):
    df_valid = df[df[col_risultato].notna() & (df[col_risultato].str.contains("-"))]
    risultati = {"1 (Casa)": 0, "X (Pareggio)": 0, "2 (Trasferta)": 0}
    for ris in df_valid[col_risultato]:
        try:
            home, away = map(int, ris.split("-"))
            if home > away:
                risultati["1 (Casa)"] += 1
            elif home < away:
                risultati["2 (Trasferta)"] += 1
            else:
                risultati["X (Pareggio)"] += 1
        except:
            continue
    totale = len(df_valid)
    stats = []
    for esito, count in risultati.items():
        perc = round((count / totale) * 100, 2) if totale > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats.append((esito, count, perc, odd_min))
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "WinRate %", "Odd Minima"])

# --- FUNZIONE CALCOLO FIRST TO SCORE ---
def calcola_first_to_score(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    
    risultati = {"Home Team": 0, "Away Team": 0, "No Goals": 0}
    total_matches = len(df_to_analyze)

    for _, row in df_to_analyze.iterrows():
        gol_home_str = str(row.get("Minutaggio_Gol_Home", ""))
        gol_away_str = str(row.get("Minutaggio_gol_Away", ""))

        gol_home = [int(x) for x in gol_home_str.split(";") if x.isdigit()]
        gol_away = [int(x) for x in gol_away_str.split(";") if x.isdigit()]

        min_home_goal = min(gol_home) if gol_home else float('inf')
        min_away_goal = min(gol_away) if gol_away else float('inf')
        
        if min_home_goal < min_away_goal:
            risultati["Home Team"] += 1
        elif min_away_goal < min_home_goal:
            risultati["Away Team"] += 1
        else:
            if min_home_goal == float('inf'):
                risultati["No Goals"] += 1

    stats = []
    for esito, count in risultati.items():
        perc = round((count / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats.append((esito, count, perc, odd_min))
    
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

# --- NUOVA FUNZIONE CALCOLO FIRST TO SCORE HT ---
def calcola_first_to_score_ht(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    
    risultati = {"Home Team": 0, "Away Team": 0, "No Goals": 0}
    total_matches = len(df_to_analyze)

    for _, row in df_to_analyze.iterrows():
        gol_home_str = str(row.get("Minutaggio_Gol_Home", ""))
        gol_away_str = str(row.get("Minutaggio_gol_Away", ""))

        gol_home = [int(x) for x in gol_home_str.split(";") if x.isdigit() and int(x) <= 45]
        gol_away = [int(x) for x in gol_away_str.split(";") if x.isdigit() and int(x) <= 45]

        min_home_goal = min(gol_home) if gol_home else float('inf')
        min_away_goal = min(gol_away) if gol_away else float('inf')
        
        if min_home_goal < min_away_goal:
            risultati["Home Team"] += 1
        elif min_away_goal < min_home_goal:
            risultati["Away Team"] += 1
        else:
            if min_home_goal == float('inf'):
                risultati["No Goals"] += 1

    stats = []
    for esito, count in risultati.items():
        perc = round((count / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats.append((esito, count, perc, odd_min))
    
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

# --- FUNZIONE RISULTATI ESATTI ---
def mostra_risultati_esatti(df, col_risultato, titolo):
    risultati_interessanti = [
        "0-0", "0-1", "0-2", "0-3",
        "1-0", "1-1", "1-2", "1-3",
        "2-0", "2-1", "2-2", "2-3",
        "3-0", "3-1", "3-2", "3-3"
    ]
    df_valid = df[df[col_risultato].notna() & (df[col_risultato].str.contains("-"))].copy()

    if df_valid.empty:
        st.subheader(f"Risultati Esatti {titolo} (0 partite)")
        st.info("Nessun dato valido per i risultati esatti nel dataset filtrato.")
        return

    def classifica_risultato(ris):
        try:
            home, away = map(int, ris.split("-"))
        except:
            return "Altro"
        if ris in risultati_interessanti:
            return ris
        if home > away:
            return "Altro risultato casa vince"
        elif home < away:
            return "Altro risultato ospite vince"
        else:
            return "Altro pareggio"

    df_valid["classificato"] = df_valid[col_risultato].apply(classifica_risultato)
    distribuzione = df_valid["classificato"].value_counts().reset_index()
    distribuzione.columns = [titolo, "Conteggio"]
    distribuzione["Percentuale %"] = (distribuzione["Conteggio"] / len(df_valid) * 100).round(2)
    distribuzione["Odd Minima"] = distribuzione["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")

    st.subheader(f"Risultati Esatti {titolo} ({len(df_valid)} partite)")
    styled_df = distribuzione.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
    st.dataframe(styled_df)

# --- FUNZIONE RIUTILIZZABILE PER DISTRIBUZIONE TIMEBAND (15 MIN) ---
def mostra_distribuzione_timeband(df_to_analyze, min_start_display=0):
    if df_to_analyze.empty:
        st.warning("Il DataFrame per l'analisi a 15 minuti è vuoto.")
        return

    all_intervalli = [(0, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90), (91, 150)]
    all_label_intervalli = ["0-15", "16-30", "31-45", "46-60", "61-75", "76-90", "90+"]

    risultati = []
    total_matches = len(df_to_analyze)
    
    for i, ((start_interval, end_interval), label) in enumerate(zip(all_intervalli, all_label_intervalli)):
        if end_interval < min_start_display:
            continue

        partite_con_gol = 0
        for _, row in df_to_analyze.iterrows():
            gol_home_str = str(row.get("Minutaggio_Gol_Home", ""))
            gol_away_str = str(row.get("Minutaggio_gol_Away", ""))

            gol_home = [int(x) for x in gol_home_str.split(";") if x.isdigit()]
            gol_away = [int(x) for x in gol_away_str.split(";") if x.isdigit()]
            
            goals_in_relevant_part_of_interval = [
                g for g in (gol_home + gol_away) 
                if max(start_interval, min_start_display) <= g <= end_interval
            ]
            
            if goals_in_relevant_part_of_interval:
                partite_con_gol += 1
        
        perc = round((partite_con_gol / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        risultati.append([label, partite_con_gol, perc, odd_min])
    
    if not risultati:
        st.info(f"Nessun intervallo di tempo rilevante dopo il minuto {min_start_display} per l'analisi a 15 minuti.")
        return

    df_result = pd.DataFrame(risultati, columns=["Timeframe", "Partite con Gol", "Percentuale %", "Odd Minima"])
    styled_df = df_result.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
    st.dataframe(styled_df)

# --- NUOVA FUNZIONE RIUTILIZZABILE PER DISTRIBUZIONE TIMEBAND (5 MIN) ---
def mostra_distribuzione_timeband_5min(df_to_analyze, min_start_display=0):
    if df_to_analyze.empty:
        return
    all_intervalli = [(0,5), (6,10), (11,15), (16,20), (21,25), (26,30), (31,35), (36,40), (41,45), (46,50), (51,55), (56,60), (61,65), (66,70), (71,75), (76,80), (81,85), (86,90), (91, 150)]
    all_label_intervalli = ["0-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50", "51-55", "56-60", "61-65", "66-70", "71-75", "76-80", "81-85", "86-90", "90+"]
    risultati = []
    total_matches = len(df_to_analyze)
    for (start_interval, end_interval), label in zip(all_intervalli, all_label_intervalli):
        if end_interval < min_start_display:
            continue

        partite_con_gol = 0
        for _, row in df_to_analyze.iterrows():
            gol_home = [int(x) for x in str(row.get("Minutaggio_Gol_Home", "")).split(";") if x.isdigit()]
            gol_away = [int(x) for x in str(row.get("Minutaggio_gol_Away", "")).split(";") if x.isdigit()]
            
            goals_in_relevant_part_of_interval = [
                g for g in (gol_home + gol_away) 
                if max(start_interval, min_start_display) <= g <= end_interval
            ]

            if goals_in_relevant_part_of_interval:
                partite_con_gol += 1
        perc = round((partite_con_gol / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        risultati.append([label, partite_con_gol, perc, odd_min])
    
    if not risultati:
        st.info(f"Nessun intervallo di tempo rilevante dopo il minuto {min_start_display} per l'analisi a 5 minuti.")
        return

    df_result = pd.DataFrame(risultati, columns=["Timeframe", "Partite con Gol", "Percentuale %", "Odd Minima"])
    styled_df = df_result.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
    st.dataframe(styled_df)

# --- FUNZIONE NEXT GOAL ---
def calcola_next_goal(df_to_analyze, start_min, end_min):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    
    risultati = {"Prossimo Gol: Home": 0, "Prossimo Gol: Away": 0, "Nessun prossimo gol": 0}
    total_matches = len(df_to_analyze)

    for _, row in df_to_analyze.iterrows():
        gol_home_str = str(row.get("Minutaggio_Gol_Home", ""))
        gol_away_str = str(row.get("Minutaggio_gol_Away", ""))

        gol_home = [int(x) for x in gol_home_str.split(";") if x.isdigit()]
        gol_away = [int(x) for x in gol_away_str.split(";") if x.isdigit()]

        next_home_goal = min([g for g in gol_home if start_min <= g <= end_min] or [float('inf')])
        next_away_goal = min([g for g in gol_away if start_min <= g <= end_min] or [float('inf')])
        
        if next_home_goal < next_away_goal:
            risultati["Prossimo Gol: Home"] += 1
        elif next_away_goal < next_home_goal:
            risultati["Prossimo Gol: Away"] += 1
        else:
            if next_home_goal == float('inf'):
                risultati["Nessun prossimo gol"] += 1

    stats = []
    for esito, count in risultati.items():
        perc = round((count / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats.append((esito, count, perc, odd_min))
    
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

# --- NUOVE FUNZIONI PER ANALISI RIMONTE ---
def calcola_rimonte(df_to_analyze, titolo_analisi):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Tipo Rimonta", "Conteggio", "Percentuale %", "Odd Minima"]), {}

    partite_rimonta_parziale = []
    partite_rimonta_completa = []
    
    df_rimonte = df_to_analyze.copy()
    
    def check_comeback(row):
        if row["Gol_Home_HT"] < row["Gol_Away_HT"] and row["Gol_Home_FT"] > row["Gol_Away_FT"]:
            return "Completa - Home"
        if row["Gol_Home_HT"] < row["Gol_Away_HT"] and row["Gol_Home_FT"] == row["Gol_Away_FT"]:
            return "Parziale - Home"
        if row["Gol_Away_HT"] < row["Gol_Home_HT"] and row["Gol_Away_FT"] > row["Gol_Home_FT"]:
            return "Completa - Away"
        if row["Gol_Away_HT"] < row["Gol_Home_HT"] and row["Gol_Away_FT"] == row["Gol_Home_FT"]:
            return "Parziale - Away"
        return "Nessuna"

    df_rimonte["rimonta"] = df_rimonte.apply(check_comeback, axis=1)
    
    rimonte_completa_home = (df_rimonte["rimonta"] == "Completa - Home").sum()
    rimonte_parziale_home = (df_rimonte["rimonta"] == "Parziale - Home").sum()
    rimonte_completa_away = (df_rimonte["rimonta"] == "Completa - Away").sum()
    rimonte_parziale_away = (df_rimonte["rimonta"] == "Parziale - Away").sum()

    total_matches = len(df_rimonte)
    
    rimonte_data = [
        ["Rimonta Completa (Home)", rimonte_completa_home, round((rimonte_completa_home / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["Rimonta Parziale (Home)", rimonte_parziale_home, round((rimonte_parziale_home / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["Rimonta Completa (Away)", rimonte_completa_away, round((rimonte_completa_away / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["Rimonta Parziale (Away)", rimonte_parziale_away, round((rimonte_parziale_away / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]

    df_rimonte_stats = pd.DataFrame(rimonte_data, columns=["Tipo Rimonta", "Conteggio", "Percentuale %"])
    df_rimonte_stats["Odd Minima"] = df_rimonte_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    squadre_rimonta_completa_home = df_rimonte[df_rimonte["rimonta"] == "Completa - Home"]["Home_Team"].tolist()
    squadre_rimonta_parziale_home = df_rimonte[df_rimonte["rimonta"] == "Parziale - Home"]["Home_Team"].tolist()
    squadre_rimonta_completa_away = df_rimonte[df_rimonte["rimonta"] == "Completa - Away"]["Away_Team"].tolist()
    squadre_rimonta_parziale_away = df_rimonte[df_rimonte["rimonta"] == "Parziale - Away"]["Away_Team"].tolist()
    
    squadre_rimonte = {
        "Rimonta Completa (Home)": squadre_rimonta_completa_home,
        "Rimonta Parziale (Home)": squadre_rimonta_parziale_home,
        "Rimonta Completa (Away)": squadre_rimonta_completa_away,
        "Rimonta Parziale (Away)": squadre_rimonta_parziale_away
    }

    return df_rimonte_stats, squadre_rimonte

# --- NUOVA FUNZIONE PER TO SCORE ---
def calcola_to_score(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

    df_to_score = df_to_analyze.copy()

    home_to_score_count = (df_to_score["Gol_Home_FT"] > 0).sum()
    away_to_score_count = (df_to_score["Gol_Away_FT"] > 0).sum()
    
    total_matches = len(df_to_score)
    
    data = [
        ["Home Team to Score", home_to_score_count, round((home_to_score_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["Away Team to Score", away_to_score_count, round((away_to_score_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]
    
    df_stats = pd.DataFrame(data, columns=["Esito", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_stats

# --- NUOVA FUNZIONE PER TO SCORE HT ---
def calcola_to_score_ht(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

    df_to_score = df_to_analyze.copy()

    home_to_score_count = (df_to_score["Gol_Home_HT"] > 0).sum()
    away_to_score_count = (df_to_score["Gol_Away_HT"] > 0).sum()
    
    total_matches = len(df_to_analyze)
    
    data = [
        ["Home Team to Score", home_to_score_count, round((home_to_score_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["Away Team to Score", away_to_score_count, round((away_to_score_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]
    
    df_stats = pd.DataFrame(data, columns=["Esito", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_stats

# --- NUOVA FUNZIONE PER BTTS HT ---
def calcola_btts_ht(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])

    df_btts_ht = df_to_analyze.copy()
    
    btts_count = ((df_btts_ht["Gol_Home_HT"] > 0) & (df_btts_ht["Gol_Away_HT"] > 0)).sum()
    no_btts_count = len(df_btts_ht) - btts_count
    
    total_matches = len(df_btts_ht)
    
    data = [
        ["BTTS SI HT (Dinamica)", btts_count, round((btts_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["BTTS NO HT (Dinamica)", no_btts_count, round((no_btts_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]

    df_stats = pd.DataFrame(data, columns=["Mercato", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_stats

# --- NUOVA FUNZIONE PER BTTS FT ---
def calcola_btts_ft(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])

    df_btts_ft = df_to_analyze.copy()
    
    btts_count = ((df_btts_ft["Gol_Home_FT"] > 0) & (df_btts_ft["Gol_Away_FT"] > 0)).sum()
    no_btts_count = len(df_btts_ft) - btts_count
    
    total_matches = len(df_btts_ft)
    
    data = [
        ["BTTS SI FT", btts_count, round((btts_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["BTTS NO FT", no_btts_count, round((no_btts_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]

    df_stats = pd.DataFrame(data, columns=["Mercato", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_stats

# --- NUOVA FUNZIONE PER BTTS DINAMICO ---
def calcola_btts_dinamico(df_to_analyze, start_min, risultati_correnti):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])

    total_matches = len(df_to_analyze)
    btts_si_count = 0

    for _, row in df_to_analyze.iterrows():
        gol_home_ft = int(row.get("Gol_Home_FT", 0))
        gol_away_ft = int(row.get("Gol_Away_FT", 0))
        
        if (gol_home_ft > 0 and gol_away_ft > 0):
            btts_si_count += 1

    no_btts_count = total_matches - btts_si_count

    data = [
        ["BTTS SI (Dinamica)", btts_si_count, round((btts_si_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["BTTS NO (Dinamica)", no_btts_count, round((no_btts_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]

    df_stats = pd.DataFrame(data, columns=["Mercato", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")

    return df_stats
    
# --- NUOVA FUNZIONE PER BTTS HT DINAMICO ---
def calcola_btts_ht_dinamico(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])

    df_btts_ht_dinamico = df_to_analyze.copy()
    
    btts_count = ((df_btts_ht_dinamico["Gol_Home_HT"] > 0) & (df_btts_ht_dinamico["Gol_Away_HT"] > 0)).sum()
    no_btts_count = len(df_btts_ht_dinamico) - btts_count
    
    total_matches = len(df_btts_ht_dinamico)
    
    data = [
        ["BTTS SI HT (Dinamica)", btts_count, round((btts_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["BTTS NO HT (Dinamica)", no_btts_count, round((no_btts_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]

    df_stats = pd.DataFrame(data, columns=["Mercato", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_stats

# --- NUOVA FUNZIONE PER CLEAN SHEET ---
def calcola_clean_sheet(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    
    df_clean_sheet = df_to_analyze.copy()
    
    home_clean_sheet_count = (df_clean_sheet["Gol_Away_FT"] == 0).sum()
    away_clean_sheet_count = (df_clean_sheet["Gol_Home_FT"] == 0).sum()
    
    total_matches = len(df_clean_sheet)
    
    data = [
        ["Clean Sheet (Casa)", home_clean_sheet_count, round((home_clean_sheet_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["Clean Sheet (Trasferta)", away_clean_sheet_count, round((away_clean_sheet_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]
    
    df_stats = pd.DataFrame(data, columns=["Esito", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_stats

# --- NUOVA FUNZIONE PER COMBO MARKETS ---
def calcola_combo_stats(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
        
    df_combo = df_to_analyze.copy()

    df_combo["tot_goals_ft"] = df_combo["Gol_Home_FT"] + df_combo["Gol_Away_FT"]
    
    # BTTS SI + Over 2.5
    btts_over_2_5_count = ((df_combo["Gol_Home_FT"] > 0) & (df_combo["Gol_Away_FT"] > 0) & (df_combo["tot_goals_ft"] > 2.5)).sum()
    
    # Home Win + Over 2.5
    home_win_over_2_5_count = ((df_combo["Gol_Home_FT"] > df_combo["Gol_Away_FT"]) & (df_combo["tot_goals_ft"] > 2.5)).sum()
    
    # Away Win + Over 2.5
    away_win_over_2_5_count = ((df_combo["Gol_Away_FT"] > df_combo["Gol_Home_FT"]) & (df_combo["tot_goals_ft"] > 2.5)).sum()
    
    total_matches = len(df_combo)
    
    data = [
        ["BTTS SI + Over 2.5", btts_over_2_5_count, round((btts_over_2_5_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["Casa vince + Over 2.5", home_win_over_2_5_count, round((home_win_over_2_5_count / total_matches) * 100, 2) if total_matches > 0 else 0],
        ["Ospite vince + Over 2.5", away_win_over_2_5_count, round((away_win_over_2_5_count / total_matches) * 100, 2) if total_matches > 0 else 0]
    ]

    df_stats = pd.DataFrame(data, columns=["Mercato", "Conteggio", "Percentuale %"])
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
    
    return df_stats

# --- NUOVA FUNZIONE PER MULTI GOL ---
def calcola_multi_gol(df_to_analyze, col_gol, titolo):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=[f"Mercato ({titolo})", "Conteggio", "Percentuale %", "Odd Minima"])
        
    df_multi_gol = df_to_analyze.copy()
    
    total_matches = len(df_multi_gol)
    
    multi_gol_ranges = [
        ("0-1", lambda x: (x >= 0) & (x <= 1)),
        ("1-2", lambda x: (x >= 1) & (x <= 2)),
        ("2-3", lambda x: (x >= 2) & (x <= 3)),
        ("3+", lambda x: (x >= 3))
    ]
    
    data = []
    for label, condition in multi_gol_ranges:
        count = df_multi_gol[condition(df_multi_gol[col_gol])].shape[0]
        perc = round((count / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        data.append([f"Multi Gol {label}", count, perc, odd_min])
        
    df_stats = pd.DataFrame(data, columns=[f"Mercato ({titolo})", "Conteggio", "Percentuale %", "Odd Minima"])
    return df_stats

# SEZIONE 1: Analisi Timeband per Campionato
st.subheader("1. Analisi Timeband per Campionato")
if selected_league != "Tutte":
    df_league_only = df[df["League"] == selected_league]
    st.write(f"Analisi basata su **{len(df_league_only)}** partite del campionato **{selected_league}**.")
    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Distribuzione Gol per Timeframe (15min)**")
        mostra_distribuzione_timeband(df_league_only)
    with col2:
        st.write("**Distribuzione Gol per Timeframe (5min)**")
        mostra_distribuzione_timeband_5min(df_league_only)
else:
    st.write("Seleziona un campionato per visualizzare questa analisi.")

# SEZIONE 2: Analisi Timeband per Campionato e Quote
st.subheader("2. Analisi Timeband per Campionato e Quote")
st.write(f"Analisi basata su **{len(filtered_df)}** partite filtrate da tutti i parametri della sidebar.")
if not filtered_df.empty:
    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Distribuzione Gol per Timeframe (15min)**")
        mostra_distribuzione_timeband(filtered_df)
    with col2:
        st.write("**Distribuzione Gol per Timeframe (5min)**")
        mostra_distribuzione_timeband_5min(filtered_df)
else:
    st.warning("Nessuna partita corrisponde ai filtri selezionati.")


# NUOVA SEZIONE: Statistiche Pre-Match Complete (Filtri Sidebar)
st.subheader("3. Analisi Pre-Match Completa (Filtri Sidebar)")
st.write(f"Analisi completa basata su **{len(filtered_df)}** partite, considerando tutti i filtri del menu a sinistra.")
if not filtered_df.empty:
    
    st.subheader("Media Gol (Pre-Match)")
    df_prematch_goals = filtered_df.copy()
    
    avg_ht_goals = (df_prematch_goals["Gol_Home_HT"] + df_prematch_goals["Gol_Away_HT"]).mean()
    avg_ft_goals = (df_prematch_goals["Gol_Home_FT"] + df_prematch_goals["Gol_Away_FT"]).mean()
    avg_sh_goals = (df_prematch_goals["Gol_Home_FT"] + df_prematch_goals["Gol_Away_FT"] - df_prematch_goals["Gol_Home_HT"] - df_prematch_goals["Gol_Away_HT"]).mean()
    
    st.table(pd.DataFrame({
        "Periodo": ["HT", "FT", "SH"],
        "Media Gol": [f"{avg_ht_goals:.2f}", f"{avg_ft_goals:.2f}", f"{avg_sh_goals:.2f}"]
    }))

    with st.expander("Mostra Statistiche HT"):
        mostra_risultati_esatti(filtered_df, "risultato_ht", f"HT ({len(filtered_df)})")
        st.subheader(f"WinRate HT ({len(filtered_df)})")
        df_winrate_ht = calcola_winrate(filtered_df, "risultato_ht")
        styled_df_ht = df_winrate_ht.style.background_gradient(cmap='RdYlGn', subset=['WinRate %'])
        st.dataframe(styled_df_ht)
        st.subheader(f"Over Goals HT ({len(filtered_df)})")
        over_ht_data = []
        df_prematch_ht = filtered_df.copy()
        df_prematch_ht["tot_goals_ht"] = df_prematch_ht["Gol_Home_HT"] + df_prematch_ht["Gol_Away_HT"]
        for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
            count = (df_prematch_ht["tot_goals_ht"] > t).sum()
            perc = round((count / len(df_prematch_ht)) * 100, 2)
            odd_min = round(100 / perc, 2) if perc > 0 else "-"
            over_ht_data.append([f"Over {t} HT", count, perc, odd_min])
        df_over_ht = pd.DataFrame(over_ht_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
        styled_over_ht = df_over_ht.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_over_ht)
        st.subheader(f"BTTS HT ({len(filtered_df)})")
        df_btts_ht = calcola_btts_ht(filtered_df)
        styled_df = df_btts_ht.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_df)
        st.subheader(f"Doppia Chance HT ({len(filtered_df)})")
        df_dc_ht = calcola_double_chance(filtered_df, 'ht')
        styled_df = df_dc_ht.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_df)
        st.subheader(f"First to Score HT ({len(filtered_df)})")
        styled_df = calcola_first_to_score_ht(filtered_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_df)
        st.subheader(f"To Score HT ({len(filtered_df)})")
        styled_df = calcola_to_score_ht(filtered_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_df)
        st.subheader(f"Goals Fatti e Subiti HT ({len(filtered_df)})")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Fatti Casa")
            st.dataframe(calcola_goals_per_team_period(filtered_df, 'home', 'fatti', 'ht').style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
        with col2:
            st.markdown("#### Subiti Casa")
            st.dataframe(calcola_goals_per_team_period(filtered_df, 'home', 'subiti', 'ht').style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### Fatti Trasferta")
            st.dataframe(calcola_goals_per_team_period(filtered_df, 'away', 'fatti', 'ht').style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
        with col4:
            st.markdown("#### Subiti Trasferta")
            st.dataframe(calcola_goals_per_team_period(filtered_df, 'away', 'subiti', 'ht').style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
    
    with st.expander("Mostra Statistiche SH (Secondo Tempo)"):
        st.write(f"Analisi basata su **{len(filtered_df)}** partite.")
        df_sh = filtered_df.copy()
        df_sh["gol_home_sh"] = df_sh["Gol_Home_FT"] - df_sh["Gol_Home_HT"]
        df_sh["gol_away_sh"] = df_sh["Gol_Away_FT"] - df_sh["Gol_Away_HT"]
        
        st.subheader(f"WinRate SH ({len(filtered_df)})")
        risultati_sh = {"1 (Casa)": 0, "X (Pareggio)": 0, "2 (Trasferta)": 0}
        for _, row in df_sh.iterrows():
            if row["gol_home_sh"] > row["gol_away_sh"]:
                risultati_sh["1 (Casa)"] += 1
            elif row["gol_home_sh"] < row["gol_away_sh"]:
                risultati_sh["2 (Trasferta)"] += 1
            else:
                risultati_sh["X (Pareggio)"] += 1
        
        total_sh_matches = len(df_sh)
        stats_sh_winrate = []
        for esito, count in risultati_sh.items():
            perc = round((count / total_sh_matches) * 100, 2) if total_sh_matches > 0 else 0
            odd_min = round(100 / perc, 2) if perc > 0 else "-"
            stats_sh_winrate.append((esito, count, perc, odd_min))
        df_winrate_sh = pd.DataFrame(stats_sh_winrate, columns=["Esito", "Conteggio", "WinRate %", "Odd Minima"])
        styled_df = df_winrate_sh.style.background_gradient(cmap='RdYlGn', subset=['WinRate %'])
        st.dataframe(styled_df)

        st.subheader(f"Over Goals SH ({len(filtered_df)})")
        over_sh_data = []
        df_sh["tot_goals_sh"] = df_sh["gol_home_sh"] + df_sh["gol_away_sh"]
        for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
            count = (df_sh["tot_goals_sh"] > t).sum()
            perc = round((count / len(df_sh)) * 100, 2)
            odd_min = round(100 / perc, 2) if perc > 0 else "-"
            over_sh_data.append([f"Over {t} SH", count, perc, odd_min])
        df_over_sh = pd.DataFrame(over_sh_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
        styled_df = df_over_sh.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_df)
        
        st.subheader(f"BTTS SH ({len(filtered_df)})")
        btts_sh_count = ((df_sh["gol_home_sh"] > 0) & (df_sh["gol_away_sh"] > 0)).sum()
        no_btts_sh_count = len(df_sh) - btts_sh_count
        btts_sh_data = [
            ["BTTS SI SH", btts_sh_count, round((btts_sh_count / total_sh_matches) * 100, 2) if total_sh_matches > 0 else 0],
            ["BTTS NO SH", no_btts_sh_count, round((no_btts_sh_count / total_sh_matches) * 100, 2) if total_sh_matches > 0 else 0]
        ]
        df_btts_sh = pd.DataFrame(btts_sh_data, columns=["Mercato", "Conteggio", "Percentuale %"])
        df_btts_sh["Odd Minima"] = df_btts_sh["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
        styled_df = df_btts_sh.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_df)
        
        st.subheader(f"Doppia Chance SH ({len(filtered_df)})")
        df_dc_sh = calcola_double_chance(filtered_df, 'sh')
        styled_df = df_dc_sh.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_df)
        
        st.subheader(f"First to Score SH ({len(filtered_df)})")
        styled_df = calcola_first_to_score_sh(filtered_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_df)

        st.subheader(f"First to Score + Risultato Finale SH ({len(filtered_df)})")
        styled_df = calcola_first_to_score_outcome_sh(filtered_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_df)

        st.subheader(f"First to Score + Risultato Prossimo Gol SH ({len(filtered_df)})")
        styled_df = calcola_first_to_score_next_goal_outcome_sh(filtered_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_df)

        st.subheader(f"To Score SH ({len(filtered_df)})")
        styled_df = calcola_to_score_sh(filtered_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_df)

        st.subheader(f"Clean Sheet SH ({len(filtered_df)})")
        styled_df = calcola_clean_sheet_sh(filtered_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_df)
        
        st.subheader(f"Goals Fatti e Subiti SH ({len(filtered_df)})")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Fatti Casa")
            st.dataframe(calcola_goals_per_team_period(filtered_df, 'home', 'fatti', 'sh').style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
        with col2:
            st.markdown("#### Subiti Casa")
            st.dataframe(calcola_goals_per_team_period(filtered_df, 'home', 'subiti', 'sh').style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### Fatti Trasferta")
            st.dataframe(calcola_goals_per_team_period(filtered_df, 'away', 'fatti', 'sh').style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
        with col4:
            st.markdown("#### Subiti Trasferta")
            st.dataframe(calcola_goals_per_team_period(filtered_df, 'away', 'subiti', 'sh').style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))

    with st.expander("Mostra Statistiche FT (Finale)"):
        mostra_risultati_esatti(filtered_df, "risultato_ft", f"FT ({len(filtered_df)})")
        st.subheader(f"WinRate FT ({len(filtered_df)})")
        df_winrate_ft = calcola_winrate(filtered_df, "risultato_ft")
        styled_df_ft = df_winrate_ft.style.background_gradient(cmap='RdYlGn', subset=['WinRate %'])
        st.dataframe(styled_df_ft)
        st.subheader(f"Over Goals FT ({len(filtered_df)})")
        over_ft_data = []
        df_prematch_ft = filtered_df.copy()
        df_prematch_ft["tot_goals_ft"] = df_prematch_ft["Gol_Home_FT"] + df_prematch_ft["Gol_Away_FT"]
        for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
            count = (df_prematch_ft["tot_goals_ft"] > t).sum()
            perc = round((count / len(df_prematch_ft)) * 100, 2)
            odd_min = round(100 / perc, 2) if perc > 0 else "-"
            over_ft_data.append([f"Over {t} FT", count, perc, odd_min])
        df_over_ft = pd.DataFrame(over_ft_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
        styled_over_ft = df_over_ft.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_over_ft)
        st.subheader(f"BTTS FT ({len(filtered_df)})")
        df_btts_ft = calcola_btts_ft(filtered_df)
        styled_df = df_btts_ft.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_df)
        st.subheader(f"Doppia Chance FT ({len(filtered_df)})")
        df_dc_ft = calcola_double_chance(filtered_df, 'ft')
        styled_df = df_dc_ft.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_df)
        st.subheader(f"Multi Gol (Pre-Match) ({len(filtered_df)})")
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Casa")
            styled_df = calcola_multi_gol(filtered_df, "Gol_Home_FT", "Home").style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
        with col2:
            st.write("### Trasferta")
            styled_df = calcola_multi_gol(filtered_df, "Gol_Away_FT", "Away").style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
        st.subheader(f"First to Score (Pre-Match) ({len(filtered_df)})")
        styled_df = calcola_first_to_score(filtered_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_df)
        st.subheader(f"First to Score + Risultato Finale (Pre-Match) ({len(filtered_df)})")
        styled_df = calcola_first_to_score_outcome(filtered_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_df)
        st.subheader(f"First to Score + Risultato Prossimo Gol (Pre-Match) ({len(filtered_df)})")
        styled_df = calcola_first_to_score_next_goal_outcome(filtered_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_df)
        st.subheader(f"To Score (Pre-Match) ({len(filtered_df)})")
        styled_df = calcola_to_score(filtered_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_df)
        st.subheader(f"Clean Sheet (Pre-Match) ({len(filtered_df)})")
        styled_df = calcola_clean_sheet(filtered_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_df)
        st.subheader(f"Combo Markets (Pre-Match) ({len(filtered_df)})")
        styled_df = calcola_combo_stats(filtered_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
        st.dataframe(styled_df)
        st.subheader(f"Analisi Rimonte (Pre-Match) ({len(filtered_df)})")
        rimonte_stats, squadre_rimonte = calcola_rimonte(filtered_df, "Pre-Match")
        if not rimonte_stats.empty:
            styled_df = rimonte_stats.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            st.markdown("**Squadre che hanno effettuato rimonte:**")
            for tipo, squadre in squadre_rimonte.items():
                if squadre:
                    st.markdown(f"**{tipo}:** {', '.join(squadre)}")
        else:
            st.warning("Nessuna rimonta trovata nel dataset filtrato.")
        
        st.subheader(f"Goals Fatti e Subiti FT ({len(filtered_df)})")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Fatti Casa")
            st.dataframe(calcola_goals_per_team_period(filtered_df, 'home', 'fatti', 'ft').style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
        with col2:
            st.markdown("#### Subiti Casa")
            st.dataframe(calcola_goals_per_team_period(filtered_df, 'home', 'subiti', 'ft').style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### Fatti Trasferta")
            st.dataframe(calcola_goals_per_team_period(filtered_df, 'away', 'fatti', 'ft').style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
        with col4:
            st.markdown("#### Subiti Trasferta")
            st.dataframe(calcola_goals_per_team_period(filtered_df, 'away', 'subiti', 'ft').style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))

else:
    st.warning("Nessuna partita corrisponde ai filtri selezionati per l'analisi pre-match.")

# SEZIONE 4: Analisi Timeband Dinamica (Minuto/Risultato)
st.subheader("4. Analisi Timeband Dinamica")
with st.expander("Mostra Analisi Dinamica (Minuto/Risultato)"):
    if not filtered_df.empty:
        min_range = st.slider("Seleziona Range Minuti", 1, 90, (45, 90))
        start_min, end_min = min_range[0], min_range[1]

        ht_results_to_show = sorted(df["risultato_ht"].dropna().unique()) if "risultato_ht" in df.columns else []
        risultati_correnti = st.multiselect("Risultato corrente al minuto iniziale",
                                             ht_results_to_show,
                                             default=["0-0"] if "0-0" in ht_results_to_show else [])

        partite_target = []
        for _, row in filtered_df.iterrows():
            gol_home_str = str(row.get("Minutaggio_Gol_Home", ""))
            gol_away_str = str(row.get("Minutaggio_gol_Away", ""))
            home_fino = sum(1 for g in [int(x) for x in gol_home_str.split(";") if x.isdigit()] if g < start_min)
            away_fino = sum(1 for g in [int(x) for x in gol_away_str.split(";") if x.isdigit()] if g < start_min)
            risultato_fino = f"{home_fino}-{away_fino}"
            if risultato_fino in risultati_correnti:
                partite_target.append(row)

        if not partite_target:
            st.warning(f"Nessuna partita con risultato selezionato al minuto {start_min}.")
        else:
            df_target = pd.DataFrame(partite_target)
            st.write(f"**Partite trovate:** {len(df_target)}")

            st.subheader("Media Gol (Dinamica)")
            df_target_goals = df_target.copy()
            
            avg_ht_goals_dynamic = (df_target_goals["Gol_Home_HT"] + df_target_goals["Gol_Away_HT"]).mean()
            avg_ft_goals_dynamic = (df_target_goals["Gol_Home_FT"] + df_target_goals["Gol_Away_FT"]).mean()
            avg_sh_goals_dynamic = (df_target_goals["Gol_Home_FT"] + df_target_goals["Gol_Away_FT"] - df_target_goals["Gol_Home_HT"] - df_target_goals["Gol_Away_HT"]).mean()
            
            st.table(pd.DataFrame({
                "Periodo": ["HT", "FT", "SH"],
                "Media Gol": [f"{avg_ht_goals_dynamic:.2f}", f"{avg_ft_goals_dynamic:.2f}", f"{avg_sh_goals_dynamic:.2f}"]
            }))
            
            mostra_risultati_esatti(df_target, "risultato_ht", f"HT ({len(df_target)})")
            mostra_risultati_esatti(df_target, "risultato_ft", f"FT ({len(df_target)})")

            st.subheader(f"WinRate (Dinamica) ({len(df_target)})")
            st.write("**HT:**")
            df_winrate_ht_dynamic = calcola_winrate(df_target, "risultato_ht")
            styled_df_ht = df_winrate_ht_dynamic.style.background_gradient(cmap='RdYlGn', subset=['WinRate %'])
            st.dataframe(styled_df_ht)
            st.write("**FT:**")
            df_winrate_ft_dynamic = calcola_winrate(df_target, "risultato_ft")
            styled_df_ft = df_winrate_ft_dynamic.style.background_gradient(cmap='RdYlGn', subset=['WinRate %'])
            st.dataframe(styled_df_ft)
            
            col1, col2 = st.columns(2)
            df_target_goals["tot_goals_ht"] = df_target_goals["Gol_Home_HT"] + df_target_goals["Gol_Away_HT"]
            df_target_goals["tot_goals_ft"] = df_target_goals["Gol_Home_FT"] + df_target_goals["Gol_Away_FT"]
            
            with col1:
                st.subheader(f"Over Goals HT (Dinamica) ({len(df_target)})")
                over_ht_data_dynamic = []
                for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
                    count = (df_target_goals["tot_goals_ht"] > t).sum()
                    perc = round((count / len(df_target_goals)) * 100, 2)
                    odd_min = round(100 / perc, 2) if perc > 0 else "-"
                    over_ht_data_dynamic.append([f"Over {t} HT", count, perc, odd_min])
                df_over_ht_dynamic = pd.DataFrame(over_ht_data_dynamic, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
                styled_over_ht_dynamic = df_over_ht_dynamic.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_over_ht_dynamic)
            
            with col2:
                st.subheader(f"Over Goals FT (Dinamica) ({len(df_target)})")
                over_ft_data = []
                for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
                    count = (df_target_goals["tot_goals_ft"] > t).sum()
                    perc = round((count / len(df_target_goals)) * 100, 2)
                    odd_min = round(100 / perc, 2) if perc > 0 else "-"
                    over_ft_data.append([f"Over {t} FT", count, perc, odd_min])
                df_over_ft = pd.DataFrame(over_ft_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
                styled_over_ft = df_over_ft.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_over_ft)
            
            st.subheader(f"BTTS (Dinamica) ({len(df_target)})")
            col1, col2 = st.columns(2)
            with col1:
                st.write("### HT")
                df_btts_ht_dynamic = calcola_btts_ht_dinamico(df_target)
                styled_df = df_btts_ht_dynamic.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            with col2:
                st.write("### FT")
                df_btts_ft_dynamic = calcola_btts_dinamico(df_target, start_min, risultati_correnti)
                styled_df = df_btts_ft_dynamic.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)

            st.subheader(f"Doppia Chance (Dinamica) ({len(df_target)})")
            col1, col2 = st.columns(2)
            with col1:
                st.write("### HT")
                df_dc_ht = calcola_double_chance(df_target, 'ht')
                styled_df = df_dc_ht.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            with col2:
                st.write("### FT")
                df_dc_ft = calcola_double_chance(df_target, 'ft')
                styled_df = df_dc_ft.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)

            st.subheader(f"Multi Gol (Dinamica) ({len(df_target)})")
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Casa")
                styled_df = calcola_multi_gol(df_target, "Gol_Home_FT", "Home").style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            with col2:
                st.write("### Trasferta")
                styled_df = calcola_multi_gol(df_target, "Gol_Away_FT", "Away").style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"First to Score HT (Dinamica) ({len(df_target)})")
                styled_df = calcola_first_to_score_ht(df_target).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            with col2:
                st.subheader(f"First to Score FT (Dinamica) ({len(df_target)})")
                styled_df = calcola_first_to_score(df_target).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            
            st.subheader(f"First to Score + Risultato Finale (Dinamica) ({len(df_target)})")
            styled_df = calcola_first_to_score_outcome(df_target).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            
            st.subheader(f"First to Score + Risultato Prossimo Gol (Dinamica) ({len(df_target)})")
            styled_df = calcola_first_to_score_next_goal_outcome(df_target).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"To Score HT (Dinamica) ({len(df_target)})")
                styled_df = calcola_to_score_ht(df_target).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            with col2:
                st.subheader(f"To Score FT (Dinamica) ({len(df_target)})")
                styled_df = calcola_to_score(df_target).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            
            st.subheader(f"Clean Sheet (Dinamica) ({len(df_target)})")
            styled_df = calcola_clean_sheet(df_target).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            
            st.subheader(f"Combo Markets (Dinamica) ({len(df_target)})")
            styled_df = calcola_combo_stats(df_target).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            
            st.subheader(f"Next Goal (Dinamica) ({len(df_target)})")
            styled_df = calcola_next_goal(df_target, start_min, end_min).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            
            st.subheader(f"Analisi Rimonte (Dinamica) ({len(df_target)})")
            rimonte_stats, squadre_rimonte = calcola_rimonte(df_target, "Dinamica")
            if not rimonte_stats.empty:
                styled_df = rimonte_stats.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
                
                st.markdown("**Squadre che hanno effettuato rimonte:**")
                for tipo, squadre in squadre_rimonte.items():
                    if squadre:
                        st.markdown(f"**{tipo}:** {', '.join(squadre)}")
            else:
                st.warning("Nessuna rimonta trovata nel dataset filtrato per questa analisi dinamica.")
            
            st.subheader("Distribuzione Gol per Timeframe (dinamica)")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**15min**")
                mostra_distribuzione_timeband(df_target, min_start_display=start_min)
            with col2:
                st.write("**5min**")
                mostra_distribuzione_timeband_5min(df_target, min_start_display=start_min)

    else:
        st.warning("Il dataset filtrato è vuoto o mancano le colonne necessarie per l'analisi.")

# --- SEZIONE 5: Analisi Head-to-Head (H2H) ---
st.subheader("5. Analisi Head-to-Head (H2H)")
st.write("Seleziona due squadre per analizzare i loro scontri diretti.")

all_teams = sorted(list(set(df['Home_Team'].dropna().unique()) | set(df['Away_Team'].dropna().unique())))
h2h_home_team = st.selectbox("Seleziona Squadra 1", ["Seleziona..."] + all_teams)
h2h_away_team = st.selectbox("Seleziona Squadra 2", ["Seleziona..."] + all_teams)

if h2h_home_team != "Seleziona..." and h2h_away_team != "Seleziona...":
    if h2h_home_team == h2h_away_team:
        st.warning("Seleziona due squadre diverse per l'analisi H2H.")
    else:
        h2h_df = df[((df['Home_Team'] == h2h_home_team) & (df['Away_Team'] == h2h_away_team)) |
                    ((df['Home_Team'] == h2h_away_team) & (df['Away_Team'] == h2h_home_team))]
        
        if h2h_df.empty:
            st.warning(f"Nessuna partita trovata tra {h2h_home_team} e {h2h_away_team}.")
        else:
            st.write(f"Analisi basata su **{len(h2h_df)}** scontri diretti tra {h2h_home_team} e {h2h_away_team}.")

            st.subheader(f"Statistiche H2H Complete tra {h2h_home_team} e {h2h_away_team} ({len(h2h_df)} partite)")
            
            st.subheader("Media Gol (H2H)")
            df_h2h_goals = h2h_df.copy()
            
            avg_ht_goals = (df_h2h_goals["Gol_Home_HT"] + df_h2h_goals["Gol_Away_HT"]).mean()
            avg_ft_goals = (df_h2h_goals["Gol_Home_FT"] + df_h2h_goals["Gol_Away_FT"]).mean()
            avg_sh_goals = (df_h2h_goals["Gol_Home_FT"] + df_h2h_goals["Gol_Away_FT"] - df_h2h_goals["Gol_Home_HT"] - df_h2h_goals["Gol_Away_HT"]).mean()
            st.table(pd.DataFrame({
                "Periodo": ["HT", "FT", "SH"],
                "Media Gol": [f"{avg_ht_goals:.2f}", f"{avg_ft_goals:.2f}", f"{avg_sh_goals:.2f}"]
            }))
            
            mostra_risultati_esatti(h2h_df, "risultato_ht", f"HT H2H ({len(h2h_df)})")
            mostra_risultati_esatti(h2h_df, "risultato_ft", f"FT H2H ({len(h2h_df)})")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"WinRate HT H2H ({len(h2h_df)})")
                df_winrate_ht_h2h = calcola_winrate(h2h_df, "risultato_ht")
                styled_df_ht = df_winrate_ht_h2h.style.background_gradient(cmap='RdYlGn', subset=['WinRate %'])
                st.dataframe(styled_df_ht)
            with col2:
                st.subheader(f"WinRate FT H2H ({len(h2h_df)})")
                df_winrate_ft_h2h = calcola_winrate(h2h_df, "risultato_ft")
                styled_df_ft = df_winrate_ft_h2h.style.background_gradient(cmap='RdYlGn', subset=['WinRate %'])
                st.dataframe(styled_df_ft)
            
            st.subheader(f"Doppia Chance (H2H) ({len(h2h_df)})")
            col1, col2 = st.columns(2)
            with col1:
                st.write("### HT")
                df_dc_ht_h2h = calcola_double_chance(h2h_df, 'ht')
                styled_df = df_dc_ht_h2h.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            with col2:
                st.write("### FT")
                df_dc_ft_h2h = calcola_double_chance(h2h_df, 'ft')
                styled_df = df_dc_ft_h2h.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)

            col1, col2 = st.columns(2)
            df_h2h_goals["tot_goals_ht"] = df_h2h_goals["Gol_Home_HT"] + df_h2h_goals["Gol_Away_HT"]
            df_h2h_goals["tot_goals_ft"] = df_h2h_goals["Gol_Home_FT"] + df_h2h_goals["Gol_Away_FT"]

            with col1:
                st.subheader(f"Over Goals HT H2H ({len(h2h_df)})")
                over_ht_data = []
                for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
                    count = (df_h2h_goals["tot_goals_ht"] > t).sum()
                    perc = round((count / len(df_h2h_goals)) * 100, 2)
                    odd_min = round(100 / perc, 2) if perc > 0 else "-"
                    over_ht_data.append([f"Over {t} HT", count, perc, odd_min])
                df_over_ht = pd.DataFrame(over_ht_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
                styled_over_ht = df_over_ht.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_over_ht)

            with col2:
                st.subheader(f"Over Goals FT H2H ({len(h2h_df)})")
                over_ft_data = []
                for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
                    count = (df_h2h_goals["tot_goals_ft"] > t).sum()
                    perc = round((count / len(df_h2h_goals)) * 100, 2)
                    odd_min = round(100 / perc, 2) if perc > 0 else "-"
                    over_ft_data.append([f"Over {t} FT", count, perc, odd_min])
                df_over_ft = pd.DataFrame(over_ft_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
                styled_over_ft = df_over_ft.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_over_ft)
            
            st.subheader(f"BTTS (H2H) ({len(h2h_df)})")
            col1, col2 = st.columns(2)
            with col1:
                st.write("### HT")
                df_btts_ht_h2h = calcola_btts_ht(h2h_df)
                styled_df = df_btts_ht_h2h.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            with col2:
                st.write("### FT")
                df_btts_ft_h2h = calcola_btts_ft(h2h_df)
                styled_df = df_btts_ft_h2h.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
                
            st.subheader(f"Multi Gol (H2H) ({len(h2h_df)})")
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Casa")
                styled_df = calcola_multi_gol(h2h_df, "Gol_Home_FT", "Home").style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            with col2:
                st.write("### Trasferta")
                styled_df = calcola_multi_gol(h2h_df, "Gol_Away_FT", "Away").style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)

            st.subheader(f"First to Score (H2H) ({len(h2h_df)})")
            styled_df = calcola_first_to_score(h2h_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            
            st.subheader(f"First to Score + Risultato Finale (H2H) ({len(h2h_df)})")
            styled_df = calcola_first_to_score_outcome(h2h_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)

            st.subheader(f"First to Score + Risultato Prossimo Gol (H2H) ({len(h2h_df)})")
            styled_df = calcola_first_to_score_next_goal_outcome(h2h_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            
            st.subheader(f"To Score (H2H) ({len(h2h_df)})")
            styled_df = calcola_to_score(h2h_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            
            st.subheader(f"Clean Sheet (H2H) ({len(h2h_df)})")
            styled_df = calcola_clean_sheet(h2h_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            
            st.subheader(f"Combo Markets (H2H) ({len(h2h_df)})")
            styled_df = calcola_combo_stats(h2h_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            
            st.subheader(f"Analisi Rimonte (H2H) ({len(h2h_df)})")
            rimonte_stats, squadre_rimonte = calcola_rimonte(h2h_df, "H2H")
            if not rimonte_stats.empty:
                styled_df = rimonte_stats.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
                
                st.markdown("**Squadre che hanno effettuato rimonte:**")
                for tipo, squadre in squadre_rimonte.items():
                    if squadre:
                        st.markdown(f"**{tipo}:** {', '.join(squadre)}")
            else:
                st.warning("Nessuna rimonta trovata nel dataset filtrato.")

# --- SEZIONE 6: Backtesting Strategie ---
st.subheader("6. Backtesting Strategie")
st.write("Testa una strategia di scommesse sui dati filtrati.")

with st.expander("Configura e avvia il Backtest"):
    
    if filtered_df.empty:
        st.warning("Il DataFrame filtrato è vuoto, non è possibile eseguire il backtest.")
    else:
        def esegui_backtest(df_to_analyze, market, strategy, stake):
            
            market_map = {
                "1 (Casa)": ("Odd_Home", lambda row: row["Gol_Home_FT"] > row["Gol_Away_FT"]),
                "X (Pareggio)": ("Odd_Draw", lambda row: row["Gol_Home_FT"] == row["Gol_Away_FT"]),
                "2 (Trasferta)": ("Odd__Away", lambda row: row["Gol_Home_FT"] < row["Gol_Away_FT"]),
                "Over 2.5 FT": ("Odd_over_2.5", lambda row: (row["Gol_Home_FT"] + row["Gol_Away_FT"]) > 2.5),
                "BTTS SI FT": ("BTTS_SI", lambda row: (row["Gol_Home_FT"] > 0 and row["Gol_Away_FT"] > 0))
            }
            
            odd_col, win_condition = market_map[market]
            
            required_cols = [odd_col, "risultato_ft", "Gol_Home_FT", "Gol_Away_FT"]
            for col in required_cols:
                if col not in df_to_analyze.columns:
                    st.error(f"Impossibile eseguire il backtest: la colonna '{col}' non è presente nel dataset.")
                    return 0, 0, 0, 0.0, 0.0, 0.0, 0.0
            
            vincite = 0
            perdite = 0
            profit_loss = 0.0
            numero_scommesse = 0
            
            df_clean = df_to_analyze.dropna(subset=required_cols).copy()
            
            for _, row in df_clean.iterrows():
                try:
                    odd = row[odd_col]
                    
                    if odd > 0:
                        is_winning = win_condition(row)
                        
                        if strategy == "Back":
                            if is_winning:
                                vincite += 1
                                profit_loss += (odd - 1) * stake
                            else:
                                perdite += 1
                                profit_loss -= stake
                        elif strategy == "Lay":
                            if is_winning:
                                perdite += 1
                                profit_loss -= (odd - 1) * stake
                            else:
                                vincite += 1
                                profit_loss += stake
                        
                        numero_scommesse += 1
                
                except (ValueError, KeyError):
                    continue

            investimento_totale = numero_scommesse * stake
            roi = (profit_loss / investimento_totale) * 100 if investimento_totale > 0 else 0
            win_rate = (vincite / numero_scommesse) * 100 if numero_scommesse > 0 else 0
            odd_minima = 100 / win_rate if win_rate > 0 else 0
            
            return vincite, perdite, numero_scommesse, profit_loss, roi, win_rate, odd_minima

        backtest_market = st.selectbox(
            "Seleziona un mercato da testare",
            ["1 (Casa)", "X (Pareggio)", "2 (Trasferta)", "Over 2.5 FT", "BTTS SI FT"]
        )
        backtest_strategy = st.selectbox(
            "Seleziona la strategia",
            ["Back", "Lay"]
        )
        stake = st.number_input("Stake per scommessa", min_value=1.0, value=1.0, step=0.5)
        
        if st.button("Avvia Backtest"):
            vincite, perdite, numero_scommesse, profit_loss, roi, win_rate, odd_minima = esegui_backtest(filtered_df, backtest_market, backtest_strategy, stake)
            
            if numero_scommesse > 0:
                col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                col_met1.metric("Numero Scommesse", numero_scommesse)
                col_met2.metric("Vincite", vincite)
                col_met3.metric("Perdite", perdite)
                col_met4.metric("Profitto/Perdita", f"{profit_loss:.2f} €")
                
                col_met5, col_met6 = st.columns(2)
                col_met5.metric("ROI", f"{roi:.2f} %")
                col_met6.metric("Win Rate", f"{win_rate:.2f} %")
                st.metric("Odd Minima per profitto", f"{odd_minima:.2f}")
            elif numero_scommesse == 0:
                st.info("Nessuna scommessa idonea trovata con i filtri e il mercato selezionati.")

