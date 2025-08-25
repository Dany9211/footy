import streamlit as st
import pandas as pd
import numpy as np
import datetime # Importazione necessaria per lavorare con le date
from lifelines import KaplanMeierFitter # Importazione della libreria lifelines

st.set_page_config(page_title="Analisi Campionati Next Gol e stats live", layout="wide")
st.title("Analisi Tabella 23agosto2023")

# --- Funzione per il caricamento del file CSV ---
@st.cache_data
def load_data(uploaded_file):
    """
    Carica i dati da un file CSV caricato dall'utente.
    Tenta diverse strategie di parsing per gestire potenziali errori
    e fornisce feedback specifico in caso di fallimento.
    """
    if uploaded_file is not None:
        # Resetta il puntatore del file per tentativi di lettura multipli
        uploaded_file.seek(0)
        
        # Strategia 1: Delimitatore ';', codifica UTF-8, salta righe malformate
        try:
            df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8', on_bad_lines='skip', header=0)
            # Verifica se il DataFrame è stato letto correttamente (non vuoto e con più di una colonna)
            if not df.empty and len(df.columns) > 1:
                st.success(f"File CSV caricato con successo (delimitatore ';', codifica utf-8). Colonne: {df.columns.tolist()}")
                return df
            uploaded_file.seek(0) # Resetta per il prossimo tentativo
        except Exception as e:
            st.error(f"Errore di caricamento (';', utf-8): {e}. Tentativo successivo...")
            uploaded_file.seek(0) # Resetta per il prossimo tentativo

        # Strategia 2: Delimitatore ';', codifica Latin-1, salta righe malformate
        try:
            df = pd.read_csv(uploaded_file, sep=';', encoding='latin1', on_bad_lines='skip', header=0)
            if not df.empty and len(df.columns) > 1:
                st.success(f"File CSV caricato con successo (delimitatore ';', codifica latin1). Colonne: {df.columns.tolist()}")
                return df
            uploaded_file.seek(0) # Resetta per il prossimo tentativo
        except Exception as e:
            st.error(f"Errore di caricamento (';', latin1): {e}. Tentativo successivo...")
            uploaded_file.seek(0) # Resetta per il prossimo tentativo

        # Strategia 3: Delimitatore ',', codifica UTF-8, usa il motore Python, salta righe malformate
        try:
            df = pd.read_csv(uploaded_file, sep=',', encoding='utf-8', on_bad_lines='skip', engine='python', header=0)
            if not df.empty and len(df.columns) > 1:
                st.success(f"File CSV caricato con successo (delimitatore ',', codifica utf-8, motore python). Colonne: {df.columns.tolist()}")
                return df
            uploaded_file.seek(0) # Resetta per il prossimo tentativo
        except Exception as e:
            st.error(f"Errore di caricamento (',', utf-8, python engine): {e}. Tentativo successivo...")
            uploaded_file.seek(0) # Resetta per il prossimo tentativo

        # Strategia 4: Rilevamento automatico del delimitatore, motore Python, salta righe malformate
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine='python', on_bad_lines='skip', header=0)
            if not df.empty and len(df.columns) > 1:
                st.success(f"File CSV caricato con successo (delimitatore auto-rilevato, motore python). Colonne: {df.columns.tolist()}")
                return df
            uploaded_file.seek(0) # Resetta per il prossimo tentativo
        except Exception as e:
            st.error(f"Errore di caricamento (auto-delimitatore, python engine): {e}. Tentativo successivo...")
            uploaded_file.seek(0) # Resetta per el prossimo tentativo

        # Se tutte le strategie falliscono
        st.error("Impossibile leggere il file CSV con le strategie di parsing automatiche. Controlla attentamente il formato del file, il delimitatore (punto e virgola, virgola o altro) e la codifica.")
        return pd.DataFrame()
    return pd.DataFrame()

# --- Funzione per convertire stringhe con virgola in float ---
def convert_to_float(series):
    # Converte in stringa prima di sostituire, per gestire vari tipi di input
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

# --- Pre-elaborazione e pulizia dati ---

# Conversione della colonna 'Data' in formato datetime
if 'Data' in df.columns:
    df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y %H:%M', errors='coerce')
    df = df.dropna(subset=['Data']) # Rimuovi righe con date non valide
else:
    st.error("Colonna 'Data' non trovata. Assicurati che il nome della colonna sia corretto (sensibile alle maiuscole).")

# Gestione specifica della colonna 'Anno'
if 'Anno' in df.columns:
    # Converti 'Anno' in numerico, trasformando errori in NaN
    df['Anno'] = pd.to_numeric(df['Anno'], errors='coerce')
    
    # Logica per convertire anni a 2 cifre (es. 15 -> 2015) - da attivare se necessario
    # Se la tua colonna 'Anno' contiene solo le ultime due cifre dell'anno (es. 15 per 2015)
    # e vuoi convertirle in anni completi (es. 2015), puoi usare la seguente logica:
    # def convert_two_digit_year(year):
    #     if pd.isna(year):
    #         return year
    #     year_int = int(year)
    #     if year_int < 100: # Assumendo che gli anni a 2 cifre siano dal 2000s
    #         return year_int + 2000
    #     return year_int
    # df['Anno'] = df['Anno'].apply(convert_two_digit_year)

    df = df.dropna(subset=['Anno']) # Rimuovi righe dove 'Anno' non è stato convertito correttamente
else:
    st.error("Colonna 'Anno' non trovata. Assicurati che il nome della colonna sia corretto (sensibile alle maiuscole).")


# Lista di tutte le colonne che dovrebbero essere numeriche e che potrebbero avere virgole come decimali
all_numeric_cols_with_comma = [
    "Odd_Home", "Odd_Draw", "Odd__Away", "Odd_Over_0.5", "Odd_over_1.5", 
    "Odd_over_2.5", "Odd_Over_3.5", "Odd_Over_4.5", "Odd_Under_0.5", 
    "Odd_Under_1.5", "Odd_Under_2.5", "Odd_Under_3.5", "Odd_Under_4.5",
    "elohomeo", "eloawayo", "formah", "formaa", "suth", "suth1", "suth2",
    "suta", "suta1", "suta2", "sutht", "sutht1", "sutht2", "sutat", "sutat1", "sutat2",
    "corh", "corh1", "corh2", "cora", "cora1", "cora2", "yellowh", "yellowh1", "yellowh2",
    "yellowa", "yellowa1", "yellowa2", "ballph", "ballph1", "ballph2", "ballpa", "ballpa1", "ballpa2"
]

# Applica convert_to_float a tutte queste colonne
for col in all_numeric_cols_with_comma:
    if col in df.columns:
        df[col] = convert_to_float(df[col])

# Conversione di altre colonne numeriche chiave che non dovrebbero avere virgole (es. Gol, Giornata)
# Queste dovrebbero essere già gestite da pd.to_numeric con errors='coerce' se non sono già numeri
other_int_cols = ["Gol_Home_FT", "Gol_Away_FT", "Gol_Home_HT", "Gol_Away_HT", 
                  "Home_Pos_Tot", "Away_Pos_Tot", "Home_Pos_H", "Away_Pos_A", "Giornata", "BTTS_SI"]

for col in other_int_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64') # Usa Int64 per gestire NaN in interi

# --- Aggiunta colonne risultato_ft e risultato_ht ---
if "Gol_Home_FT" in df.columns and "Gol_Away_FT" in df.columns:
    df["risultato_ft"] = df["Gol_Home_FT"].astype(str) + "-" + df["Gol_Away_FT"].astype(str)
if "Gol_Home_HT" in df.columns and "Gol_Away_HT" in df.columns:
    df["risultato_ht"] = df["Gol_Home_HT"].astype(str) + "-" + df["Gol_Away_HT"].astype(str)

filters = {}

# --- FILTRI INIZIALI ---
st.sidebar.header("Filtri Dati")

# Filtro League (Campionato) - Deve essere il primo per filtrare le squadre
if "League" in df.columns:
    leagues = ["Tutte"] + sorted(df["League"].dropna().unique())
    selected_league = st.sidebar.selectbox("Seleziona Campionato", leagues)
    if selected_league != "Tutte":
        filters["League"] = selected_league
    
    # Crea un DataFrame temporaneo per filtrare le squadre in base al campionato
    if selected_league != "Tutte":
        filtered_teams_df = df[df["League"] == selected_league]
    else:
        filtered_teams_df = df.copy()
else:
    filtered_teams_df = df.copy()
    selected_league = "Tutte"
    st.sidebar.error("Colonna 'League' non trovata. Il filtro per campionato non sarà disponibile.")


# Filtro Anno
if "Anno" in df.columns:
    df_anni_numeric = df["Anno"].dropna()
    if not df_anni_numeric.empty:
        # Ottieni tutti gli anni unici presenti nel dataset e ordinali in ordine decrescente
        all_unique_years = sorted(df_anni_numeric.unique().astype(int), reverse=True)
        
        # Opzioni per intervalli di anni dinamici (es. Ultimi 3 anni)
        dynamic_range_options_labels = []
        # Aggiungi "Anno Corrente"
        dynamic_range_options_labels.append("Anno Corrente")
        # Aggiungi "Ultimi X anni" per X da 2 a 10
        for num_years in range(2, 11): # Genera da 2 a 10
            label = f"Ultimi {num_years} anni"
            dynamic_range_options_labels.append(label)

        # Combina "Tutti", gli intervalli dinamici e gli anni individuali per la visualizzazione
        display_options = ["Tutti"] + dynamic_range_options_labels + [str(y) for y in all_unique_years]
        
        selected_anno_display = st.sidebar.selectbox("Seleziona Anno", display_options)

        if selected_anno_display == "Tutti":
            # Se è "Tutti", rimuovi il filtro se presente
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
            # Se è stata selezionata un'opzione "Ultimi X anni"
            num_years_back = int(selected_anno_display.split(' ')[1])
            
            # Prendi gli ultimi 'num_years_back' anni dal dataset disponibile
            if len(all_unique_years) >= num_years_back:
                # Seleziona i 'num_years_back' anni più recenti dalla lista ordinata
                years_to_filter = all_unique_years[:num_years_back]
                min_year_to_filter = min(years_to_filter)
                max_year_to_filter = max(years_to_filter)
                filters["Anno"] = (min_year_to_filter, max_year_to_filter)
            elif all_unique_years: # Se ci sono anni, ma meno di quelli richiesti, usa tutti quelli disponibili
                min_year_to_filter = min(all_unique_years)
                max_year_to_filter = max(all_unique_years)
                filters["Anno"] = (min_year_to_filter, max_year_to_filter)
                st.sidebar.info(f"Il dataset contiene solo {len(all_unique_years)} anni. Verranno utilizzati tutti gli anni disponibili per '{selected_anno_display}'.")
            else: # Nessun anno nel dataset
                st.sidebar.info(f"Nessun dato disponibile per '{selected_anno_display}' nel dataset caricato.")
                if "Anno" in filters:
                    del filters["Anno"]
        else:
            # Se è stato selezionato un singolo anno (ad esempio, "2023")
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


# Filtro Giornata
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


# --- FILTRI SQUADRE (ora dinamici) ---
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


# --- NUOVO FILTRO: Risultato HT ---
if "risultato_ht" in df.columns:
    ht_results = sorted(df["risultato_ht"].dropna().unique())
    selected_ht_results = st.sidebar.multiselect("Seleziona Risultato HT", ht_results, default=None)
    if selected_ht_results:
        filters["risultato_ht"] = selected_ht_results
else:
    st.sidebar.error("Colonna 'risultato_ht' non trovata. Il filtro per risultato HT non sarà disponibile.")


# --- FUNZIONE per filtri range ---
def add_range_filter(col_name, label=None):
    if col_name in df.columns:
        # Assicurati che la colonna sia numerica prima di procedere
        numeric_col_series = convert_to_float(df[col_name])
        
        # Evita di calcolare min/max su serie completamente NaN
        if not numeric_col_series.isnull().all():
            col_min = float(numeric_col_series.min(skipna=True))
            col_max = float(numeric_col_series.max(skipna=True))
            
            st.sidebar.write(f"Range attuale {label or col_name}: {col_min} - {col_max}")
            # Aggiungi chiavi uniche per gli input di testo
            min_val_input = st.sidebar.text_input(f"Min {label or col_name}", key=f"min_{col_name}", value="")
            max_val_input = st.sidebar.text_input(f"Max {label or col_name}", key=f"max_{col_name}", value="")
            
            if min_val_input.strip() != "" and max_val_input.strip() != "":
                try:
                    # Converti a float qui e memorizza come float
                    filters[col_name] = (float(min_val_input), float(max_val_input))
                except ValueError:
                    st.sidebar.error(f"Valori non validi per {label or col_name}. Inserisci numeri.")
                    # Se i valori non sono validi, assicurati che il filtro non venga impostato
                    if col_name in filters:
                        del filters[col_name]
            else:
                # Se i campi di input sono vuoti, rimuovi il filtro se esiste
                if col_name in filters:
                    del filters[col_name]
        else:
            st.sidebar.info(f"Colonna '{label or col_name}' non contiene valori numerici validi per il filtro.")
            if col_name in filters: # Rimuovi anche se la colonna è tutta NaN
                del filters[col_name]
    else:
        st.sidebar.error(f"Colonna '{label or col_name}' non trovata per il filtro.")
        if col_name in filters: # Rimuovi anche se la colonna non esiste
            del filters[col_name]


st.sidebar.header("Filtri Quote")
for col in ["Odd_Home", "Odd_Draw", "Odd__Away"]:
    add_range_filter(col)
for col in ["Odd_Over_0.5", "Odd_over_1.5", "Odd_over_2.5", "Odd_Over_3.5", "Odd_Over_4.5",
            "Odd_Under_0.5", "Odd_Under_1.5", "Odd_Under_2.5", "Odd_Under_3.5", "Odd_Under_4.5", "BTTS_SI"]:
    add_range_filter(col)


# --- APPLICA FILTRI AL DATAFRAME PRINCIPALE ---
filtered_df = df.copy()
for col, val in filters.items():
    
    # Per i filtri di range numerici (Giornata, Quote, ecc.)
    if col in ["Odd_Home", "Odd_Draw", "Odd__Away", "Odd_Over_0.5", "Odd_over_1.5", 
                "Odd_over_2.5", "Odd_Over_3.5", "Odd_Over_4.5", "Odd_Under_0.5", 
                "Odd_Under_1.5", "Odd_Under_2.5", "Odd_Under_3.5", "Odd_Under_4.5", 
                "BTTS_SI", "Giornata"]:
        
        # CRUCIAL: Ensure val is a tuple for range filters
        if not isinstance(val, tuple) or len(val) != 2:
            st.error(f"Errore: il valore del filtro per la colonna '{col}' ({val}) non è un intervallo numerico valido. Ignoro il filtro.")
            continue

        # Converte la serie da filtrare in float, gestendo gli errori
        series_to_filter = convert_to_float(filtered_df[col])
        
        # Assicurati che i limiti del filtro siano float validi
        try:
            lower_bound = float(val[0])
            upper_bound = float(val[1])
        except (ValueError, TypeError) as e:
            st.error(f"Errore: i valori del filtro per la colonna '{col}' ({val[0]}, {val[1]}) non sono convertibili in numeri. Dettagli: {e}. Ignoro il filtro.")
            continue # Salta questo filtro se i limiti non sono validi

        # Applica il filtro. La serie è già numerica (float o NaN) e i limiti sono float.
        mask = series_to_filter.between(lower_bound, upper_bound)
        filtered_df = filtered_df[mask.fillna(True)]
    elif col == "risultato_ht":
        # Per i filtri multiselect, val è una lista di stringhe
        if isinstance(val, list):
            filtered_df = filtered_df[filtered_df[col].isin(val)]
        else:
            st.error(f"Errore: il valore del filtro per la colonna '{col}' non è una lista come previsto. Ignoro il filtro.")
            continue
    elif col == "Anno": # Gestione specifica per il filtro Anno
        # Se 'val' è una tupla, significa che è un intervallo di anni (es. "Ultimi 5 anni")
        if isinstance(val, tuple) and len(val) == 2:
            lower_bound, upper_bound = val
            series_to_filter = pd.to_numeric(filtered_df[col], errors='coerce')
            mask = series_to_filter.between(lower_bound, upper_bound)
            filtered_df = filtered_df[mask.fillna(True)]
        else: # Altrimenti, è un singolo anno selezionato
            filtered_df = filtered_df[filtered_df[col] == val]
    else: # Per i filtri a selezione singola (es. League, Home_Team, Away_Team)
        filtered_df = filtered_df[filtered_df[col] == val]

st.subheader("Dati Filtrati")
st.write(f"**Righe visualizzate:** {len(filtered_df)}")

# --- NUOVA SEZIONE: Riepilogo Risultati per Anno ---
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
# --- FINE NUOVA SEZIONE ---

st.dataframe(filtered_df.head(50))


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
            # Home segna per primo
            if home_vince:
                risultati["Casa Segna Primo e Vince"] += 1
            else:
                risultati["Casa Segna Primo e Non Vince"] += 1
        elif min_away_goal < min_home_goal:
            # Away segna per primo
            if away_vince:
                risultati["Trasferta Segna Prima e Vince"] += 1
            else:
                risultati["Trasferta Segna Prima e Non Vince"] += 1
        else:
            # Nessun gol
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
            
        # Ordina tutti i gol per minuto
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

        # Considera solo i gol segnati nel secondo tempo (minuto > 45)
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
        # Restituisce un DataFrame vuoto con le colonne attese
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
        # Restituisce un DataFrame vuoto con le colonne attese
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

        # Considera solo i gol segnati nel primo tempo (minuto <= 45)
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

    if df_valid.empty: # Aggiunto controllo per DataFrame vuoto
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
def mostra_distribuzione_timeband(df_to_analyze, min_start_display=0): # Aggiunto parametro
    if df_to_analyze.empty:
        st.warning("Il DataFrame per l'analisi a 15 minuti è vuoto.")
        return

    all_intervalli = [(0, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90), (91, 150)]
    all_label_intervalli = ["0-15", "16-30", "31-45", "46-60", "61-75", "76-90", "90+"]

    risultati = []
    total_matches = len(df_to_analyze)
    
    for i, ((start_interval, end_interval), label) in enumerate(zip(all_intervalli, all_label_intervalli)):
        # Salta gli intervalli che terminano prima del minuto di inizio visualizzazione
        if end_interval < min_start_display:
            continue

        partite_con_gol = 0
        for _, row in df_to_analyze.iterrows():
            gol_home_str = str(row.get("Minutaggio_Gol_Home", ""))
            gol_away_str = str(row.get("Minutaggio_gol_Away", ""))

            gol_home = [int(x) for x in gol_home_str.split(";") if x.isdigit()]
            gol_away = [int(x) for x in gol_away_str.split(";") if x.isdigit()]
            
            # Conta i gol solo se cadono all'interno dell'intervallo corrente E sono dopo o al min_start_display
            goals_in_relevant_part_of_interval = [
                g for g in (gol_home + gol_away) 
                if max(start_interval, min_start_display) <= g <= end_interval
            ]
            
            if goals_in_relevant_part_of_interval:
                partite_con_gol += 1
        
        perc = round((partite_con_gol / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        risultati.append([label, partite_con_gol, perc, odd_min])
    
    if not risultati: # Se tutti gli intervalli sono stati saltati
        st.info(f"Nessun intervallo di tempo rilevante dopo il minuto {min_start_display} per l'analisi a 15 minuti.")
        return

    df_result = pd.DataFrame(risultati, columns=["Timeframe", "Partite con Gol", "Percentuale %", "Odd Minima"])
    styled_df = df_result.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
    st.dataframe(styled_df)

# --- NUOVA FUNZIONE RIUTILIZZABILE PER DISTRIBUZIONE TIMEBAND (5 MIN) ---
def mostra_distribuzione_timeband_5min(df_to_analyze, min_start_display=0): # Aggiunto parametro
    if df_to_analyze.empty:
        return
    all_intervalli = [(0,5), (6,10), (11,15), (16,20), (21,25), (26,30), (31,35), (36,40), (41,45), (46,50), (51,55), (56,60), (61,65), (66,70), (71,75), (76,80), (81,85), (86,90), (91, 150)]
    all_label_intervalli = ["0-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50", "51-55", "56-60", "61-65", "66-70", "71-75", "76-80", "81-85", "86-90", "90+"]
    risultati = []
    total_matches = len(df_to_analyze)
    for (start_interval, end_interval), label in zip(all_intervalli, all_label_intervalli):
        # Salta gli intervalli che terminano prima del minuto di inizio visualizzazione
        if end_interval < min_start_display:
            continue

        partite_con_gol = 0
        for _, row in df_to_analyze.iterrows():
            gol_home = [int(x) for x in str(row.get("Minutaggio_Gol_Home", "")).split(";") if x.isdigit()]
            gol_away = [int(x) for x in str(row.get("Minutaggio_gol_Away", "")).split(";") if x.isdigit()]
            
            # Conta i gol solo se cadono all'interno dell'intervallo corrente E sono dopo o al min_start_display
            goals_in_relevant_part_of_interval = [
                g for g in (gol_home + gol_away) 
                if max(start_interval, min_start_display) <= g <= end_interval
            ]

            if goals_in_relevant_part_of_interval:
                partite_con_gol += 1
        perc = round((partite_con_gol / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        risultati.append([label, partite_con_gol, perc, odd_min])
    
    if not risultati: # Se tutti gli intervalli sono stati saltati
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
        elif next_away_goal < next_home_goal: # Correzione qui
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
    
    # Aggiungi colonne per i gol HT e FT
    # Queste colonne dovrebbero essere già numeriche
    
    def check_comeback(row):
        # Rimonta Home
        if row["Gol_Home_HT"] < row["Gol_Away_HT"] and row["Gol_Home_FT"] > row["Gol_Away_FT"]:
            return "Completa - Home"
        if row["Gol_Home_HT"] < row["Gol_Away_HT"] and row["Gol_Home_FT"] == row["Gol_Away_FT"]:
            return "Parziale - Home"
        # Rimonta Away
        if row["Gol_Away_HT"] < row["Gol_Home_HT"] and row["Gol_Away_FT"] > row["Gol_Home_FT"]:
            return "Completa - Away"
        if row["Gol_Away_HT"] < row["Gol_Home_HT"] and row["Gol_Away_FT"] == row["Gol_Home_FT"]:
            return "Parziale - Away"
        return "Nessuna"

    df_rimonte["rimonta"] = df_rimonte.apply(check_comeback, axis=1)
    
    # Filtra e conta i risultati
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
    
    # Crea la lista di squadre per ogni tipo di rimonta
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

    # Poiché `df_to_analyze` qui è già `df_target`, che è stato filtrato in base a `risultati_correnti`
    # e `start_min`, dobbiamo solo verificare se entrambe le squadre hanno segnato a fine partita.
    for _, row in df_to_analyze.iterrows():
        gol_home_ft = int(row.get("Gol_Home_FT", 0))
        gol_away_ft = int(row.get("Gol_Away_FT", 0))
        
        if (gol_home_ft > 0 and gol_away_ft > 0):
            btts_si_count += 1

    no_btts_count = total_matches - btts_si_count # Calcolato qui dopo il loop

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
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"]) # Ensure columns are present
    
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
        return pd.DataFrame(columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"]) # Ensure columns are present
        
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
        return pd.DataFrame(columns=[f"Mercato ({titolo})", "Conteggio", "Percentuale %", "Odd Minima"]) # Ensure columns are present
    
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

# --- NUOVA FUNZIONE PER CALCOLARE LA PROBABILITÀ DEL PROSSIMO GOL CON MODELLO HAZARD ---
def calcola_probabilita_prossimo_gol_hazard(df_to_analyze, minuto_corrente):
    """
    Calcola la probabilità cumulativa di segnare un gol a partire da un
    determinato minuto, utilizzando un modello di rischio (Kaplan-Meier).

    Args:
        df_to_analyze (pd.DataFrame): Il DataFrame filtrato su cui eseguire l'analisi.
        minuto_corrente (int): Il minuto della partita da cui iniziare la previsione.

    Returns:
        pd.DataFrame: Un DataFrame con la probabilità di un gol nel tempo.
    """
    if df_to_analyze.empty:
        return pd.DataFrame()

    tempi_tra_gol = []
    for _, row in df_to_analyze.iterrows():
        gol_home = [int(x) for x in str(row.get("Minutaggio_Gol_Home", "")).split(";") if x.isdigit()]
        gol_away = [int(x) for x in str(row.get("Minutaggio_gol_Away", "")).split(";") if x.isdigit()]

        all_goals_times = sorted(gol_home + gol_away)

        # Considera solo i gol che si verificano dopo il minuto corrente
        all_goals_times = [t for t in all_goals_times if t >= minuto_corrente]

        if not all_goals_times:
            # Nessun gol dopo il minuto corrente, il tempo di "sopravvivenza" è fino a 90 minuti
            tempi_tra_gol.append(90 - minuto_corrente)
        else:
            last_time = minuto_corrente
            for t in all_goals_times:
                # Registra l'intervallo tra i gol
                tempi_tra_gol.append(t - last_time)
                last_time = t
            # Registra anche l'intervallo dall'ultimo gol alla fine della partita (censored)
            tempi_tra_gol.append(90 - last_time)
    
    # DataFrame per il modello di rischio
    df_tempi = pd.DataFrame({'time_to_event': tempi_tra_gol})

    # Filtra i tempi validi e positivi
    df_tempi = df_tempi[df_tempi['time_to_event'] > 0]
    if df_tempi.empty:
        return pd.DataFrame()

    # Calcola il modello di sopravvivenza (Kaplan-Meier)
    kmf = KaplanMeierFitter()
    kmf.fit(df_tempi['time_to_event'])

    # Calcola la probabilità di un gol tra il minuto corrente e la fine
    df_probabilita = pd.DataFrame()
    
    # Genera i minuti da minuto_corrente a 90
    minutes_for_prediction = list(range(minuto_corrente, 91))
    
    probabilities = []
    for minute_abs in minutes_for_prediction:
        # Calcola il tempo relativo dall'inizio della previsione
        time_relative = minute_abs - minuto_corrente
        if time_relative < 0: # Dovrebbe essere sempre >= 0
            probabilities.append(0)
            continue
        
        # Probabilità di sopravvivenza (nessun evento) fino a questo punto relativo
        s_t = kmf.survival_function_.loc[time_relative].iloc[0] if time_relative in kmf.survival_function_.index else \
              (kmf.survival_function_.loc[kmf.survival_function_.index.max()].iloc[0] if time_relative > kmf.survival_function_.index.max() else 1.0)
        
        # Probabilità di sopravvivenza (nessun evento) fino al punto relativo successivo
        s_t_plus_1 = kmf.survival_function_.loc[time_relative + 1].iloc[0] if (time_relative + 1) in kmf.survival_function_.index else \
                     (kmf.survival_function_.loc[kmf.survival_function_.index.max()].iloc[0] if (time_relative + 1) > kmf.survival_function_.index.max() else 0.0)
        
        # Evita divisione per zero
        if s_t == 0:
            prob_gol_in_minuto = 0
        else:
            # Probabilità condizionale che un gol si verifichi nel prossimo minuto, dato che non si è verificato fino a 'time_relative'
            prob_gol_in_minuto = (s_t - s_t_plus_1) / s_t
        
        probabilities.append(prob_gol_in_minuto * 100) # In percentuale
            
    df_probabilita['Minuto'] = minutes_for_prediction
    df_probabilita['Probabilita (%)'] = probabilities
    
    return df_probabilita


# SEZIONE 1: Analisi Timeband per Campionato
st.subheader("1. Analisi Timeband per Campionato")
if selected_league != "Tutte":
    df_league_only = df[df["League"] == selected_league]
    st.write(f"Analisi basata su **{len(df_league_only)}** partite del campionato **{selected_league}**.")
    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Distribuzione Gol per Timeframe (15min)**")
        mostra_distribuzione_timeband(df_league_only) # Chiamata senza min_start_display
    with col2:
        st.write("**Distribuzione Gol per Timeframe (5min)**")
        mostra_distribuzione_timeband_5min(df_league_only) # Chiamata senza min_start_display
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
        mostra_distribuzione_timeband(filtered_df) # Chiamata senza min_start_display
    with col2:
        st.write("**Distribuzione Gol per Timeframe (5min)**")
        mostra_distribuzione_timeband_5min(filtered_df) # Chiamata senza min_start_display
else:
    st.warning("Nessuna partita corrisponde ai filtri selezionati.")


# NUOVA SEZIONE: Statistiche Pre-Match Complete (Filtri Sidebar)
st.subheader("3. Analisi Pre-Match Completa (Filtri Sidebar)")
st.write(f"Analisi completa basata su **{len(filtered_df)}** partite, considerando tutti i filtri del menu a sinistra.")
if not filtered_df.empty:
    
    # Calcolo e visualizzazione media gol
    st.subheader("Media Gol (Pre-Match)")
    df_prematch_goals = filtered_df.copy()
    
    # Queste colonne dovrebbero essere già numeriche
    
    # Media gol HT
    avg_ht_goals = (df_prematch_goals["Gol_Home_HT"] + df_prematch_goals["Gol_Away_HT"]).mean()
    # Media gol FT
    avg_ft_goals = (df_prematch_goals["Gol_Home_FT"] + df_prematch_goals["Gol_Away_FT"]).mean()
    # Media gol SH (secondo tempo)
    avg_sh_goals = (df_prematch_goals["Gol_Home_FT"] + df_prematch_goals["Gol_Away_FT"] - df_prematch_goals["Gol_Home_HT"] - df_prematch_goals["Gol_Away_HT"]).mean()
    
    st.table(pd.DataFrame({
        "Periodo": ["HT", "FT", "SH"],
        "Media Gol": [f"{avg_ht_goals:.2f}", f"{avg_ft_goals:.2f}", f"{avg_sh_goals:.2f}"]
    }))

    # --- Expander per Statistiche HT ---
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
    
    # --- Nuove Expander per Statistiche SH ---
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

    # --- Expander per Statistiche FT ---
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
        # --- ANALISI DAL MINUTO (integrata) ---
        # Cursore unico per il range di minuti
        min_range = st.slider("Seleziona Range Minuti", 1, 90, (45, 90))
        start_min, end_min = min_range[0], min_range[1]

        # Assicurati che ht_results esista e non sia vuoto
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

            # Calcolo e visualizzazione media gol dinamica
            st.subheader("Media Gol (Dinamica)")
            df_target_goals = df_target.copy()
            
            # Queste colonne dovrebbero essere già numeriche
            
            # Media gol HT
            avg_ht_goals_dynamic = (df_target_goals["Gol_Home_HT"] + df_target_goals["Gol_Away_HT"]).mean()
            # Media gol FT
            avg_ft_goals_dynamic = (df_target_goals["Gol_Home_FT"] + df_target_goals["Gol_Away_FT"]).mean()
            # Media gol SH (secondo tempo)
            avg_sh_goals_dynamic = (df_target_goals["Gol_Home_FT"] + df_target_goals["Gol_Away_FT"] - df_target_goals["Gol_Home_HT"] - df_target_goals["Gol_Away_HT"]).mean()
            
            st.table(pd.DataFrame({
                "Periodo": ["HT", "FT", "SH"],
                "Media Gol": [f"{avg_ht_goals_dynamic:.2f}", f"{avg_ft_goals_dynamic:.2f}", f"{avg_sh_goals_dynamic:.2f}"]
            }))
            
            mostra_risultati_esatti(df_target, "risultato_ht", f"HT ({len(df_target)})")
            mostra_risultati_esatti(df_target, "risultato_ft", f"FT ({len(df_target)})")

            # WinRate
            st.subheader(f"WinRate (Dinamica) ({len(df_target)})")
            st.write("**HT:**")
            df_winrate_ht_dynamic = calcola_winrate(df_target, "risultato_ht")
            styled_df_ht = df_winrate_ht_dynamic.style.background_gradient(cmap='RdYlGn', subset=['WinRate %'])
            st.dataframe(styled_df_ht)
            st.write("**FT:**")
            df_winrate_ft_dynamic = calcola_winrate(df_target, "risultato_ft")
            styled_df_ft = df_winrate_ft_dynamic.style.background_gradient(cmap='RdYlGn', subset=['WinRate %'])
            st.dataframe(styled_df_ft)
            
            # Over Goals HT e FT
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
            
            # BTTS
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

            # Doppia Chance Dinamica
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

            # Multi Gol
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
            
            # First to Score nell'analisi dinamica (HT e FT)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"First to Score HT (Dinamica) ({len(df_target)})")
                styled_df = calcola_first_to_score_ht(df_target).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            with col2:
                st.subheader(f"First to Score FT (Dinamica) ({len(df_target)})")
                styled_df = calcola_first_to_score(df_target).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            
            # First to Score + Outcome Dinamica
            st.subheader(f"First to Score + Risultato Finale (Dinamica) ({len(df_target)})")
            styled_df = calcola_first_to_score_outcome(df_target).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            
            # First to Score + Next Goal Dinamica
            st.subheader(f"First to Score + Risultato Prossimo Gol (Dinamica) ({len(df_target)})")
            styled_df = calcola_first_to_score_next_goal_outcome(df_target).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            
            # To Score nell'analisi dinamica (HT e FT)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"To Score HT (Dinamica) ({len(df_target)})")
                styled_df = calcola_to_score_ht(df_target).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            with col2:
                st.subheader(f"To Score FT (Dinamica) ({len(df_target)})")
                styled_df = calcola_to_score(df_target).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_df)
            
            # Clean Sheet nell'analisi dinamica
            st.subheader(f"Clean Sheet (Dinamica) ({len(df_target)})")
            styled_df = calcola_clean_sheet(df_target).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            
            # Combo Markets nell'analisi dinamica
            st.subheader(f"Combo Markets (Dinamica) ({len(df_target)})")
            styled_df = calcola_combo_stats(df_target).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            
            # Next Goal nell'analisi dinamica
            st.subheader(f"Next Goal (Dinamica) ({len(df_target)})")
            styled_df = calcola_next_goal(df_target, start_min, end_min).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            
            # Analisi Rimonte Dinamica
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
            
            # Qui viene mostrata la timeband basata sull'analisi dinamica
            st.subheader("Distribuzione Gol per Timeframe (dinamica)")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**15min**")
                mostra_distribuzione_timeband(df_target, min_start_display=start_min) # Passa start_min
            with col2:
                st.write("**5min**")
                mostra_distribuzione_timeband_5min(df_target, min_start_display=start_min) # Passa start_min

            # --- NUOVA SEZIONE: Previsione Probabilità Prossimo Gol (Modello di Rischio) ---
            st.markdown("---")
            st.subheader("Previsione Probabilità Prossimo Gol (Modello di Rischio)")
            
            # Selezionare il minuto da cui iniziare l'analisi
            start_time_prediction = st.slider(
                "Minuto di inizio previsione per il prossimo gol (Modello di Rischio)", 
                min_value=1, 
                max_value=90, 
                value=start_min # Usa lo stesso minuto iniziale della tua analisi dinamica
            )
            
            # Esegui l'analisi di rischio sul DataFrame filtrato
            df_hazard_prediction = calcola_probabilita_prossimo_gol_hazard(df_target, start_time_prediction)
            
            if not df_hazard_prediction.empty:
                df_to_display_hazard = df_hazard_prediction.copy()
                df_to_display_hazard['Odd Minima'] = df_to_display_hazard['Probabilita (%)'].apply(lambda x: round(100/x, 2) if x > 0 else '-')
            
                st.dataframe(df_to_display_hazard)
                
                # Puoi anche visualizzare un grafico per una migliore comprensione
                st.line_chart(df_hazard_prediction.set_index('Minuto'))
            else:
                st.info("Nessun dato sufficiente per calcolare il modello di rischio per il prossimo gol.")
            # --- FINE NUOVA SEZIONE ---

    else:
        st.warning("Il dataset filtrato è vuoto o mancano le colonne necessarie per l'analisi.")

# --- SEZIONE 5: Analisi Head-to-Head (H2H) ---
st.subheader("5. Analisi Head-to-Head (H2H)")
st.write("Seleziona due squadre per analizzare i loro scontri diretti.")

# Recupera l'elenco completo di tutte le squadre disponibili nel dataset
all_teams = sorted(list(set(df['Home_Team'].dropna().unique()) | set(df['Away_Team'].dropna().unique())))
h2h_home_team = st.selectbox("Seleziona Squadra 1", ["Seleziona..."] + all_teams)
h2h_away_team = st.selectbox("Seleziona Squadra 2", ["Seleziona..."] + all_teams)

if h2h_home_team != "Seleziona..." and h2h_away_team != "Seleziona...":
    if h2h_home_team == h2h_away_team:
        st.warning("Seleziona due squadre diverse per l'analisi H2H.")
    else:
        # Filtra il DataFrame per trovare tutti i match tra le due squadre selezionate
        # NOTA: I filtri per le quote della sidebar non vengono applicati qui per avere il dataset H2H completo
        h2h_df = df[((df['Home_Team'] == h2h_home_team) & (df['Away_Team'] == h2h_away_team)) |
                    ((df['Home_Team'] == h2h_away_team) & (df['Away_Team'] == h2h_home_team))]
        
        if h2h_df.empty:
            st.warning(f"Nessuna partita trovata tra {h2h_home_team} e {h2h_away_team}.")
        else:
            st.write(f"Analisi basata su **{len(h2h_df)}** scontri diretti tra {h2h_home_team} e {h2h_away_team}.")

            # Esegui le stesse analisi pre-match, ma sul DataFrame H2H
            st.subheader(f"Statistiche H2H Complete tra {h2h_home_team} e {h2h_away_team} ({len(h2h_df)} partite)")
            
            # Media gol
            st.subheader("Media Gol (H2H)")
            df_h2h_goals = h2h_df.copy()
            
            avg_ht_goals = (df_h2h_goals["Gol_Home_HT"] + df_h2h_goals["Gol_Away_HT"]).mean()
            avg_ft_goals = (df_h2h_goals["Gol_Home_FT"] + df_h2h_goals["Gol_Away_FT"]).mean()
            avg_sh_goals = (df_h2h_goals["Gol_Home_FT"] + df_h2h_goals["Gol_Away_FT"] - df_h2h_goals["Gol_Home_HT"] - df_h2h_goals["Gol_Away_HT"]).mean()
            st.table(pd.DataFrame({
                "Periodo": ["HT", "FT", "SH"],
                "Media Gol": [f"{avg_ht_goals:.2f}", f"{avg_ft_goals:.2f}", f"{avg_sh_goals:.2f}"]
            }))
            
            # Risultati Esatti H2H
            mostra_risultati_esatti(h2h_df, "risultato_ht", f"HT H2H ({len(h2h_df)})")
            mostra_risultati_esatti(h2h_df, "risultato_ft", f"FT H2H ({len(h2h_df)})")

            # WinRate H2H
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
            
            # Doppia Chance H2H
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

            # Over Goals H2H
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
            
            # BTTS H2H
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
                
            # Multi Gol H2H
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

            # First to Score H2H
            st.subheader(f"First to Score (H2H) ({len(h2h_df)})")
            styled_df = calcola_first_to_score(h2h_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            
            # First to Score + Outcome H2H
            st.subheader(f"First to Score + Risultato Finale (H2H) ({len(h2h_df)})")
            styled_df = calcola_first_to_score_outcome(h2h_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)

            # First to Score + Next Goal H2H
            st.subheader(f"First to Score + Risultato Prossimo Gol (H2H) ({len(h2h_df)})")
            styled_df = calcola_first_to_score_next_goal_outcome(h2h_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            
            # To Score H2H
            st.subheader(f"To Score (H2H) ({len(h2h_df)})")
            styled_df = calcola_to_score(h2h_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            
            # Clean Sheet H2H
            st.subheader(f"Clean Sheet (H2H) ({len(h2h_df)})")
            styled_df = calcola_clean_sheet(h2h_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            
            # Combo Markets H2H
            st.subheader(f"Combo Markets (H2H) ({len(h2h_df)})")
            styled_df = calcola_combo_stats(h2h_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
            st.dataframe(styled_df)
            
            # Analisi Rimonte H2H
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

# Aggiungi un expander per contenere la logica di backtesting
with st.expander("Configura e avvia il Backtest"):
    
    if filtered_df.empty:
        st.warning("Il DataFrame filtrato è vuoto, non è possibile eseguire il backtest.")
    else:
        # Funzione per eseguire il backtest
        def esegui_backtest(df_to_analyze, market, strategy, stake):
            
            # Definizione dei mercati e delle colonne necessarie
            market_map = {
                "1 (Casa)": ("Odd_Home", lambda row: row["Gol_Home_FT"] > row["Gol_Away_FT"]),
                "X (Pareggio)": ("Odd_Draw", lambda row: row["Gol_Home_FT"] == row["Gol_Away_FT"]),
                "2 (Trasferta)": ("Odd__Away", lambda row: row["Gol_Home_FT"] < row["Gol_Away_FT"]),
                "Over 2.5 FT": ("Odd_over_2.5", lambda row: (row["Gol_Home_FT"] + row["Gol_Away_FT"]) > 2.5),
                "BTTS SI FT": ("BTTS_SI", lambda row: (row["Gol_Home_FT"] > 0 and row["Gol_Away_FT"] > 0))
            }
            
            odd_col, win_condition = market_map[market]
            
            # Controllo che le colonne necessarie esistano nel DataFrame
            required_cols = [odd_col, "risultato_ft", "Gol_Home_FT", "Gol_Away_FT"]
            for col in required_cols:
                if col not in df_to_analyze.columns:
                    st.error(f"Impossibile eseguire il backtest: la colonna '{col}' non è presente nel dataset.")
                    return 0, 0, 0, 0.0, 0.0, 0.0, 0.0
            
            vincite = 0
            perdite = 0
            profit_loss = 0.0
            numero_scommesse = 0
            
            # Rimuovi le righe con valori nulli nelle colonne chiave
            df_clean = df_to_analyze.dropna(subset=required_cols).copy()
            
            # Assicurati che le colonne quote e gol siano numeriche
            # Queste colonne dovrebbero essere già numeriche grazie alla pre-elaborazione
            
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
                    # Gestione di righe con dati mancanti o non validi
                    continue

            investimento_totale = numero_scommesse * stake
            roi = (profit_loss / investimento_totale) * 100 if investimento_totale > 0 else 0
            win_rate = (vincite / numero_scommesse) * 100 if numero_scommesse > 0 else 0
            odd_minima = 100 / win_rate if win_rate > 0 else 0
            
            return vincite, perdite, numero_scommesse, profit_loss, roi, win_rate, odd_minima

        # UI per il backtest
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
