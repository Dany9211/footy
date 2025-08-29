import streamlit as st
import pandas as pd
import numpy as np
import datetime # Importazione necessaria per lavorare con le date

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

def convert_to_float(series):
    return pd.to_numeric(series.astype(str).str.replace(",", "."), errors="coerce")

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

if 'Data' in df.columns:
    df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y %H:%M', errors='coerce')
    df = df.dropna(subset=['Data'])
else:
    st.error("Colonna 'Data' non trovata. Assicurati che il nome della colonna sia corretto (sensibile alle maioline).")

if 'Anno' in df.columns:
    df['Anno'] = pd.to_numeric(df['Anno'], errors='coerce')
    df = df.dropna(subset=['Anno'])
else:
    st.error("Colonna 'Anno' non trovata. Assicurati che il nome della colonna sia corretto (sensibile alle maioline).")

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

if "Gol_Home_FT" in df.columns and "Gol_Away_FT" in df.columns:
    df["risultato_ft"] = df["Gol_Home_FT"].astype(str) + "-" + df["Gol_Away_FT"].astype(str)
if "Gol_Home_HT" in df.columns and "Gol_Away_HT" in df.columns:
    df["risultato_ht"] = df["Gol_Home_HT"].astype(str) + "-" + df["Gol_Away_HT"].astype(str)

filters = {}

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
            
            st.sidebar.write(f"Range attuale {label or col_name}: {col_min} - {col_max}")
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
        odd_min = round(100 / perc, 2) if perc > 0 else np.nan
        stats.append((esito, count, perc, odd_min))
    
    df_stats = pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    df_stats["Odd Minima"] = df_stats["Odd Minima"].fillna('-').astype(str)
    return df_stats

def calcola_first_to_score_next_gol_outcome(df_to_analyze):
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
        odd_min = round(100 / perc, 2) if perc > 0 else np.nan
        stats.append((esito, count, perc, odd_min))
    
    df_stats = pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    df_stats["Odd Minima"] = df_stats["Odd Minima"].fillna('-').astype(str)
    return df_stats

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
    df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else np.nan)
    df_stats["Odd Minima"] = df_stats["Odd Minima"].fillna('-').astype(str)
    return df_stats

def calcola_stats_sh(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    df_sh = df_to_analyze.copy()
    
    df_sh["gol_home_sh"] = df_sh["Gol_Home_FT"] - df_sh["Gol_Home_HT"]
    df_sh["gol_away_sh"] = df_sh["Gol_Away_FT"] - df_sh["Gol_Away_HT"]
    
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
        odd_min = round(100 / perc, 2) if perc > 0 else np.nan
        stats_sh_winrate.append((esito, count, perc, odd_min))
    df_winrate_sh = pd.DataFrame(stats_sh_winrate, columns=["Esito", "Conteggio", "WinRate %", "Odd Minima"])
    df_winrate_sh["Odd Minima"] = df_winrate_sh["Odd Minima"].fillna('-').astype(str)
    
    over_sh_data = []
    df_sh["tot_goals_sh"] = df_sh["gol_home_sh"] + df_sh["gol_away_sh"]
    for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
        count = (df_sh["tot_goals_sh"] > t).sum()
        perc = round((count / len(df_sh)) * 100, 2)
        odd_min = round(100 / perc, 2) if perc > 0 else np.nan
        over_sh_data.append([f"Over {t} SH", count, perc, odd_min])
    df_over_sh = pd.DataFrame(over_sh_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
    df_over_sh["Odd Minima"] = df_over_sh["Odd Minima"].fillna('-').astype(str)

    btts_sh_count = ((df_sh["gol_home_sh"] > 0) & (df_sh["gol_away_sh"] > 0)).sum()
    no_btts_sh_count = len(df_sh) - btts_sh_count
    btts_sh_data = [
        ["BTTS SI SH", btts_sh_count, round((btts_sh_count / total_sh_matches) * 100, 2) if total_sh_matches > 0 else 0],
        ["BTTS NO SH", no_btts_sh_count, round((no_btts_sh_count / total_sh_matches) * 100, 2) if total_sh_matches > 0 else 0]
    ]
    df_btts_sh = pd.DataFrame(btts_sh_data, columns=["Mercato", "Conteggio", "Percentuale %"])
    df_btts_sh["Odd Minima"] = df_btts_sh["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else np.nan)
    df_btts_sh["Odd Minima"] = df_btts_sh["Odd Minima"].fillna('-').astype(str)
    
    return df_winrate_sh, df_over_sh, df_btts_sh

def calcola_first_to_score_sh(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    
    risultati = {"Home Team": 0, "Away Team": 0, "No Goals SH": 0}
    total_matches_sh = len(df_to_analyze)

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
        perc = round((count / total_matches_sh) * 100, 2) if total_matches_sh > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else np.nan
        stats.append((esito, count, perc, odd_min))
    
    df_stats = pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    df_stats["Odd Minima"] = df_stats["Odd Minima"].fillna('-').astype(str)
    return df_stats

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
    
    total_matches_sh = len(df_to_analyze)

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
        perc = round((count / total_matches_sh) * 100, 2) if total_matches_sh > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else np.nan
        stats.append((esito, count, perc, odd_min))
    
    df_stats = pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    df_stats["Odd Minima"] = df_stats["Odd Minima"].fillna('-').astype(str)
    return df_stats

def calcola_first_to_score_next_gol_outcome_sh(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    
    risultati = {
        "Casa Segna Primo SH e Segna di Nuovo SH": 0,
        "Casa Segna Primo SH e Subisce Gol SH": 0,
        "Trasferta Segna Prima SH e Segna di Nuovo SH": 0,
        "Trasferta Segna Prima SH e Subisce Gol SH": 0,
        "Solo un gol SH o nessuno": 0
    }
    
    total_matches_sh = len(df_to_analyze)

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
        perc = round((count / total_matches_sh) * 100, 2) if total_matches_sh > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else np.nan
        stats.append((esito, count, perc, odd_min))
    
    df_stats = pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    df_stats["Odd Minima"] = df_stats["Odd Minima"].fillna('-').astype(str)
    return df_stats

def calcola_to_score_sh(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

    df_to_score = df_to_analyze.copy()

    df_to_score["gol_home_sh"] = df_to_analyze["Gol_Home_FT"] - df_to_analyze["Gol_Home_HT"]
    df_to_score["gol_away_sh"] = df_to_analyze["Gol_Away_FT"] - df_to_analyze["Gol_Away_HT"]

    home_to_score_count = (df_to_score["gol_home_sh"] > 0).sum()
    away_to_score_count = (df_to_score["gol_away_sh"] > 0).sum()
    
    total_matches = len(df_to_score)
    
    data = []
    
    perc_home = round((home_to_score_count / total_matches) * 100, 2) if total_matches > 0 else 0
    odd_min_home = round(100 / perc_home, 2) if perc_home > 0 else np.nan
    data.append(["Home Team to Score SH", home_to_score_count, perc_home, odd_min_home])

    perc_away = round((away_to_score_count / total_matches) * 100, 2) if total_matches > 0 else 0
    odd_min_away = round(100 / perc_away, 2) if perc_away > 0 else np.nan
    data.append(["Away Team to Score SH", away_to_score_count, perc_away, odd_min_away])
    
    df_stats = pd.DataFrame(data, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    df_stats["Odd Minima"] = df_stats["Odd Minima"].fillna('-').astype(str)
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
    
    data = []
    
    perc_home = round((home_clean_sheet_count / total_matches) * 100, 2) if total_matches > 0 else 0
    odd_min_home = round(100 / perc_home, 2) if perc_home > 0 else np.nan
    data.append(["Clean Sheet SH (Casa)", home_clean_sheet_count, perc_home, odd_min_home])

    perc_away = round((away_clean_sheet_count / total_matches) * 100, 2) if total_matches > 0 else 0
    odd_min_away = round(100 / perc_away, 2) if perc_away > 0 else np.nan
    data.append(["Clean Sheet SH (Trasferta)", away_clean_sheet_count, perc_away, odd_min_away])
    
    df_stats = pd.DataFrame(data, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    df_stats["Odd Minima"] = df_stats["Odd Minima"].fillna('-').astype(str)
    return df_stats

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
        odd_min = round(100 / perc, 2) if perc > 0 else np.nan
        data.append([f"Over {r}", count, perc, odd_min])
        
    df_results = pd.DataFrame(data, columns=[f"Mercato (Over {period})", "Conteggio", "Percentuale %", "Odd Minima"])
    df_results["Odd Minima"] = df_results["Odd Minima"].fillna('-').astype(str)
    return df_results

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
        odd_min = round(100 / perc, 2) if perc > 0 else np.nan
        stats.append((esito, count, perc, odd_min))
    df_stats = pd.DataFrame(stats, columns=["Esito", "Conteggio", "WinRate %", "Odd Minima"])
    df_stats["Odd Minima"] = df_stats["Odd Minima"].fillna('-').astype(str)
    return df_stats

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
        odd_min = round(100 / perc, 2) if perc > 0 else np.nan
        stats.append((esito, count, perc, odd_min))
    
    df_stats = pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    df_stats["Odd Minima"] = df_stats["Odd Minima"].fillna('-').astype(str)
    return df_stats

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
        odd_min = round(100 / perc, 2) if perc > 0 else np.nan
        stats.append((esito, count, perc, odd_min))
    
    df_stats = pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    df_stats["Odd Minima"] = df_stats["Odd Minima"].fillna('-').astype(str)
    return df_stats

def mostra_risultati_esatti(df, col_risultato, titolo):
    risultati_interessanti = [
        "0-0", "0-1", "0-2", "0-3",
        "1-0", "1-1", "1-2", "1-3",
        "2-0", "2-1", "2-2", "2-3",
        "3-0", "3-1", "3-2", "3-3"
    ]
    df_valid = df[df[col_risultato].notna() & (df[col_risultato].str.contains("-"))].copy()

    if df_valid.empty:
        return pd.DataFrame(columns=[titolo, "Conteggio", "Percentuale %", "Odd Minima"])

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
    distribuzione["Odd Minima"] = distribuzione["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else np.nan)
    distribuzione["Odd Minima"] = distribuzione["Odd Minima"].fillna('-').astype(str)

    return distribuzione

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
        partite_con_almeno_2_gol = 0
        gol_fatti_home = 0
        gol_subiti_home = 0
        gol_fatti_away = 0
        gol_subiti_away = 0
        
        ht_total_goals_in_timeframe = []
        ft_total_goals_in_timeframe = []

        for _, row in df_to_analyze.iterrows():
            gol_home_str = str(row.get("Minutaggio_Gol_Home", ""))
            gol_away_str = str(row.get("Minutaggio_gol_Away", ""))

            gol_home = [int(x) for x in gol_home_str.split(";") if x.isdigit()]
            gol_away = [int(x) for x in gol_away_str.split(";") if x.isdigit()]
            
            goals_in_interval_home = [g for g in gol_home if max(start_interval, min_start_display) <= g <= end_interval]
            goals_in_interval_away = [g for g in gol_away if max(start_interval, min_start_display) <= g <= end_interval]
            
            total_goals_in_interval = len(goals_in_interval_home) + len(goals_in_interval_away)

            if total_goals_in_interval > 0:
                partite_con_gol += 1
                ht_total_goals_in_timeframe.append(row["Gol_Home_HT"] + row["Gol_Away_HT"])
                ft_total_goals_in_timeframe.append(row["Gol_Home_FT"] + row["Gol_Away_FT"])

            if total_goals_in_interval >= 2:
                partite_con_almeno_2_gol += 1

            gol_fatti_home += len(goals_in_interval_home)
            gol_subiti_home += len(goals_in_interval_away)
            gol_fatti_away += len(goals_in_interval_away)
            gol_subiti_away += len(goals_in_interval_home)
            
        perc_con_gol = round((partite_con_gol / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_min_con_gol = round(100 / perc_con_gol, 2) if perc_con_gol > 0 else np.nan
        
        perc_almeno_2_gol = round((partite_con_almeno_2_gol / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_min_almeno_2_gol = round(100 / perc_almeno_2_gol, 2) if perc_almeno_2_gol > 0 else np.nan

        avg_ht_goals_after_first_gol = round(np.mean(ht_total_goals_in_timeframe), 2) if ht_total_goals_in_timeframe else 0.00
        avg_ft_goals_after_first_gol = round(np.mean(ft_total_goals_in_timeframe), 2) if ft_total_goals_in_timeframe else 0.00

        risultati.append([
            label, 
            partite_con_gol, 
            perc_con_gol, 
            odd_min_con_gol, 
            perc_almeno_2_gol,
            odd_min_almeno_2_gol,
            gol_fatti_home, 
            gol_subiti_home, 
            gol_fatti_away, 
            gol_subiti_away,
            avg_ht_goals_after_first_gol,
            avg_ft_goals_after_first_gol
        ])
    
    if not risultati:
        st.info(f"Nessun intervallo di tempo rilevante dopo il minuto {min_start_display} per l'analisi a 15 minuti.")
        return

    df_result = pd.DataFrame(risultati, columns=[
        "Timeframe", 
        "Partite con Gol", 
        "Percentuale %", 
        "Odd Minima",
        ">= 2 Gol %", 
        "Odd Minima >= 2 Gol",
        "Gol Fatti Casa",
        "Gol Subiti Casa",
        "Gol Fatti Trasferta",
        "Gol Subiti Trasferta",
        "Media Gol HT (dopo 1° gol in timeframe)",
        "Media Gol FT (dopo 1° gol in timeframe)"
    ])
    df_result["Odd Minima"] = df_result["Odd Minima"].fillna('-').astype(str)
    df_result["Odd Minima >= 2 Gol"] = df_result["Odd Minima >= 2 Gol"].fillna('-').astype(str)

    styled_df = df_result.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %', '>= 2 Gol %']) 
    st.dataframe(styled_df)

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
        partite_con_almeno_2_gol = 0
        gol_fatti_home = 0
        gol_subiti_home = 0
        gol_fatti_away = 0
        gol_subiti_away = 0

        ht_total_goals_in_timeframe = []
        ft_total_goals_in_timeframe = []

        for _, row in df_to_analyze.iterrows():
            gol_home = [int(x) for x in str(row.get("Minutaggio_Gol_Home", "")).split(";") if x.isdigit()]
            gol_away = [int(x) for x in str(row.get("Minutaggio_gol_Away", "")).split(";") if x.isdigit()]
            
            goals_in_interval_home = [g for g in gol_home if max(start_interval, min_start_display) <= g <= end_interval]
            goals_in_interval_away = [g for g in gol_away if max(start_interval, min_start_display) <= g <= end_interval]
            
            total_goals_in_interval = len(goals_in_interval_home) + len(goals_in_interval_away)

            if total_goals_in_interval > 0:
                partite_con_gol += 1
                ht_total_goals_in_timeframe.append(row["Gol_Home_HT"] + row["Gol_Away_HT"])
                ft_total_goals_in_timeframe.append(row["Gol_Home_FT"] + row["Gol_Away_FT"])

            if total_goals_in_interval >= 2:
                partite_con_almeno_2_gol += 1

            gol_fatti_home += len(goals_in_interval_home)
            gol_subiti_home += len(goals_in_interval_away)
            gol_fatti_away += len(goals_in_interval_away)
            gol_subiti_away += len(goals_in_interval_home)
            
        perc_con_gol = round((partite_con_gol / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_min_con_gol = round(100 / perc_con_gol, 2) if perc_con_gol > 0 else np.nan

        perc_almeno_2_gol = round((partite_con_almeno_2_gol / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_min_almeno_2_gol = round(100 / perc_almeno_2_gol, 2) if perc_almeno_2_gol > 0 else np.nan

        avg_ht_goals_after_first_gol = round(np.mean(ht_total_goals_in_timeframe), 2) if ht_total_goals_in_timeframe else 0.00
        avg_ft_goals_after_first_gol = round(np.mean(ft_total_goals_in_timeframe), 2) if ft_total_goals_in_timeframe else 0.00

        risultati.append([
            label, 
            partite_con_gol, 
            perc_con_gol, 
            odd_min_con_gol,
            perc_almeno_2_gol,
            odd_min_almeno_2_gol,
            gol_fatti_home,
            gol_subiti_home,
            gol_fatti_away,
            gol_subiti_away,
            avg_ht_goals_after_first_gol,
            avg_ft_goals_after_first_gol
        ])
    
    if not risultati:
        st.info(f"Nessun intervallo di tempo rilevante dopo il minuto {min_start_display} per l'analisi a 5 minuti.")
        return

    df_result = pd.DataFrame(risultati, columns=[
        "Timeframe", 
        "Partite con Gol", 
        "Percentuale %", 
        "Odd Minima",
        ">= 2 Gol %", 
        "Odd Minima >= 2 Gol",
        "Gol Fatti Casa",
        "Gol Subiti Casa",
        "Gol Fatti Trasferta",
        "Gol Subiti Trasferta",
        "Media Gol HT (dopo 1° gol in timeframe)",
        "Media Gol FT (dopo 1° gol in timeframe)"
    ])
    df_result["Odd Minima"] = df_result["Odd Minima"].fillna('-').astype(str)
    df_result["Odd Minima >= 2 Gol"] = df_result["Odd Minima >= 2 Gol"].fillna('-').astype(str)

    styled_df = df_result.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %', '>= 2 Gol %']) 
    st.dataframe(styled_df)

def mostra_distribuzione_timeband_custom(df_to_analyze, min_start_display=0):
    if df_to_analyze.empty:
        st.warning("Il DataFrame per l'analisi a timing personalizzato è vuoto.")
        return

    custom_intervalli = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 45), (46, 55), (56, 65), (66, 75), (76, 85), (86, 90)]
    custom_label_intervalli = ["1-10", "11-20", "21-30", "31-40", "41-45", "46-55", "56-65", "66-75", "76-85", "86-90"]

    risultati = []
    total_matches = len(df_to_analyze)
    
    for (start_interval, end_interval), label in zip(custom_intervalli, custom_label_intervalli):
        if end_interval < min_start_display:
            continue

        partite_con_gol = 0
        partite_con_almeno_2_gol = 0
        gol_fatti_home = 0
        gol_subiti_home = 0
        gol_fatti_away = 0
        gol_subiti_away = 0

        ht_total_goals_in_timeframe = []
        ft_total_goals_in_timeframe = []

        for _, row in df_to_analyze.iterrows():
            gol_home = [int(x) for x in str(row.get("Minutaggio_Gol_Home", "")).split(";") if x.isdigit()]
            gol_away = [int(x) for x in str(row.get("Minutaggio_gol_Away", "")).split(";") if x.isdigit()]
            
            goals_in_interval_home = [g for g in gol_home if max(start_interval, min_start_display) <= g <= end_interval]
            goals_in_interval_away = [g for g in gol_away if max(start_interval, min_start_display) <= g <= end_interval]
            
            total_goals_in_interval = len(goals_in_interval_home) + len(goals_in_interval_away)

            if total_goals_in_interval > 0:
                partite_con_gol += 1
                ht_total_goals_in_timeframe.append(row["Gol_Home_HT"] + row["Gol_Away_HT"])
                ft_total_goals_in_timeframe.append(row["Gol_Home_FT"] + row["Gol_Away_FT"])

            if total_goals_in_interval >= 2:
                partite_con_almeno_2_gol += 1

            gol_fatti_home += len(goals_in_interval_home)
            gol_subiti_home += len(goals_in_interval_away)
            gol_fatti_away += len(goals_in_interval_away)
            gol_subiti_away += len(goals_in_interval_home)
            
        perc_con_gol = round((partite_con_gol / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_min_con_gol = round(100 / perc_con_gol, 2) if perc_con_gol > 0 else np.nan

        perc_almeno_2_gol = round((partite_con_almeno_2_gol / total_matches) * 100, 2) if total_matches > 0 else 0
        odd_min_almeno_2_gol = round(100 / perc_almeno_2_gol, 2) if perc_almeno_2_gol > 0 else np.nan

        avg_ht_goals_after_first_gol = round(np.mean(ht_total_goals_in_timeframe), 2) if ht_total_goals_in_timeframe else 0.00
        avg_ft_goals_after_first_gol = round(np.mean(ft_total_goals_in_timeframe), 2) if ft_total_goals_in_timeframe else 0.00

        risultati.append([
            label, 
            partite_con_gol, 
            perc_con_gol, 
            odd_min_con_gol,
            perc_almeno_2_gol,
            odd_min_almeno_2_gol,
            gol_fatti_home,
            gol_subiti_home,
            gol_fatti_away,
            gol_subiti_away,
            avg_ht_goals_after_first_gol,
            avg_ft_goals_after_first_gol
        ])
    
    if not risultati:
        st.info(f"Nessun intervallo di tempo rilevante dopo il minuto {min_start_display} per l'analisi a timing personalizzato.")
        return

    df_result = pd.DataFrame(risultati, columns=[
        "Timeframe", 
        "Partite con Gol", 
        "Percentuale %", 
        "Odd Minima",
        ">= 2 Gol %", 
        "Odd Minima >= 2 Gol",
        "Gol Fatti Casa",
        "Gol Subiti Casa",
        "Gol Fatti Trasferta",
        "Gol Subiti Trasferta",
        "Media Gol HT (dopo 1° gol in timeframe)",
        "Media Gol FT (dopo 1° gol in timeframe)"
    ])
    df_result["Odd Minima"] = df_result["Odd Minima"].fillna('-').astype(str)
    df_result["Odd Minima >= 2 Gol"] = df_result["Odd Minima >= 2 Gol"].fillna('-').astype(str)

    styled_df = df_result.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %', '>= 2 Gol %']) 
    st.dataframe(styled_df)

def calcola_next_gol(df_to_analyze, start_min, end_min):
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
        odd_min = round(100 / perc, 2) if perc > 0 else np.nan
        stats.append((esito, count, perc, odd_min))
    
    df_stats = pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    df_stats["Odd Minima"] = df_stats["Odd Minima"].fillna('-').astype(str)
    return df_stats

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
    df_rimonte_stats["Odd Minima"] = df_rimonte_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else np.nan)
    df_rimonte_stats["Odd Minima"] = df_rimonte_stats["Odd Minima"].fillna('-').astype(str)
    
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

def calcola_to_score(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

    df_to_score = df_to_analyze.copy()

    home_to_score_count = (df_to_score["Gol_Home_FT"] > 0).sum()
    away_to_score_count = (df_to_score["Gol_Away_FT"] > 0).sum()
    
    total_matches = len(df_to_score)
    
    data = []
    
    perc_home = round((home_to_score_count / total_matches) * 100, 2) if total_matches > 0 else 0
    odd_min_home = round(100 / perc_home, 2) if perc_home > 0 else np.nan
    data.append(["Home Team to Score", home_to_score_count, perc_home, odd_min_home])

    perc_away = round((away_to_score_count / total_matches) * 100, 2) if total_matches > 0 else 0
    odd_min_away = round(100 / perc_away, 2) if perc_away > 0 else np.nan
    data.append(["Away Team to Score", away_to_score_count, perc_away, odd_min_away])
    
    df_stats = pd.DataFrame(data, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    df_stats["Odd Minima"] = df_stats["Odd Minima"].fillna('-').astype(str)
    return df_stats

def calcola_to_score_ht(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

    df_to_score = df_to_analyze.copy()

    home_to_score_count = (df_to_score["Gol_Home_HT"] > 0).sum()
    away_to_score_count = (df_to_score["Gol_Away_HT"] > 0).sum()
    
    total_matches = len(df_to_analyze)
    
    data = []
    
    perc_home = round((home_to_score_count / total_matches) * 100, 2) if total_matches > 0 else 0
    odd_min_home = round(100 / perc_home, 2) if perc_home > 0 else np.nan
    data.append(["Home Team to Score", home_to_score_count, perc_home, odd_min_home])

    perc_away = round((away_to_score_count / total_matches) * 100, 2) if total_matches > 0 else 0
    odd_min_away = round(100 / perc_away, 2) if perc_away > 0 else np.nan
    data.append(["Away Team to Score", away_to_score_count, perc_away, odd_min_away])
    
    df_stats = pd.DataFrame(data, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    df_stats["Odd Minima"] = df_stats["Odd Minima"].fillna('-').astype(str)
    return df_stats

def calcola_btts_ht(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])

    df_btts_ht = df_to_analyze.copy()
    
    btts_count = ((df_btts_ht["Gol_Home_HT"] > 0) & (df_btts_ht["Gol_Away_HT"] > 0)).sum()
    no_btts_count = len(df_btts_ht) - btts_count
    
    total_matches = len(df_btts_ht)
    
    data = []
    
    perc_si = round((btts_count / total_matches) * 100, 2) if total_matches > 0 else 0
    odd_min_si = round(100 / perc_si, 2) if perc_si > 0 else np.nan
    data.append(["BTTS SI HT (Dinamica)", btts_count, perc_si, odd_min_si])

    perc_no = round((no_btts_count / total_matches) * 100, 2) if total_matches > 0 else 0
    odd_min_no = round(100 / perc_no, 2) if perc_no > 0 else np.nan
    data.append(["BTTS NO HT (Dinamica)", no_btts_count, perc_no, odd_min_no])
    
    df_stats = pd.DataFrame(data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
    df_stats["Odd Minima"] = df_stats["Odd Minima"].fillna('-').astype(str)
    return df_stats

def calcola_btts_ft(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])

    df_btts_ft = df_to_analyze.copy()
    
    btts_count = ((df_btts_ft["Gol_Home_FT"] > 0) & (df_btts_ft["Gol_Away_FT"] > 0)).sum()
    no_btts_count = len(df_btts_ft) - btts_count
    
    total_matches = len(df_btts_ft)
    
    data = []
    
    perc_si = round((btts_count / total_matches) * 100, 2) if total_matches > 0 else 0
    odd_min_si = round(100 / perc_si, 2) if perc_si > 0 else np.nan
    data.append(["BTTS SI FT", btts_count, perc_si, odd_min_si])

    perc_no = round((no_btts_count / total_matches) * 100, 2) if total_matches > 0 else 0
    odd_min_no = round(100 / perc_no, 2) if perc_no > 0 else np.nan
    data.append(["BTTS NO FT", no_btts_count, perc_no, odd_min_no])
    
    df_stats = pd.DataFrame(data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
    df_stats["Odd Minima"] = df_stats["Odd Minima"].fillna('-').astype(str)
    return df_stats

def calcola_btts_dinamico(df_to_analyze, current_minute_filter_val):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])

    total_matches = len(df_to_analyze)
    btts_si_count = 0

    for _, row in df_to_analyze.iterrows():
        # Score at current minute
        score_home_at_current_minute = sum(1 for g in [int(x) for x in str(row.get("Minutaggio_Gol_Home", "")).split(";") if x.isdigit()] if g <= current_minute_filter_val)
        score_away_at_current_minute = sum(1 for g in [int(x) for x in str(row.get("Minutaggio_gol_Away", "")).split(";") if x.isdigit()] if g <= current_minute_filter_val)

        # Goals scored after current minute
        goals_home_after = sum(1 for g in [int(x) for x in str(row.get("Minutaggio_Gol_Home", "")).split(";") if x.isdigit()] if g > current_minute_filter_val)
        goals_away_after = sum(1 for g in [int(x) for x in str(row.get("Minutaggio_gol_Away", "")).split(";") if x.isdigit()] if g > current_minute_filter_val)

        # Determine if BTTS happened considering the state at current_minute_filter_val
        home_scored_already = score_home_at_current_minute > 0
        away_scored_already = score_away_at_current_minute > 0

        btts_achieved = False
        if home_scored_already and away_scored_already:
            btts_achieved = True # Already BTTS at current minute
        elif home_scored_already and not away_scored_already:
            if goals_away_after > 0: # Away scores after current minute
                btts_achieved = True
        elif not home_scored_already and away_scored_already:
            if goals_home_after > 0: # Home scores after current minute
                btts_achieved = True
        elif not home_scored_already and not away_scored_already:
            if goals_home_after > 0 and goals_away_after > 0: # Both score after current minute
                btts_achieved = True
        
        if btts_achieved:
            btts_si_count += 1

    no_btts_count = total_matches - btts_si_count

    data = []
    
    perc_si = round((btts_si_count / total_matches) * 100, 2) if total_matches > 0 else 0
    odd_min_si = round(100 / perc_si, 2) if perc_si > 0 else np.nan
    data.append(["BTTS SI (risultato finale)", btts_si_count, perc_si, odd_min_si])

    perc_no = round((no_btts_count / total_matches) * 100, 2) if total_matches > 0 else 0
    odd_min_no = round(100 / perc_no, 2) if perc_no > 0 else np.nan
    data.append(["BTTS NO (risultato finale)", no_btts_count, perc_no, odd_min_no])
    
    df_stats = pd.DataFrame(data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
    df_stats["Odd Minima"] = df_stats["Odd Minima"].fillna('-').astype(str)
    return df_stats
    
def calcola_btts_ht_dinamico(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])

    df_btts_ht_dinamico = df_to_analyze.copy()
    
    btts_count = ((df_btts_ht_dinamico["Gol_Home_HT"] > 0) & (df_btts_ht_dinamico["Gol_Away_HT"] > 0)).sum()
    no_btts_count = len(df_btts_ht_dinamico) - btts_count
    
    total_matches = len(df_btts_ht_dinamico)
    
    data = []
    
    perc_si = round((btts_count / total_matches) * 100, 2) if total_matches > 0 else 0
    odd_min_si = round(100 / perc_si, 2) if perc_si > 0 else np.nan
    data.append(["BTTS SI HT (Dinamica)", btts_count, perc_si, odd_min_si])

    perc_no = round((no_btts_count / total_matches) * 100, 2) if total_matches > 0 else 0
    odd_min_no = round(100 / perc_no, 2) if perc_no > 0 else np.nan
    data.append(["BTTS NO HT (Dinamica)", no_btts_count, perc_no, odd_min_no])
    
    df_stats = pd.DataFrame(data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
    df_stats["Odd Minima"] = df_stats["Odd Minima"].fillna('-').astype(str)
    return df_stats

def calcola_clean_sheet(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    
    df_clean_sheet = df_to_analyze.copy()
    
    home_clean_sheet_count = (df_clean_sheet["Gol_Away_FT"] == 0).sum()
    away_clean_sheet_count = (df_clean_sheet["Gol_Home_FT"] == 0).sum()
    
    total_matches = len(df_clean_sheet)
    
    data = []
    
    perc_home = round((home_clean_sheet_count / total_matches) * 100, 2) if total_matches > 0 else 0
    odd_min_home = round(100 / perc_home, 2) if perc_home > 0 else np.nan
    data.append(["Clean Sheet (Casa)", home_clean_sheet_count, perc_home, odd_min_home])

    perc_away = round((away_clean_sheet_count / total_matches) * 100, 2) if total_matches > 0 else 0
    odd_min_away = round(100 / perc_away, 2) if perc_away > 0 else np.nan
    data.append(["Clean Sheet (Trasferta)", away_clean_sheet_count, perc_away, odd_min_away])
    
    df_stats = pd.DataFrame(data, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    df_stats["Odd Minima"] = df_stats["Odd Minima"].fillna('-').astype(str)
    return df_stats

def calcola_combo_stats(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
        
    df_combo = df_to_analyze.copy()

    df_combo["tot_goals_ft"] = df_combo["Gol_Home_FT"] + df_combo["Gol_Away_FT"]
    
    btts_over_2_5_count = ((df_combo["Gol_Home_FT"] > 0) & (df_combo["Gol_Away_FT"] > 0) & (df_combo["tot_goals_ft"] > 2.5)).sum()
    
    home_win_over_2_5_count = ((df_combo["Gol_Home_FT"] > df_combo["Gol_Away_FT"]) & (df_combo["tot_goals_ft"] > 2.5)).sum()
    
    away_win_over_2_5_count = ((df_combo["Gol_Away_FT"] > df_combo["Gol_Home_FT"]) & (df_combo["tot_goals_ft"] > 2.5)).sum()
    
    total_matches = len(df_combo)
    
    data = []
    
    perc_btts_over = round((btts_over_2_5_count / total_matches) * 100, 2) if total_matches > 0 else 0
    odd_min_btts_over = round(100 / perc_btts_over, 2) if perc_btts_over > 0 else np.nan
    data.append(["BTTS SI + Over 2.5", btts_over_2_5_count, perc_btts_over, odd_min_btts_over])

    perc_home_win_over = round((home_win_over_2_5_count / total_matches) * 100, 2) if total_matches > 0 else 0
    odd_min_home_win_over = round(100 / perc_home_win_over, 2) if perc_home_win_over > 0 else np.nan
    data.append(["Casa vince + Over 2.5", home_win_over_2_5_count, perc_home_win_over, odd_min_home_win_over])

    perc_away_win_over = round((away_win_over_2_5_count / total_matches) * 100, 2) if total_matches > 0 else 0
    odd_min_away_win_over = round(100 / perc_away_win_over, 2) if perc_away_win_over > 0 else np.nan
    data.append(["Ospite vince + Over 2.5", away_win_over_2_5_count, perc_away_win_over, odd_min_away_win_over])
    
    df_stats = pd.DataFrame(data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
    df_stats["Odd Minima"] = df_stats["Odd Minima"].fillna('-').astype(str)
    return df_stats

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
        odd_min = round(100 / perc, 2) if perc > 0 else np.nan
        data.append([f"Multi Gol {label}", count, perc, odd_min])
        
    df_stats = pd.DataFrame(data, columns=[f"Mercato ({titolo})", "Conteggio", "Percentuale %", "Odd Minima"])
    df_stats["Odd Minima"] = df_stats["Odd Minima"].fillna('-').astype(str)
    return df_stats

def get_score_at_minute(row, target_minute):
    gol_home_str = str(row.get("Minutaggio_Gol_Home", ""))
    gol_away_str = str(row.get("Minutaggio_gol_Away", ""))

    gol_home_minutes = [int(x) for x in gol_home_str.split(";") if x.isdigit()]
    gol_away_minutes = [int(x) for x in gol_away_str.split(";") if x.isdigit()]

    score_home = sum(1 for g_min in gol_home_minutes if g_min <= target_minute)
    score_away = sum(1 for g_min in gol_away_minutes if g_min <= target_minute)
    return f"{score_home}-{score_away}"

def calcola_analisi_dinamica_avanzata(df_base, first_goal_result_at_minute_str, first_goal_timeband_label, 
                                     current_minute_filter_val,
                                     second_goal_result_at_minute_str=None, second_goal_timeband_label=None,
                                     min_odd_home=None, max_odd_home=None, min_odd_away=None, max_odd_away=None):
    
    df_filtered_step1 = pd.DataFrame()
    primo_gol_timebands = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 45), (46, 55), (56, 65), (66, 75), (76, 85), (86, 90)]
    primo_gol_label_timebands = ["1-10", "11-20", "21-30", "31-40", "41-45", "46-55", "56-65", "66-75", "76-85", "86-90"]

    df_current_filtered = df_base.copy()
    if min_odd_home is not None and max_odd_home is not None:
        df_current_filtered = df_current_filtered[df_current_filtered["Odd_Home"].between(min_odd_home, max_odd_home)]
    if min_odd_away is not None and max_odd_away is not None:
        df_current_filtered = df_current_filtered[df_current_filtered["Odd__Away"].between(min_odd_away, max_odd_away)]

    first_goal_interval = None
    for i, label in enumerate(primo_gol_label_timebands):
        if label == first_goal_timeband_label:
            first_goal_interval = primo_gol_timebands[i]
            break

    if first_goal_interval is None:
        st.error(f"Timeband primo gol '{first_goal_timeband_label}' non valido.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for _, row in df_current_filtered.iterrows():
        gol_home_str = str(row.get("Minutaggio_Gol_Home", ""))
        gol_away_str = str(row.get("Minutaggio_gol_Away", ""))

        all_goals_minutes = []
        if gol_home_str:
            all_goals_minutes.extend([int(x) for x in gol_home_str.split(";") if x.isdigit()])
        if gol_away_str:
            all_goals_minutes.extend([int(x) for x in gol_away_str.split(";") if x.isdigit()])
        
        if all_goals_minutes:
            first_goal_minute = min(all_goals_minutes)
            
            if first_goal_interval[0] <= first_goal_minute <= first_goal_interval[1]:
                score_at_first_goal = get_score_at_minute(row, first_goal_minute)
                
                if first_goal_result_at_minute_str == "Tutti" or score_at_first_goal == first_goal_result_at_minute_str:
                    if first_goal_minute <= current_minute_filter_val:
                        df_filtered_step1 = pd.concat([df_filtered_step1, pd.DataFrame([row])], ignore_index=True)

    if df_filtered_step1.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    final_filtered_df = df_filtered_step1.copy()
    if second_goal_result_at_minute_str and second_goal_result_at_minute_str != "Tutti":
        df_filtered_step2 = pd.DataFrame()
        second_goal_interval = None
        if second_goal_timeband_label and second_goal_timeband_label != "Qualsiasi":
            second_goal_timebands = primo_gol_timebands 
            for i, label in enumerate(primo_gol_label_timebands):
                if label == second_goal_timeband_label:
                    second_goal_interval = second_goal_timebands[i]
                    break
        
        for _, row in final_filtered_df.iterrows():
            gol_home_str = str(row.get("Minutaggio_Gol_Home", ""))
            gol_away_str = str(row.get("Minutaggio_gol_Away", ""))

            all_goals_minutes = []
            if gol_home_str:
                all_goals_minutes.extend([ (int(x), 'home') for x in gol_home_str.split(";") if x.isdigit() ])
            if gol_away_str:
                all_goals_minutes.extend([ (int(x), 'away') for x in gol_away_str.split(";") if x.isdigit() ])
            
            all_goals_minutes.sort()

            if len(all_goals_minutes) >= 2:
                first_goal_min = all_goals_minutes[0][0]
                second_goal_min = all_goals_minutes[1][0]

                if second_goal_min > first_goal_min and second_goal_min <= current_minute_filter_val:
                    score_at_second_goal = get_score_at_minute(row, second_goal_min)
                    
                    if second_goal_result_at_minute_str == "Tutti" or score_at_second_goal == second_goal_result_at_minute_str:
                        if second_goal_interval is None or (second_goal_interval[0] <= second_goal_min <= second_goal_interval[1]):
                            df_filtered_step2 = pd.concat([df_filtered_step2, pd.DataFrame([row])], ignore_index=True)
        final_filtered_df = df_filtered_step2
    
    if final_filtered_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    over_ht_data = []
    df_temp_ht = final_filtered_df.copy()
    df_temp_ht["tot_goals_ht"] = df_temp_ht["Gol_Home_HT"] + df_temp_ht["Gol_Away_HT"]
    for t in [0.5, 1.5, 2.5]:
        count = (df_temp_ht["tot_goals_ht"] > t).sum()
        perc = round((count / len(df_temp_ht)) * 100, 2) if len(df_temp_ht) > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else np.nan
        over_ht_data.append([f"Over {t} HT", count, perc, odd_min])
    df_over_ht = pd.DataFrame(over_ht_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
    df_over_ht["Odd Minima"] = df_over_ht["Odd Minima"].fillna('-').astype(str)

    over_ft_data = []
    df_temp_ft = final_filtered_df.copy()
    df_temp_ft["tot_goals_ft"] = df_temp_ft["Gol_Home_FT"] + df_temp_ft["Gol_Away_FT"]
    for t in [0.5, 1.5, 2.5, 3.5, 4.5]:
        count = (df_temp_ft["tot_goals_ft"] > t).sum()
        perc = round((count / len(df_temp_ft)) * 100, 2) if len(df_temp_ft) > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else np.nan
        over_ft_data.append([f"Over {t} FT", count, perc, odd_min])
    df_over_ft = pd.DataFrame(over_ft_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
    df_over_ft["Odd Minima"] = df_over_ft["Odd Minima"].fillna('-').astype(str)

    df_winrate_ht = calcola_winrate(final_filtered_df, "risultato_ht")
    df_winrate_ft = calcola_winrate(final_filtered_df, "risultato_ft")

    def get_exact_scores_df(df_input, col_risultato, titolo):
        risultati_interessanti = [
            "0-0", "0-1", "0-2", "0-3",
            "1-0", "1-1", "1-2", "1-3",
            "2-0", "2-1", "2-2", "2-3",
            "3-0", "3-1", "3-2", "3-3"
        ]
        df_valid = df_input[df_input[col_risultato].notna() & (df_input[col_risultato].str.contains("-"))].copy()

        if df_valid.empty:
            return pd.DataFrame(columns=[titolo, "Conteggio", "Percentuale %", "Odd Minima"])

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
        distribuzione["Odd Minima"] = distribuzione["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else np.nan)
        distribuzione["Odd Minima"] = distribuzione["Odd Minima"].fillna('-').astype(str)
        return distribuzione

    df_exact_scores_ht = get_exact_scores_df(final_filtered_df, "risultato_ht", "Risultato Esatto HT")
    df_exact_scores_ft = get_exact_scores_df(final_filtered_df, "risultato_ft", "Risultato Esatto FT")

    df_btts_ft_after_current_minute = calcola_btts_dinamico(final_filtered_df, current_minute_filter_val)

    return df_over_ht, df_over_ft, df_winrate_ht, df_winrate_ft, df_exact_scores_ht, df_exact_scores_ft, df_btts_ft_after_current_minute


st.subheader("1. Analisi Timeband per Campionato")
if selected_league != "Tutte":
    df_league_only = df[df["League"] == selected_league]
    st.write(f"Analisi basata su **{len(df_league_only)}** partite del campionato **{selected_league}**.")
    st.write("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Distribuzione Gol per Timeframe (15min)**")
        mostra_distribuzione_timeband(df_league_only)
    with col2:
        st.write("**Distribuzione Gol per Timeframe (5min)**")
        mostra_distribuzione_timeband_5min(df_league_only)
    with col3:
        st.write("**Distribuzione Gol per Timeframe (Personalizzata)**")
        mostra_distribuzione_timeband_custom(df_league_only)
else:
    st.write("Seleziona un campionato per visualizzare questa analisi.")

st.subheader("2. Analisi Timeband per Campionato e Quote")
st.write(f"Analisi basata su **{len(filtered_df)}** partite filtrate da tutti i parametri della sidebar.")
if not filtered_df.empty:
    st.write("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Distribuzione Gol per Timeframe (15min)**")
        mostra_distribuzione_timeband(filtered_df)
    with col2:
        st.write("**Distribuzione Gol per Timeframe (5min)**")
        mostra_distribuzione_timeband_5min(filtered_df)
    with col3:
        st.write("**Distribuzione Gol per Timeframe (Personalizzata)**")
        mostra_distribuzione_timeband_custom(filtered_df)
else:
    st.warning("Nessuna partita corrisponde ai filtri selezionati.")

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
        st.subheader(f"Risultati Esatti HT ({len(filtered_df)})")
        df_exact_ht_prematch = mostra_risultati_esatti(filtered_df, "risultato_ht", "Risultato Esatto HT")
        if not df_exact_ht_prematch.empty:
            st.dataframe(df_exact_ht_prematch.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
        else:
            st.info("Nessun risultato esatto HT disponibile per i filtri selezionati.")

        st.subheader(f"WinRate HT ({len(filtered_df)})")
        df_winrate_ht = calcola_winrate(filtered_df, "risultato_ht")
        if not df_winrate_ht.empty:
            styled_df_ht = df_winrate_ht.style.background_gradient(cmap='RdYlGn', subset=['WinRate %'])
            st.dataframe(styled_df_ht)
        else:
            st.info("Nessun WinRate HT disponibile per i filtri selezionati.")

        st.subheader(f"Over Goals HT ({len(filtered_df)})")
        over_ht_data = []
        df_prematch_ht = filtered_df.copy()
        df_prematch_ht["tot_goals_ht"] = df_prematch_ht["Gol_Home_HT"] + df_prematch_ht["Gol_Away_HT"]
        for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
            count = (df_prematch_ht["tot_goals_ht"] > t).sum()
            perc = round((count / len(df_prematch_ht)) * 100, 2)
            odd_min = round(100 / perc, 2) if perc > 0 else np.nan
            over_ht_data.append([f"Over {t} HT", count, perc, odd_min])
        df_over_ht = pd.DataFrame(over_ht_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
        df_over_ht["Odd Minima"] = df_over_ht["Odd Minima"].fillna('-').astype(str)
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
            df_goals_fatti_home_ht = calcola_goals_per_team_period(filtered_df, 'home', 'fatti', 'ht')
            if not df_goals_fatti_home_ht.empty:
                st.dataframe(df_goals_fatti_home_ht.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
            else:
                st.info("Nessun dato disponibile.")
        with col2:
            st.markdown("#### Subiti Casa")
            df_goals_subiti_home_ht = calcola_goals_per_team_period(filtered_df, 'home', 'subiti', 'ht')
            if not df_goals_subiti_home_ht.empty:
                st.dataframe(df_goals_subiti_home_ht.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
            else:
                st.info("Nessun dato disponibile.")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### Fatti Trasferta")
            df_goals_fatti_away_ht = calcola_goals_per_team_period(filtered_df, 'away', 'fatti', 'ht')
            if not df_goals_fatti_away_ht.empty:
                st.dataframe(df_goals_fatti_away_ht.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
            else:
                st.info("Nessun dato disponibile.")
        with col4:
            st.markdown("#### Subiti Trasferta")
            df_goals_subiti_away_ht = calcola_goals_per_team_period(filtered_df, 'away', 'subiti', 'ht')
            if not df_goals_subiti_away_ht.empty:
                st.dataframe(df_goals_subiti_away_ht.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
            else:
                st.info("Nessun dato disponibile.")
    
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
            odd_min = round(100 / perc, 2) if perc > 0 else np.nan
            stats_sh_winrate.append((esito, count, perc, odd_min))
        df_winrate_sh = pd.DataFrame(stats_sh_winrate, columns=["Esito", "Conteggio", "WinRate %", "Odd Minima"])
        df_winrate_sh["Odd Minima"] = df_winrate_sh["Odd Minima"].fillna('-').astype(str)
        styled_df = df_winrate_sh.style.background_gradient(cmap='RdYlGn', subset=['WinRate %'])
        st.dataframe(styled_df)

        st.subheader(f"Over Goals SH ({len(filtered_df)})")
        over_sh_data = []
        df_sh["tot_goals_sh"] = df_sh["gol_home_sh"] + df_sh["gol_away_sh"]
        for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
            count = (df_sh["tot_goals_sh"] > t).sum()
            perc = round((count / len(df_sh)) * 100, 2)
            odd_min = round(100 / perc, 2) if perc > 0 else np.nan
            over_sh_data.append([f"Over {t} SH", count, perc, odd_min])
        df_over_sh = pd.DataFrame(over_sh_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
        df_over_sh["Odd Minima"] = df_over_sh["Odd Minima"].fillna('-').astype(str)
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
        df_btts_sh["Odd Minima"] = df_btts_sh["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else np.nan)
        df_btts_sh["Odd Minima"] = df_btts_sh["Odd Minima"].fillna('-').astype(str)
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
        styled_df = calcola_first_to_score_next_gol_outcome_sh(filtered_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
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
            df_goals_fatti_home_sh = calcola_goals_per_team_period(filtered_df, 'home', 'fatti', 'sh')
            if not df_goals_fatti_home_sh.empty:
                st.dataframe(df_goals_fatti_home_sh.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
            else:
                st.info("Nessun dato disponibile.")
        with col2:
            st.markdown("#### Subiti Casa")
            df_goals_subiti_home_sh = calcola_goals_per_team_period(filtered_df, 'home', 'subiti', 'sh')
            if not df_goals_subiti_home_sh.empty:
                st.dataframe(df_goals_subiti_home_sh.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
            else:
                st.info("Nessun dato disponibile.")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### Fatti Trasferta")
            df_goals_fatti_away_sh = calcola_goals_per_team_period(filtered_df, 'away', 'fatti', 'sh')
            if not df_goals_fatti_away_sh.empty:
                st.dataframe(df_goals_fatti_away_sh.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
            else:
                st.info("Nessun dato disponibile.")
        with col4:
            st.markdown("#### Subiti Trasferta")
            df_goals_subiti_away_sh = calcola_goals_per_team_period(filtered_df, 'away', 'subiti', 'sh')
            if not df_goals_subiti_away_sh.empty:
                st.dataframe(df_goals_subiti_away_sh.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
            else:
                st.info("Nessun dato disponibile.")

    with st.expander("Mostra Statistiche FT (Finale)"):
        st.subheader(f"Risultati Esatti FT ({len(filtered_df)})")
        df_exact_ft_prematch = mostra_risultati_esatti(filtered_df, "risultato_ft", "Risultato Esatto FT")
        if not df_exact_ft_prematch.empty:
            st.dataframe(df_exact_ft_prematch.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
        else:
            st.info("Nessun risultato esatto FT disponibile per i filtri selezionati.")

        st.subheader(f"WinRate FT ({len(filtered_df)})")
        df_winrate_ft = calcola_winrate(filtered_df, "risultato_ft")
        if not df_winrate_ft.empty:
            styled_df_ft = df_winrate_ft.style.background_gradient(cmap='RdYlGn', subset=['WinRate %'])
            st.dataframe(styled_df_ft)
        else:
            st.info("Nessun WinRate FT disponibile per i filtri selezionati.")

        st.subheader(f"Over Goals FT ({len(filtered_df)})")
        over_ft_data = []
        df_prematch_ft = filtered_df.copy()
        df_prematch_ft["tot_goals_ft"] = df_prematch_ft["Gol_Home_FT"] + df_prematch_ft["Gol_Away_FT"]
        for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
            count = (df_prematch_ft["tot_goals_ft"] > t).sum()
            perc = round((count / len(df_prematch_ft)) * 100, 2)
            odd_min = round(100 / perc, 2) if perc > 0 else np.nan
            over_ft_data.append([f"Over {t} FT", count, perc, odd_min])
        df_over_ft = pd.DataFrame(over_ft_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
        df_over_ft["Odd Minima"] = df_over_ft["Odd Minima"].fillna('-').astype(str)
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
        styled_df = calcola_first_to_score_next_gol_outcome(filtered_df).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
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
            df_goals_fatti_home_ft = calcola_goals_per_team_period(filtered_df, 'home', 'fatti', 'ft')
            if not df_goals_fatti_home_ft.empty:
                st.dataframe(df_goals_fatti_home_ft.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
            else:
                st.info("Nessun dato disponibile.")
        with col2:
            st.markdown("#### Subiti Casa")
            df_goals_subiti_home_ft = calcola_goals_per_team_period(filtered_df, 'home', 'subiti', 'ft')
            if not df_goals_subiti_home_ft.empty:
                st.dataframe(df_goals_subiti_home_ft.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
            else:
                st.info("Nessun dato disponibile.")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### Fatti Trasferta")
            df_goals_fatti_away_ft = calcola_goals_per_team_period(filtered_df, 'away', 'fatti', 'ft')
            if not df_goals_fatti_away_ft.empty:
                st.dataframe(df_goals_fatti_away_ft.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
            else:
                st.info("Nessun dato disponibile.")
        with col4:
            st.markdown("#### Subiti Trasferta")
            df_goals_subiti_away_ft = calcola_goals_per_team_period(filtered_df, 'away', 'subiti', 'ft')
            if not df_goals_subiti_away_ft.empty:
                st.dataframe(df_goals_subiti_away_ft.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
            else:
                st.info("Nessun dato disponibile.")

else:
    st.warning("Nessuna partita corrisponde ai filtri selezionati per l'analisi pre-match.")

st.subheader("4. Analisi Timeband Dinamica (Minuto/Risultato)")
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
            
            st.subheader(f"Risultati Esatti HT ({len(df_target)})")
            df_exact_ht_dynamic = mostra_risultati_esatti(df_target, "risultato_ht", "Risultato Esatto HT")
            if not df_exact_ht_dynamic.empty:
                st.dataframe(df_exact_ht_dynamic.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
            else:
                st.info("Nessun risultato esatto HT dinamico disponibile.")

            st.subheader(f"Risultati Esatti FT ({len(df_target)})")
            df_exact_ft_dynamic = mostra_risultati_esatti(df_target, "risultato_ft", "Risultato Esatto FT")
            if not df_exact_ft_dynamic.empty:
                st.dataframe(df_exact_ft_dynamic.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
            else:
                st.info("Nessun risultato esatto FT dinamico disponibile.")


            st.subheader(f"WinRate (Dinamica) ({len(df_target)})")
            st.write("**HT:**")
            df_winrate_ht_dynamic = calcola_winrate(df_target, "risultato_ht")
            if not df_winrate_ht_dynamic.empty:
                styled_df_ht = df_winrate_ht_dynamic.style.background_gradient(cmap='RdYlGn', subset=['WinRate %'])
                st.dataframe(styled_df_ht)
            else:
                st.info("Nessun WinRate HT dinamico disponibile.")
            st.write("**FT:**")
            df_winrate_ft_dynamic = calcola_winrate(df_target, "risultato_ft")
            if not df_winrate_ft_dynamic.empty:
                styled_df_ft = df_winrate_ft_dynamic.style.background_gradient(cmap='RdYlGn', subset=['WinRate %'])
                st.dataframe(styled_df_ft)
            else:
                st.info("Nessun WinRate FT dinamico disponibile.")
            
            col1, col2 = st.columns(2)
            df_target_goals["tot_goals_ht"] = df_target_goals["Gol_Home_HT"] + df_target_goals["Gol_Away_HT"]
            df_target_goals["tot_goals_ft"] = df_target_goals["Gol_Home_FT"] + df_target_goals["Gol_Away_FT"]
            
            with col1:
                st.subheader(f"Over Goals HT (Dinamica) ({len(df_target)})")
                over_ht_data_dynamic = []
                for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
                    count = (df_target_goals["tot_goals_ht"] > t).sum()
                    perc = round((count / len(df_target_goals)) * 100, 2)
                    odd_min = round(100 / perc, 2) if perc > 0 else np.nan
                    over_ht_data_dynamic.append([f"Over {t} HT", count, perc, odd_min])
                df_over_ht_dynamic = pd.DataFrame(over_ht_data_dynamic, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
                df_over_ht_dynamic["Odd Minima"] = df_over_ht_dynamic["Odd Minima"].fillna('-').astype(str)
                styled_over_ht_dynamic = df_over_ht_dynamic.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
                st.dataframe(styled_over_ht_dynamic)
            
            with col2:
                st.subheader(f"Over Goals FT (Dinamica) ({len(df_target)})")
                over_ft_data = []
                for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
                    count = (df_target_goals["tot_goals_ft"] > t).sum()
                    perc = round((count / len(df_target_goals)) * 100, 2)
                    odd_min = round(100 / perc, 2) if perc > 0 else np.nan
                    over_ft_data.append([f"Over {t} FT", count, perc, odd_min])
                df_over_ft = pd.DataFrame(over_ft_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
                df_over_ft["Odd Minima"] = df_over_ft["Odd Minima"].fillna('-').astype(str)
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
                df_btts_ft_dynamic = calcola_btts_dinamico(df_target, start_min) 
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
            styled_df = calcola_first_to_score_next_gol_outcome(df_target).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
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
            
            st.subheader(f"Next Gol (Dinamica) ({len(df_target)})")
            styled_df = calcola_next_gol(df_target, start_min, end_min).style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
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
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**15min**")
                mostra_distribuzione_timeband(df_target, min_start_display=start_min)
            with col2:
                st.write("**5min**")
                mostra_distribuzione_timeband_5min(df_target, min_start_display=start_min)
            with col3:
                st.write("**Personalizzata**")
                mostra_distribuzione_timeband_custom(df_target, min_start_display=start_min)

    else:
        st.warning("Il dataset filtrato è vuoto o mancano le colonne necessarie per l'analisi.")
