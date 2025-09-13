import streamlit as st
import pandas as pd
import re

# Set page configuration
st.set_page_config(layout="wide")

# Function to parse goal timings
def parse_goal_timings(timings_str):
    if pd.isna(timings_str) or timings_str.strip() == '':
        return []
    
    timings = []
    for t in timings_str.split(','):
        match = re.search(r'(\d+)\'(\d+)', t)
        if match:
            base_min = int(match.group(1))
            extra_min = int(match.group(2))
            timings.append(base_min + extra_min)
        else:
            try:
                timings.append(int(t))
            except ValueError:
                continue
    return sorted(list(set(timings)))

# Function to determine outcome
def get_outcome(home_goals, away_goals):
    if home_goals > away_goals:
        return 'Home Win'
    elif away_goals > home_goals:
        return 'Away Win'
    else:
        return 'Draw'

st.title("⚽ Analisi Statistica sui Gol")
st.markdown("Usa i filtri a sinistra per affinare la tua analisi.")
st.markdown("---")

# Sidebar file uploader
st.sidebar.header("Carica il tuo file")
uploaded_file = st.sidebar.file_uploader("Carica il tuo file CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, delimiter=';')
    except Exception as e:
        st.error(f"Errore durante il caricamento del file: {e}")
        st.stop()
    
    # Data Cleaning and Preparation
    df = df.rename(columns={'Game Week': 'Game_Week'})
    df = df[df['status'] == 'complete'].copy()

    # Dynamic column selection
    all_columns = df.columns.tolist()
    st.sidebar.markdown("---")
    st.sidebar.header("Configurazione Colonne")
    selected_league_col = st.sidebar.selectbox('Seleziona la colonna del campionato', all_columns)
    
    # New odds column selectors
    home_odds_col = st.sidebar.selectbox('Seleziona la colonna per le quote Home', all_columns)
    away_odds_col = st.sidebar.selectbox('Seleziona la colonna per le quote Away', all_columns)
    
    # Add a combined timings column
    df['all_goal_timings'] = df.apply(
        lambda row: parse_goal_timings(row['home_team_goal_timings']) + parse_goal_timings(row['away_team_goal_timings']), 
        axis=1
    )

    # Sidebar filters
    st.sidebar.header("Filtri Partita")
    leagues = sorted(df[selected_league_col].unique())
    selected_league = st.sidebar.selectbox('Seleziona Campionato', ['Tutti'] + leagues)
        
    home_teams = sorted(df['home_team_name'].unique())
    selected_home_team = st.sidebar.selectbox('Seleziona Squadra di Casa', ['Tutte'] + home_teams)

    away_teams = sorted(df['away_team_name'].unique())
    selected_away_team = st.sidebar.selectbox('Seleziona Squadra in Trasferta', ['Tutte'] + away_teams)

    # New odds filters
    st.sidebar.markdown("---")
    st.sidebar.header("Filtri Quote")
    min_home_odds, max_home_odds = st.sidebar.slider(
        'Range quote Home',
        min_value=float(df[home_odds_col].min()),
        max_value=float(df[home_odds_col].max()),
        value=(float(df[home_odds_col].min()), float(df[home_odds_col].max()))
    )
    min_away_odds, max_away_odds = st.sidebar.slider(
        'Range quote Away',
        min_value=float(df[away_odds_col].min()),
        max_value=float(df[away_odds_col].max()),
        value=(float(df[away_odds_col].min()), float(df[away_odds_col].max()))
    )

    # Filter the dataframe
    filtered_df = df.copy()
    if selected_league != 'Tutti':
        filtered_df = filtered_df[filtered_df[selected_league_col] == selected_league]
    if selected_home_team != 'Tutte':
        filtered_df = filtered_df[filtered_df['home_team_name'] == selected_home_team]
    if selected_away_team != 'Tutte':
        filtered_df = filtered_df[filtered_df['away_team_name'] == selected_away_team]
    
    # Apply odds filters
    filtered_df = filtered_df[
        (filtered_df[home_odds_col] >= min_home_odds) & (filtered_df[home_odds_col] <= max_home_odds) &
        (filtered_df[away_odds_col] >= min_away_odds) & (filtered_df[away_odds_col] <= max_away_odds)
    ]

    # Main Page Inputs
    st.header("Impostazioni per l'Analisi Statistica")

    st.subheader("Primo Gol")
    first_goal_score = st.radio("Risultato del Primo Gol", ['1-0', '0-1'])
    first_goal_timebands = st.selectbox(
        "Fascia oraria del Primo Gol",
        ['Nessuno'] + ['0-5', '0-10', '11-20', '21-30', '31-39', '40-45', '46-55', '56-65', '66-75', '75-80', '75-90', '80-90']
    )

    st.subheader("Secondo Gol (Opzionale)")
    has_second_goal = st.checkbox("Considera il secondo gol?")
    second_goal_score = None
    second_goal_timebands = 'Nessuno'

    if has_second_goal:
        second_goal_score = st.radio("Risultato del Secondo Gol", ['2-0', '0-2', '1-1'])
        second_goal_timebands = st.selectbox(
            "Fascia oraria del Secondo Gol",
            ['Nessuno'] + ['0-5', '0-10', '11-20', '21-30', '31-39', '40-45', '46-55', '56-65', '66-75', '75-80', '75-90', '80-90']
        )

    st.subheader("Stato Attuale della Partita")
    min_start = st.slider("Minuto dal quale partire con le statistiche", 0, 90, 45)
    current_score = st.text_input("Risultato attuale (es. 1-0)", "0-0")

    st.markdown("---")

    # Logic to filter based on goal events
    final_df = filtered_df.copy()

    if first_goal_timebands != 'Nessuno':
        min_start_fg, min_end_fg = map(int, first_goal_timebands.split('-'))
        
        temp_df = []
        for _, row in final_df.iterrows():
            home_timings = parse_goal_timings(row['home_team_goal_timings'])
            away_timings = parse_goal_timings(row['away_team_goal_timings'])
            
            if len(home_timings) + len(away_timings) >= 1:
                first_goal_min = min(home_timings + away_timings)
                
                if first_goal_min in home_timings:
                    goal_scorer = 'home'
                else:
                    goal_scorer = 'away'
                
                is_valid_timing = min_start_fg <= first_goal_min <= min_end_fg
                is_valid_score = (first_goal_score == '1-0' and goal_scorer == 'home') or \
                                 (first_goal_score == '0-1' and goal_scorer == 'away')
                
                if is_valid_timing and is_valid_score:
                    temp_df.append(row)
        
        final_df = pd.DataFrame(temp_df)

    if has_second_goal and second_goal_timebands != 'Nessuno':
        min_start_sg, min_end_sg = map(int, second_goal_timebands.split('-'))
        home_sg_count, away_sg_count = map(int, second_goal_score.split('-'))

        temp_df = []
        for _, row in final_df.iterrows():
            home_timings = parse_goal_timings(row['home_team_goal_timings'])
            away_timings = parse_goal_timings(row['away_team_goal_timings'])
            
            if len(home_timings) + len(away_timings) >= 2:
                all_timings = sorted(home_timings + away_timings)
                second_goal_min = all_timings[1]

                home_goals_at_second_goal = len([m for m in home_timings if m <= second_goal_min])
                away_goals_at_second_goal = len([m for m in away_timings if m <= second_goal_min])

                is_valid_timing = min_start_sg <= second_goal_min <= min_end_sg
                is_valid_score = (home_goals_at_second_goal == home_sg_count and away_goals_at_second_goal == away_sg_count)
                
                if is_valid_timing and is_valid_score:
                    temp_df.append(row)

        final_df = pd.DataFrame(temp_df)

    # Adjust data based on current score
    if current_score != "0-0":
        try:
            current_home_goals, current_away_goals = map(int, current_score.split('-'))
            
            temp_df = []
            for _, row in final_df.iterrows():
                home_goals = len([t for t in parse_goal_timings(row['home_team_goal_timings']) if t <= min_start])
                away_goals = len([t for t in parse_goal_timings(row['away_team_goal_timings']) if t <= min_start])
                
                if home_goals == current_home_goals and away_goals == current_away_goals:
                    temp_df.append(row)
            final_df = pd.DataFrame(temp_df)

        except ValueError:
            st.error("Formato del risultato attuale non valido. Usa il formato 'X-Y'.")
            final_df = pd.DataFrame()

    # Display results
    st.header("Risultati dell'Analisi")

    if final_df.empty:
        st.warning("Nessuna partita trovata che corrisponda ai criteri di ricerca.")
    else:
        st.write(f"Numero di partite trovate: **{len(final_df)}**")
        
        # Calculate FT win rates
        total_matches = len(final_df)
        home_wins = (final_df['home_team_goal_count'] > final_df['away_team_goal_count']).sum()
        draws = (final_df['home_team_goal_count'] == final_df['away_team_goal_count']).sum()
        away_wins = (final_df['home_team_goal_count'] < final_df['away_team_goal_count']).sum()
        
        winrate_data = {
            'Statistica': ['Home Win', 'Draw', 'Away Win'],
            'Valore (%)': [
                round(home_wins / total_matches * 100, 2),
                round(draws / total_matches * 100, 2),
                round(away_wins / total_matches * 100, 2)
            ]
        }
        winrate_df = pd.DataFrame(winrate_data)
        st.subheader("Winrate FT")
        st.dataframe(winrate_df.style.background_gradient(cmap='Greens', subset=['Valore (%)']))

        over_results = {}
        for threshold in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
            over_count = (final_df['total_goal_count'] > threshold).sum()
            over_results[f"Over {threshold}"] = round(over_count / total_matches * 100, 2)
        
        over_df = pd.DataFrame(over_results.items(), columns=['Statistica', 'Valore (%)'])
        st.subheader("Over FT")
        st.dataframe(over_df.style.background_gradient(cmap='Blues', subset=['Valore (%)']))

        st.subheader("Fasce Orarie dei Gol Successivi")
        all_goals_after_start = []
        
        for _, row in final_df.iterrows():
            all_timings = parse_goal_timings(row['home_team_goal_timings']) + parse_goal_timings(row['away_team_goal_timings'])
            goals_after_start = [t for t in all_timings if t > min_start]
            all_goals_after_start.extend(goals_after_start)

        time_bands = {
            '1-15': 0, '16-30': 0, '31-45': 0, '45+': 0, '46-60': 0, '61-75': 0, '76-90': 0, '90+': 0
        }

        for goal_min in all_goals_after_start:
            if 1 <= goal_min <= 15:
                time_bands['1-15'] += 1
            elif 16 <= goal_min <= 30:
                time_bands['16-30'] += 1
            elif 31 <= goal_min <= 45:
                time_bands['31-45'] += 1
            elif 45 < goal_min <= 45 + 5:
                time_bands['45+'] += 1
            elif 46 <= goal_min <= 60:
                time_bands['46-60'] += 1
            elif 61 <= goal_min <= 75:
                time_bands['61-75'] += 1
            elif 76 <= goal_min <= 90:
                time_bands['76-90'] += 1
            elif goal_min > 90:
                time_bands['90+'] += 1
        
        time_bands_df = pd.DataFrame(time_bands.items(), columns=['Fascia Oraria', 'Numero di Gol'])
        st.dataframe(time_bands_df.style.background_gradient(cmap='Oranges', subset=['Numero di Gol']))

else:
    st.info("Per iniziare, carica un file CSV usando il pannello a sinistra. L'app configurerà automaticamente i filtri successivi.")
