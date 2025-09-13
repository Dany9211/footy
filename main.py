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
    
    # Automatically select 'league' if it exists
    league_col_index = all_columns.index('league') if 'league' in all_columns else 0
    selected_league_col = st.sidebar.selectbox(
        'Seleziona la colonna del campionato',
        all_columns,
        index=league_col_index
    )
    
    # New odds column selectors in sidebar
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

    # Filter the dataframe
    filtered_df = df.copy()
    if selected_league != 'Tutti':
        filtered_df = filtered_df[filtered_df[selected_league_col] == selected_league]
    if selected_home_team != 'Tutte':
        filtered_df = filtered_df[filtered_df['home_team_name'] == selected_home_team]
    if selected_away_team != 'Tutte':
        filtered_df = filtered_df[filtered_df['away_team_name'] == selected_away_team]
    
    # Main Page Inputs
    st.header("Impostazioni per l'Analisi Statistica")

    # New odds filters in the main section
    st.subheader("Filtri Quote")
    try:
        min_home_odds = st.number_input('Quota minima Home', min_value=1.0, value=1.0, step=0.01)
        max_home_odds = st.number_input('Quota massima Home', min_value=1.0, value=50.0, step=0.01)
        min_away_odds = st.number_input('Quota minima Away', min_value=1.0, value=1.0, step=0.01)
        max_away_odds = st.number_input('Quota massima Away', min_value=1.0, value=50.0, step=0.01)

        # Check for invalid ranges
        if min_home_odds > max_home_odds:
            st.error("La quota minima Home non può essere maggiore di quella massima.")
        if min_away_odds > max_away_odds:
            st.error("La quota minima Away non può essere maggiore di quella massima.")
            
        # Apply odds filters
        filtered_df = filtered_df[
            (filtered_df[home_odds_col] >= min_home_odds) & (filtered_df[home_odds_col] <= max_home_odds) &
            (filtered_df[away_odds_col] >= min_away_odds) & (filtered_df[away_odds_col] <= max_away_odds)
        ]
    except KeyError:
        st.warning("Assicurati di aver selezionato le colonne corrette per le quote nella barra laterale.")
    except ValueError:
        st.warning("Le colonne selezionate per le quote non contengono valori numerici validi. Prova a selezionare altre colonne.")


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
    
    # Display Filter Summary
    st.subheader("Riepilogo Filtri Applicati")
    summary = []
    if selected_league != 'Tutti': summary.append(f"Campionato: **{selected_league}**")
    if selected_home_team != 'Tutte': summary.append(f"Squadra di casa: **{selected_home_team}**")
    if selected_away_team != 'Tutte': summary.append(f"Squadra in trasferta: **{selected_away_team}**")
    if first_goal_timebands != 'Nessuno': summary.append(f"Primo gol ({first_goal_score}) nella fascia **{first_goal_timebands}**")
    if has_second_goal and second_goal_timebands != 'Nessuno': summary.append(f"Secondo gol ({second_goal_score}) nella fascia **{second_goal_timebands}**")
    if current_score != "0-0": summary.append(f"Risultato attuale al minuto **{min_start}** è **{current_score}**")
    if not (min_home_odds == 1.0 and max_home_odds == 50.0): summary.append(f"Quota Home: **{min_home_odds} - {max_home_odds}**")
    if not (min_away_odds == 1.0 and max_away_odds == 50.0): summary.append(f"Quota Away: **{min_away_odds} - {max_away_odds}**")
    
    if summary:
        for item in summary: st.markdown(f"- {item}")
    else:
        st.info("Nessun filtro specifico applicato.")


    # Logic to filter based on goal events
    final_df = filtered_df.copy()

    # Apply first goal filter
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
    
    # Apply second goal filter
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
        st.warning("Nessuna partita trovata che corrisponde ai criteri di ricerca.")
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
        time_band_matches = {
            '1-15': set(), '16-30': set(), '31-45': set(), '45+': set(), '46-60': set(), '61-75': set(), '76-90': set(), '90+': set()
        }
        
        for _, row in final_df.iterrows():
            match_id = row['timestamp']
            all_timings = parse_goal_timings(row['home_team_goal_timings']) + parse_goal_timings(row['away_team_goal_timings'])
            for goal_min in all_timings:
                if goal_min > min_start:
                    if 1 <= goal_min <= 15:
                        time_bands['1-15'] += 1
                        time_band_matches['1-15'].add(match_id)
                    elif 16 <= goal_min <= 30:
                        time_bands['16-30'] += 1
                        time_band_matches['16-30'].add(match_id)
                    elif 31 <= goal_min <= 45:
                        time_bands['31-45'] += 1
                        time_band_matches['31-45'].add(match_id)
                    elif 45 < goal_min <= 45 + 5:
                        time_bands['45+'] += 1
                        time_band_matches['45+'].add(match_id)
                    elif 46 <= goal_min <= 60:
                        time_bands['46-60'] += 1
                        time_band_matches['46-60'].add(match_id)
                    elif 61 <= goal_min <= 75:
                        time_bands['61-75'] += 1
                        time_band_matches['61-75'].add(match_id)
                    elif 76 <= goal_min <= 90:
                        time_bands['76-90'] += 1
                        time_band_matches['76-90'].add(match_id)
                    elif goal_min > 90:
                        time_bands['90+'] += 1
                        time_band_matches['90+'].add(match_id)
        
        time_bands_df = pd.DataFrame(time_bands.items(), columns=['Fascia Oraria', 'Numero di Gol'])
        
        total_matches_for_percentage = len(final_df)
        time_bands_df['Percentuale Partite (%)'] = [
            round(len(time_band_matches[band]) / total_matches_for_percentage * 100, 2)
            if total_matches_for_percentage > 0 else 0
            for band in time_bands_df['Fascia Oraria']
        ]
        
        st.dataframe(time_bands_df.style.background_gradient(cmap='Oranges', subset=['Percentuale Partite (%)']))
else:
    st.info("Per iniziare, carica un file CSV usando il pannello a sinistra. L'app configurerà automaticamente i filtri successivi.")
