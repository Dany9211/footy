import streamlit as st
import pandas as pd
import re

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Analisi Statistica sui Gol", layout="wide")
st.title("⚽ Analisi Statistica sui Gol")
st.markdown("Usa i filtri a sinistra per affinare la tua analisi.")
st.markdown("---")

# -----------------------------
# Helpers
# -----------------------------
def parse_goal_timings(timings_str: str):
    """
    Converte stringhe di tempi gol in minuti (float) ordinati e unici.

    Regole:
    - "45+X" o "45'X" -> 45 + X/100 (es. 45.01)
    - "90+X" o "90'X" -> 90 + X/100 (es. 90.02)
    - "M" semplice (es. "23") -> 23.0
    - "M+X" generico (non 45/90) -> M + X (fallback raro)
    """
    if pd.isna(timings_str) or str(timings_str).strip() == "":
        return []

    items = re.split(r"\s*,\s*", str(timings_str).strip())
    result = []
    for it in items:
        it = it.strip()
        # Match pattern tipo: 45+2, 45'2, 23+1, 90+3, 12
        m = re.match(r"^(\d+)(?:[+'\s](\d+))?$", it)
        if not m:
            # tenta numero secco
            try:
                result.append(float(int(it)))
            except Exception:
                continue
            continue

        base = int(m.group(1))
        extra = m.group(2)
        if extra is not None:
            x = int(extra)
            if base == 45 or base == 90:
                minute = base + (x / 100.0)  # 45.01... 90.01...
            else:
                # fallback: se capita 30+1 -> 31
                minute = float(base + x)
        else:
            minute = float(base)

        result.append(minute)

    # unici e ordinati
    result = sorted(set(result))
    return result


def minute_to_band(minute: float) -> str:
    """
    Mappa un minuto (float) in una fascia senza overlap.
    """
    if 1 <= minute <= 15:
        return '1-15'
    if 16 <= minute <= 30:
        return '16-30'
    if 31 <= minute <= 45.0:
        return '31-45'
    if 45.0 < minute < 46.0:
        return '45+'
    if 46.0 <= minute <= 60.0:
        return '46-60'
    if 61.0 <= minute <= 75.0:
        return '61-75'
    if 76.0 <= minute <= 90.0:
        return '76-90'
    if 90.0 < minute < 91.0:
        return '90+'
    # oltre 91 consideralo comunque 90+
    if minute >= 91.0:
        return '90+'
    # <1 fuori fascia: metti in 1-15
    return '1-15'


def goals_up_to_cutoff(home_timings, away_timings, cutoff: float, second_half_start_rule=True):
    """
    Conta i gol fino al cutoff per valutare il 'risultato attuale'.

    Se second_half_start_rule è True e cutoff >= 46:
        - conta i gol con minuto < cutoff (strict), così un gol al 46.0 non è incluso.
    Se cutoff < 46:
        - include i gol con minuto <= cutoff (inclusive).
    """
    if second_half_start_rule and cutoff >= 46.0:
        h = sum(1 for t in home_timings if t < cutoff)
        a = sum(1 for t in away_timings if t < cutoff)
    else:
        h = sum(1 for t in home_timings if t <= cutoff)
        a = sum(1 for t in away_timings if t <= cutoff)
    return h, a


def ordered_goals_with_side(home_timings, away_timings):
    """
    Ritorna lista ordinata di tuple (minute, side) con side in {'home','away'}.
    """
    tagged = [(t, 'home') for t in home_timings] + [(t, 'away') for t in away_timings]
    tagged.sort(key=lambda x: x[0])
    return tagged


# -----------------------------
# Sidebar: file
# -----------------------------
st.sidebar.header("Carica il tuo file")
uploaded_file = st.sidebar.file_uploader("Carica il tuo file CSV", type="csv")

if uploaded_file is None:
    st.info("Per iniziare, carica un file CSV usando il pannello a sinistra. L'app configurerà automaticamente i filtri successivi.")
    st.stop()

# -----------------------------
# Load data
# -----------------------------
try:
    df = pd.read_csv(uploaded_file, delimiter=';')
except Exception:
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)  # fallback delimiter auto

# Normalizza alcune colonne comuni
if 'Game Week' in df.columns:
    df = df.rename(columns={'Game Week': 'Game_Week'})

if 'status' in df.columns:
    df = df[df['status'] == 'complete'].copy()

league_col = 'league'
home_odds_col = 'odds_ft_home_team_win'
away_odds_col = 'odds_ft_away_team_win'

# Precompute goal timings
df['home_parsed'] = df.get('home_team_goal_timings', '').apply(parse_goal_timings)
df['away_parsed'] = df.get('away_team_goal_timings', '').apply(parse_goal_timings)
df['all_parsed'] = df.apply(lambda r: sorted(set(r['home_parsed'] + r['away_parsed'])), axis=1)

# -----------------------------
# Sidebar: filtri base
# -----------------------------
st.sidebar.header("Filtri Partita")

leagues = ['Tutti']
if league_col in df.columns:
    leagues.extend(sorted(df[league_col].dropna().unique()))
selected_league = st.sidebar.selectbox('Seleziona Campionato', leagues)

home_teams = ['Tutte']
if 'home_team_name' in df.columns:
    home_teams.extend(sorted(df['home_team_name'].dropna().unique()))
selected_home_team = st.sidebar.selectbox('Seleziona Squadra di Casa', home_teams)

away_teams = ['Tutte']
if 'away_team_name' in df.columns:
    away_teams.extend(sorted(df['away_team_name'].dropna().unique()))
selected_away_team = st.sidebar.selectbox('Seleziona Squadra in Trasferta', away_teams)

filtered_df = df.copy()
if selected_league != 'Tutti' and league_col in filtered_df.columns:
    filtered_df = filtered_df[filtered_df[league_col] == selected_league]
if selected_home_team != 'Tutte' and 'home_team_name' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['home_team_name'] == selected_home_team]
if selected_away_team != 'Tutte' and 'away_team_name' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['away_team_name'] == selected_away_team]

# -----------------------------
# Main inputs
# -----------------------------
st.header("Impostazioni per l'Analisi Statistica")

# Quote
st.subheader("Filtri Quote")
if home_odds_col in filtered_df.columns and away_odds_col in filtered_df.columns:
    colq1, colq2, colq3, colq4 = st.columns(4)
    with colq1:
        min_home_odds = st.number_input('Quota minima Home', min_value=1.0, value=1.0, step=0.01)
    with colq2:
        max_home_odds = st.number_input('Quota massima Home', min_value=1.0, value=50.0, step=0.01)
    with colq3:
        min_away_odds = st.number_input('Quota minima Away', min_value=1.0, value=1.0, step=0.01)
    with colq4:
        max_away_odds = st.number_input('Quota massima Away', min_value=1.0, value=50.0, step=0.01)

    if min_home_odds > max_home_odds or min_away_odds > max_away_odds:
        st.error("Controlla gli intervalli delle quote: min non può superare max.")
    else:
        filtered_df = filtered_df[
            (filtered_df[home_odds_col] >= min_home_odds) & (filtered_df[home_odds_col] <= max_home_odds) &
            (filtered_df[away_odds_col] >= min_away_odds) & (filtered_df[away_odds_col] <= max_away_odds)
        ]
else:
    st.warning(f"Colonne quote non trovate: '{home_odds_col}' o '{away_odds_col}'. Filtri quote disattivati.")

# Primo gol
st.subheader("Primo Gol")
first_goal_score = st.radio("Risultato del Primo Gol", ['1-0', '0-1'])
first_goal_timebands = st.selectbox(
    "Fascia oraria del Primo Gol",
    ['Nessuno', '0-5', '0-10', '11-20', '21-30', '31-39', '40-45', '46-55', '56-65', '66-75', '75-80', '75-90', '80-90']
)

# Secondo gol (opzionale)
st.subheader("Secondo Gol (Opzionale)")
has_second_goal = st.checkbox("Considera il secondo gol?")
second_goal_score = None
second_goal_timebands = 'Nessuno'
if has_second_goal:
    second_goal_score = st.radio("Risultato del Secondo Gol", ['2-0', '0-2', '1-1'])
    second_goal_timebands = st.selectbox(
        "Fascia oraria del Secondo Gol",
        ['Nessuno', '0-5', '0-10', '11-20', '21-30', '31-39', '40-45', '46-55', '56-65', '66-75', '75-80', '75-90', '80-90']
    )

# Stato attuale & cursore
st.subheader("Stato Attuale della Partita")
min_start = st.slider("Minuto dal quale partire con le statistiche", 0, 90, 45)
current_score = st.text_input("Risultato attuale (es. 1-0)", "0-0")

# -----------------------------
# Filtraggio principale
# -----------------------------
st.markdown("---")
st.header("Risultati dell'Analisi")

# Valida current_score
cur_h, cur_a = 0, 0
if current_score != "0-0":
    try:
        cur_h, cur_a = map(int, current_score.strip().split('-'))
    except Exception:
        st.error("Formato del risultato attuale non valido. Usa il formato 'X-Y'.")
        st.stop()

filtered_rows = []
for idx, row in filtered_df.iterrows():
    home_t = row['home_parsed']
    away_t = row['away_parsed']
    all_t = sorted(set(home_t + away_t))

    # 1) Primo gol (se richiesto)
    first_ok = True
    if first_goal_timebands != 'Nessuno':
        if len(all_t) == 0:
            first_ok = False
        else:
            first_min, first_side = ordered_goals_with_side(home_t, away_t)[0]
            
            def in_req_band(m, label):
                a, b = label.split('-')
                return float(a) <= m <= float(b)

            band_ok = in_req_band(first_min, first_goal_timebands)
            
            side_ok = (first_goal_score == '1-0' and first_side == 'home') or \
                      (first_goal_score == '0-1' and first_side == 'away')
            first_ok = band_ok and side_ok

    if not first_ok:
        continue

    # 2) Secondo gol (se richiesto)
    second_ok = True
    if has_second_goal and second_goal_timebands != 'Nessuno':
        tagged = ordered_goals_with_side(home_t, away_t)
        if len(tagged) < 2:
            second_ok = False
        else:
            second_min, _ = tagged[1]
            def in_req_band(m, label):
                a, b = label.split('-')
                return float(a) <= m <= float(b)
            band_ok = in_req_band(second_min, second_goal_timebands)

            h_to_second = sum(1 for t in home_t if t <= second_min)
            a_to_second = sum(1 for t in away_t if t <= second_min)
            try:
                exp_h, exp_a = map(int, second_goal_score.split('-'))
            except Exception:
                second_ok = False
            else:
                score_ok = (h_to_second == exp_h and a_to_second == exp_a)
                second_ok = band_ok and score_ok

    if not second_ok:
        continue

    # 3) Coerenza risultato attuale al cutoff (regola strict dal 46 in poi)
    h_before, a_before = goals_up_to_cutoff(home_t, away_t, float(min_start), second_half_start_rule=True)
    if not (h_before == cur_h and a_before == cur_a):
        continue

    filtered_rows.append(row)

final_df = pd.DataFrame(filtered_rows)

# -----------------------------
# Output
# -----------------------------
if final_df.empty:
    st.warning("Nessuna partita trovata che corrisponde ai criteri di ricerca.")
    st.stop()

st.write(f"Numero di partite trovate: **{len(final_df)}**")

st.subheader("Anteprima Partite Filtrate")
cols_preview = [c for c in [
    'home_team_name', 'away_team_name',
    'home_team_goal_count', 'away_team_goal_count',
    'home_team_goal_timings', 'away_team_goal_timings'
] if c in final_df.columns]
if cols_preview:
    st.dataframe(final_df[cols_preview])
else:
    st.dataframe(final_df.head(50))

# Winrate FT
total_matches = len(final_df)
if 'home_team_goal_count' in final_df.columns and 'away_team_goal_count' in final_df.columns:
    home_wins = (final_df['home_team_goal_count'] > final_df['away_team_goal_count']).sum()
    draws = (final_df['home_team_goal_count'] == final_df['away_team_goal_count']).sum()
    away_wins = (final_df['home_team_goal_count'] < final_df['away_team_goal_count']).sum()

    winrate_data = {
        'Statistica': ['Home Win', 'Draw', 'Away Win'],
        '%': [
            round(home_wins / total_matches * 100, 2),
            round(draws / total_matches * 100, 2),
            round(away_wins / total_matches * 100, 2)
        ]
    }
    winrate_df = pd.DataFrame(winrate_data)
    st.subheader("Winrate FT")
    st.dataframe(winrate_df.style.background_gradient(cmap='RdYlGn', subset=['%']))

# Over FT
if 'total_goal_count' not in final_df.columns and \
   'home_team_goal_count' in final_df.columns and 'away_team_goal_count' in final_df.columns:
    final_df['total_goal_count'] = final_df['home_team_goal_count'] + final_df['away_team_goal_count']

if 'total_goal_count' in final_df.columns:
    over_results = {}
    for threshold in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
        over_count = (final_df['total_goal_count'] > threshold).sum()
        over_results[f"Over {threshold}"] = round(over_count / total_matches * 100, 2)

    over_df = pd.DataFrame(over_results.items(), columns=['Statistica', '%'])
    st.subheader("Over FT")
    st.dataframe(over_df.style.background_gradient(cmap='RdYlGn', subset=['%']))

# Fasce orarie dei gol successivi (dal cutoff in poi)
st.subheader("Fasce Orarie dei Gol Successivi")
time_bands = {
    '1-15': 0, '16-30': 0, '31-45': 0, '45+': 0, '46-60': 0, '61-75': 0, '76-90': 0, '90+': 0
}
time_band_matches = {k: set() for k in time_bands.keys()}

for _, row in final_df.iterrows():
    match_id = row.get('timestamp', str(row.name))
    all_t = row['all_parsed']
    for t in all_t:
        if t > float(min_start):
            band = minute_to_band(t)
            time_bands[band] += 1
            time_band_matches[band].add(match_id)
            
tb_df = pd.DataFrame(time_bands.items(), columns=['Fascia Oraria', 'Numero di Gol'])

time_band_matches = {
    '1-15': set(), '16-30': set(), '31-45': set(), '45+': set(), '46-60': set(), '61-75': set(), '76-90': set(), '90+': set()
}

for _, row in final_df.iterrows():
    match_id = row.get('timestamp', str(row.name))
    all_t = row['all_parsed']
    for t in all_t:
        if t > float(min_start):
            band = minute_to_band(t)
            time_band_matches[band].add(match_id)

tb_df['Numero di Partite'] = [len(time_band_matches[band]) for band in tb_df['Fascia Oraria']]
tb_df['%'] = [
    round(len(time_band_matches[band]) / total_matches * 100, 2) if total_matches > 0 else 0
    for band in tb_df['Fascia Oraria']
]

st.dataframe(tb_df[['Fascia Oraria', 'Numero di Partite', '%']].style.background_gradient(cmap='RdYlGn', subset=['%']))
