# app.py
import ast, re
import numpy as np
import pandas as pd
import streamlit as st
import psycopg2

st.set_page_config(page_title="Analisi Campionati – footy", layout="wide")
st.title("Analisi Campionati • tabella: footy")

# =============== DB ===============
@st.cache_data
def run_query(query: str):
    try:
        conn = psycopg2.connect(
            host=st.secrets["postgres"]["host"],
            port=st.secrets["postgres"]["port"],
            dbname=st.secrets["postgres"]["dbname"],
            user=st.secrets["postgres"]["user"],
            password=st.secrets["postgres"]["password"],
            sslmode="require"
        )
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Errore connessione DB: {e}")
        st.stop()
        return pd.DataFrame()

# =============== UTIL: mapping colonne / parsing ===============
ALIASES = {
    # anagrafica
    "league": ["league", "div", "campionato"],
    "anno":   ["anno", "season", "year"],
    "giornata": ["giornata", "game week", "game_week", "gw", "matchday"],

    # squadre
    "home_team": ["home_team", "home_team_name", "hometeam"],
    "away_team": ["away_team", "away_team_name", "awayteam"],

    # gol FT / HT
    "gol_home_ft": ["gol_home_ft", "home_team_goal_count", "fthg", "home_ft"],
    "gol_away_ft": ["gol_away_ft", "away_team_goal_count", "ftag", "away_ft"],
    "gol_home_ht": ["gol_home_ht", "home_team_goal_count_half_time", "hthg", "home_ht"],
    "gol_away_ht": ["gol_away_ht", "away_team_goal_count_half_time", "htag", "away_ht"],

    # minutaggi
    "minutaggio_gol": ["minutaggio_gol", "home_team_goal_timings", "goals_home_minute", "home_goal_minutes"],
    "minutaggio_gol_away": ["minutaggio_gol_away", "away_team_goal_timings", "goals_away_minute", "away_goal_minutes"],

    # quote 1X2
    "odd_home": ["odd_home", "odds_ft_home_team_win", "psch", "b365h", "whh", "vch", "lbh", "iwh", "bwh"],
    "odd_draw": ["odd_draw", "odds_ft_draw", "pscd", "b365d", "whd", "vcd", "lbd", "iwd", "bwd"],
    "odd_away": ["odd_away", "odds_ft_away_team_win", "psca", "b365a", "wha", "vca", "lba", "iwa", "bwa"],

    # extra (se presenti)
    "odds_o25": ["odds_ft_over25", "oddso25", "b365o2_5"],
    "odds_btts": ["odds_btts_yes", "odds_btts"],
}

def norm(s: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-z]+", "_", s.strip().lower())).strip("_")

def find_col(df, keys):
    cols = {norm(c): c for c in df.columns}
    for k in keys:
        nk = norm(k)
        if nk in cols: 
            return cols[nk]
    return None

def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Crea viste standard con nomi interni; non rimuove le originali."""
    out = df.copy()
    # crea colonne target se trovate
    for target, aliases in ALIASES.items():
        col = find_col(df, aliases)
        if col and target != col:
            out[target] = df[col]
    # risultati composti se mancano
    if "gol_home_ft" in out.columns and "gol_away_ft" in out.columns and "risultato_ft" not in out.columns:
        out["risultato_ft"] = out["gol_home_ft"].astype(str) + "-" + out["gol_away_ft"].astype(str)
    if "gol_home_ht" in out.columns and "gol_away_ht" in out.columns and "risultato_ht" not in out.columns:
        out["risultato_ht"] = out["gol_home_ht"].astype(str) + "-" + out["gol_away_ht"].astype(str)
    return out

def parse_min_list(value):
    """
    Accetta: "[23, 45+1, 78]" | "23;45+1;78" | "23,45+1,78" | lista.
    45+2 -> 47; 90+3 -> 93.
    """
    if pd.isna(value) or value == "": return []
    if isinstance(value, list): raw = value
    else:
        s = str(value).strip()
        if s.startswith("[") and s.endswith("]"):
            try: raw = ast.literal_eval(s)
            except: raw = re.split(r"[;,]\s*", s.strip("[] "))
        else:
            raw = re.split(r"[;,]\s*", s)
    out = []
    for x in raw:
        if x is None: continue
        xs = str(x).strip()
        if not xs: continue
        m = re.match(r"^(\d+)\s*\+\s*(\d+)$", xs)
        if m:
            out.append(int(m.group(1)) + int(m.group(2)))
        else:
            try: out.append(int(float(xs)))
            except: pass
    return out

def to_num(s):
    try: return float(str(s).replace(",", "."))
    except: return np.nan

# =============== SORGENTE DATI ===============
st.sidebar.header("Sorgente dati")
use_db = st.sidebar.radio("Origine", ["Supabase: footy", "Carica CSV"], index=0)

df = pd.DataFrame()
if use_db == "Supabase: footy":
    df = run_query('SELECT * FROM "footy";')  # <— tabella richiesta
else:
    up = st.sidebar.file_uploader("CSV pulito (opzionale)", type=["csv"])
    if up:
        df = pd.read_csv(up, dtype=str, encoding="utf-8")

if df.empty:
    st.warning("Nessun dato da mostrare.")
    st.stop()

st.caption(f"Righe caricate: **{len(df)}**")
df = map_columns(df)

# tipi numerici base
for c in ["gol_home_ft","gol_away_ft","gol_home_ht","gol_away_ht","giornata"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")

# minutaggi normalizzati
df["_home_mins"] = df["minutaggio_gol"].map(parse_min_list) if "minutaggio_gol" in df.columns else [[]]*len(df)
df["_away_mins"] = df["minutaggio_gol_away"].map(parse_min_list) if "minutaggio_gol_away" in df.columns else [[]]*len(df)

# =============== FILTRI ===============
st.sidebar.header("Filtri")
# League
if "league" in df.columns:
    leagues = ["Tutte"] + sorted([x for x in df["league"].dropna().unique()])
    selected_league = st.sidebar.selectbox("Campionato", leagues, index=0)
else:
    selected_league = "Tutte"

# Anno
if "anno" in df.columns:
    anni = ["Tutti"] + sorted([x for x in df["anno"].dropna().unique()])
    selected_anno = st.sidebar.selectbox("Anno", anni, index=0)
else:
    selected_anno = "Tutti"

# Giornata
if "giornata" in df.columns and df["giornata"].notna().any():
    gmin = int(np.nanmin(df["giornata"]))
    gmax = int(np.nanmax(df["giornata"]))
    giornata_range = st.sidebar.slider("Giornata", min_value=gmin, max_value=gmax, value=(gmin,gmax))
else:
    giornata_range = None

# Squadre (dipendono da campionato)
tmp = df if selected_league=="Tutte" else df[df["league"]==selected_league]
home_sel = st.sidebar.selectbox("Home team", ["Tutte"] + sorted(tmp["home_team"].dropna().unique()) if "home_team" in tmp.columns else ["Tutte"])
away_sel = st.sidebar.selectbox("Away team", ["Tutte"] + sorted(tmp["away_team"].dropna().unique()) if "away_team" in tmp.columns else ["Tutte"])

# Filtro per risultato HT (se presente)
if "risultato_ht" in df.columns:
    ht_vals = sorted(df["risultato_ht"].dropna().unique())
    ht_mult = st.sidebar.multiselect("Risultato HT", ht_vals, default=[])

# Range quote 1X2
def add_range(col, label):
    if col not in df.columns: return None
    series = pd.to_numeric(df[col].astype(str).replace(",", ".", regex=False), errors="coerce")
    lo = float(np.nanmin(series)) if series.notna().any() else 1.01
    hi = float(np.nanmax(series)) if series.notna().any() else 20.0
    return st.sidebar.slider(label, min_value=float(max(1.01, round(lo,2))), max_value=float(round(max(hi,1.01),2)), value=(float(round(lo,2)), float(round(min(hi,5.0),2))))

r_h = add_range("odd_home", "Quota 1 (Home)")
r_d = add_range("odd_draw", "Quota X (Draw)")
r_a = add_range("odd_away", "Quota 2 (Away)")

# Applica filtri
mask = pd.Series(True, index=df.index)
if selected_league!="Tutte" and "league" in df.columns:
    mask &= (df["league"]==selected_league)
if selected_anno!="Tutti" and "anno" in df.columns:
    mask &= (df["anno"]==selected_anno)
if giornata_range and "giornata" in df.columns:
    mask &= df["giornata"].between(giornata_range[0], giornata_range[1])
if home_sel!="Tutte" and "home_team" in df.columns:
    mask &= (df["home_team"]==home_sel)
if away_sel!="Tutte" and "away_team" in df.columns:
    mask &= (df["away_team"]==away_sel)
if ht_mult and "risultato_ht" in df.columns:
    mask &= df["risultato_ht"].isin(ht_mult)
def range_mask(col, rng):
    if col not in df.columns or not rng: return True
    ser = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
    return ser.between(rng[0], rng[1])
if r_h: mask &= range_mask("odd_home", r_h)
if r_d: mask &= range_mask("odd_draw", r_d)
if r_a: mask &= range_mask("odd_away", r_a)

d = df[mask].copy()
st.subheader("Dati filtrati")
st.write(f"**Righe visualizzate:** {len(d)}")
st.dataframe(d.head(50), use_container_width=True)

# =============== KPI base ===============
if {"gol_home_ft","gol_away_ft"} <= set(d.columns):
    d["_tg"] = d["gol_home_ft"].fillna(0) + d["gol_away_ft"].fillna(0)
    btts = ((d["gol_home_ft"]>0)&(d["gol_away_ft"]>0)).mean()*100 if len(d) else 0
    k1,k2,k3,k4,k5 = st.columns(5)
    with k1: st.metric("Media Gol", f"{d['_tg'].mean():.2f}" if len(d) else "—")
    with k2: st.metric("BTTS %", f"{btts:.1f}%")
    with k3: st.metric("Over 1.5", f"{(d['_tg']>=2).mean()*100:.1f}%")
    with k4: st.metric("Over 2.5", f"{(d['_tg']>=3).mean()*100:.1f}%")
    with k5: st.metric("Over 3.5", f"{(d['_tg']>=4).mean()*100:.1f}%")

# =============== Distribuzione minuti gol ===============
st.markdown("---")
st.subheader("Distribuzione gol per fasce (15')")
def goals_bins(minutes):
    return pd.cut(minutes, bins=[0,15,30,45,60,75,105], labels=["0-15","16-30","31-45","46-60","61-75","76-90+"], include_lowest=True, right=True)

all_mins = []
for hm, am in zip(d["_home_mins"], d["_away_mins"]):
    if isinstance(hm, list): all_mins += hm
    if isinstance(am, list): all_mins += am
ms = pd.Series(all_mins, dtype="float").dropna()
if not ms.empty:
    dist = goals_bins(ms).value_counts().sort_index().rename_axis("fascia").reset_index(name="gol")
    st.bar_chart(dist.set_index("fascia"))
else:
    st.info("Minutaggi non disponibili nel subset.")

# =============== HT–FT Matrix (stima HT da minutaggi se serve) ===============
st.subheader("Matrice HT–FT")
def res_label(h,a):
    if pd.isna(h) or pd.isna(a): return np.nan
    if h>a: return "H"
    if h<a: return "A"
    return "D"

if "gol_home_ht" in d.columns and "gol_away_ht" in d.columns:
    d["_ht"] = d.apply(lambda r: res_label(r["gol_home_ht"], r["gol_away_ht"]), axis=1)
else:
    # stima HT dai gol ≤ 48'
    def ht_sign_row(r):
        hm = [m for m in (r["_home_mins"] if isinstance(r["_home_mins"],list) else []) if m<=48]
        am = [m for m in (r["_away_mins"] if isinstance(r["_away_mins"],list) else []) if m<=48]
        return res_label(len(hm), len(am))
    d["_ht"] = d.apply(ht_sign_row, axis=1)

if {"gol_home_ft","gol_away_ft"} <= set(d.columns):
    d["_ft"] = d.apply(lambda r: res_label(r["gol_home_ft"], r["gol_away_ft"]), axis=1)
    mat = pd.crosstab(d["_ht"], d["_ft"], normalize="index")*100
    st.dataframe(mat.round(1))
else:
    st.info("Mancano i gol FT per costruire la matrice.")

# =============== ROI semplice (1X2 / O2.5 / BTTS) ===============
st.markdown("---")
st.subheader("ROI (stake 1) – selezione semplice")
d["_home_win"] = d["gol_home_ft"] > d["gol_away_ft"] if {"gol_home_ft","gol_away_ft"} <= set(d.columns) else False
d["_draw"]     = d["gol_home_ft"] == d["gol_away_ft"] if {"gol_home_ft","gol_away_ft"} <= set(d.columns) else False
d["_away_win"] = d["gol_home_ft"] < d["gol_away_ft"] if {"gol_home_ft","gol_away_ft"} <= set(d.columns) else False
d["_over25"]   = (d["gol_home_ft"] + d["gol_away_ft"] >= 3) if {"gol_home_ft","gol_away_ft"} <= set(d.columns) else False
d["_btts"]     = ((d["gol_home_ft"]>0)&(d["gol_away_ft"]>0)) if {"gol_home_ft","gol_away_ft"} <= set(d.columns) else False

def roi_from_bets(sel, odds_col, outcome_col):
    if sel.empty or odds_col not in sel.columns: return (0,0.0,0.0,0.0)
    bets = sel.copy()
    bets["_o"] = bets[odds_col].map(lambda v: float(str(v).replace(",", ".")) if pd.notna(v) and str(v).strip()!="" else np.nan)
    bets = bets[~bets["_o"].isna()]
    if bets.empty: return (0,0.0,0.0,0.0)
    bets["_win"] = bets[outcome_col].astype(bool)
    bets["_pl"] = np.where(bets["_win"], bets["_o"]-1.0, -1.0)
    n = len(bets); strike = bets["_win"].mean()*100; profit = bets["_pl"].sum(); roi = (profit/n)*100
    return int(n), strike, profit, roi

colx1,colx2,colx3,colx4 = st.columns([1.2,1,1,1.2])
with colx1:
    strat = st.selectbox("Mercato", ["1 (Casa)","X (Pareggio)","2 (Trasferta)","Over 2.5","BTTS Sì"])
with colx2:
    omin = st.number_input("Quota min", value=1.01, step=0.01)
with colx3:
    omax = st.number_input("Quota max", value=10.00, step=0.01)
with colx4:
    only_side = st.selectbox("Solo lato coinvolto (Home/Away)", ["No","Sì"])

sel = d.copy()
if strat=="1 (Casa)" and "odd_home" in d.columns:
    if only_side=="Sì" and "home_team" in d.columns and home_sel!="Tutte":
        sel = sel[sel["home_team"]==home_sel]
    sel = sel[(pd.to_numeric(sel["odd_home"].astype(str).str.replace(",","."), errors="coerce").between(omin, omax))]
    n,sr,p,roi = roi_from_bets(sel, "odd_home", "_home_win")
elif strat=="X (Pareggio)" and "odd_draw" in d.columns:
    sel = sel[(pd.to_numeric(sel["odd_draw"].astype(str).str.replace(",","."), errors="coerce").between(omin, omax))]
    n,sr,p,roi = roi_from_bets(sel, "odd_draw", "_draw")
elif strat=="2 (Trasferta)" and "odd_away" in d.columns:
    if only_side=="Sì" and "away_team" in d.columns and away_sel!="Tutte":
        sel = sel[sel["away_team"]==away_sel]
    sel = sel[(pd.to_numeric(sel["odd_away"].astype(str).str.replace(",","."), errors="coerce").between(omin, omax))]
    n,sr,p,roi = roi_from_bets(sel, "odd_away", "_away_win")
elif strat=="Over 2.5" and "odds_o25" in d.columns:
    sel = sel[(pd.to_numeric(sel["odds_o25"].astype(str).str.replace(",","."), errors="coerce").between(omin, omax))]
    n,sr,p,roi = roi_from_bets(sel, "odds_o25", "_over25")
elif strat=="BTTS Sì" and "odds_btts" in d.columns:
    sel = sel[(pd.to_numeric(sel["odds_btts"].astype(str).str.replace(",","."), errors="coerce").between(omin, omax))]
    n,sr,p,roi = roi_from_bets(sel, "odds_btts", "_btts")
else:
    n,sr,p,roi = (0,0.0,0.0,0.0)
st.write(f"**Bets**: {n} | **Strike**: {sr:.1f}% | **P/L**: {p:.2f} | **ROI**: {roi:.1f}%")

# =============== Riepilogo per Anno ===============
st.markdown("---")
st.subheader("Riepilogo partite per Anno")
if "anno" in d.columns and not d.empty:
    rie = d["anno"].value_counts().sort_index().rename_axis("Anno").reset_index(name="Partite")
    st.table(rie)
else:
    st.info("Colonna 'anno' non disponibile o nessun dato filtrato.")
