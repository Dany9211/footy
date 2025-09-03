
# --- SEZIONE 6: Analisi Pattern (Live: 1° gol, 2° gol, risultato attuale + minuto) ---
try:
    import pandas as pd
    import re as _re

    st.subheader("6. Analisi Pattern (Live)")

    def _camp_col_name(_df):
        for cc in ["Campionato", "campionato", "League", "league", "Camp"]:
            if isinstance(_df, pd.DataFrame) and cc in _df.columns:
                return cc
        return None

    def _find_goal_time_cols(_df):
        pairs = [
            ("GoalTimes_Home", "GoalTimes_Away"),
            ("Minuti_Gol_Home", "Minuti_Gol_Away"),
            ("GolMinuti_Home", "GolMinuti_Away"),
            ("Home_Goals_Times", "Away_Goals_Times"),
            ("HomeGoalsMinutes", "AwayGoalsMinutes"),
        ]
        for h,a in pairs:
            if h in _df.columns and a in _df.columns:
                return h,a
        home_col = None; away_col = None
        for c in _df.columns:
            cl = str(c).lower()
            if ("home" in cl or "casa" in cl) and any(k in cl for k in ["min", "time", "gol", "goal"]):
                if home_col is None:
                    home_col = c
            if ("away" in cl or "trasferta" in cl) and any(k in cl for k in ["min", "time", "gol", "goal"]):
                if away_col is None:
                    away_col = c
        return home_col, away_col

    def _parse_minute_token(tok):
        s = str(tok).strip()
        if not s:
            return None
        m = _re.search(r"(\d+)(?:\s*\+\s*(\d+))?", s)
        if not m:
            return None
        base = int(m.group(1))
        extra = int(m.group(2)) if m.group(2) else 0
        return base + extra

    def _parse_goal_times(val):
        if val is None:
            return []
        if isinstance(val, list):
            tokens = val
        else:
            s = str(val)
            if s.startswith("[") and s.endswith("]"):
                s2 = s.strip("[]")
                tokens = [t.strip() for t in _re.split(r"[;,]", s2) if t.strip()]
            else:
                tokens = [t.strip() for t in _re.split(r"[;,]", s) if t.strip()]
        mins = []
        for t in tokens:
            m = _parse_minute_token(t)
            if m is not None:
                mins.append(m)
        mins.sort()
        return mins

    _df_base = None
    if 'filtered_df' in globals() and isinstance(filtered_df, pd.DataFrame) and not filtered_df.empty:
        _df_base = filtered_df.copy()
    elif 'df' in globals() and isinstance(df, pd.DataFrame) and not df.empty:
        _df_base = df.copy()

    if _df_base is None or _df_base.empty:
        st.info("Carica dati e applica i filtri a sinistra per avviare l'Analisi Pattern.")
    else:
        _selected_league = None
        try:
            if 'filters' in globals() and isinstance(filters, dict) and 'Campionato' in filters:
                _selected_league = filters['Campionato']
        except Exception:
            pass
        if isinstance(_selected_league, (list, tuple, set)):
            _selected_league = list(_selected_league)[0] if _selected_league else None

        _camp_col = _camp_col_name(_df_base)
        if _camp_col and _selected_league not in (None, "Tutti", "All"):
            _df_base = _df_base[_df_base[_camp_col] == _selected_league]

        if _df_base.empty:
            st.info("Nessuna partita nel campionato selezionato con i filtri attuali.")
        else:
            home_times_col, away_times_col = _find_goal_time_cols(_df_base)
            if not home_times_col or not away_times_col:
                st.warning("Colonne minuti gol non trovate. Attese ad es.: 'GoalTimes_Home'/'GoalTimes_Away' o simili.")
            else:
                c1,c2,c3 = st.columns(3)
                with c1:
                    first_team = st.selectbox("Primo gol di", ["Qualsiasi","Home","Away"], index=0, key="pat_first_team")
                with c2:
                    fg_min = st.number_input("Primo gol da min", min_value=0, max_value=120, value=1, step=1, key="pat_fg_min")
                with c3:
                    fg_max = st.number_input("Primo gol a min", min_value=0, max_value=120, value=45, step=1, key="pat_fg_max")
                if fg_max < fg_min: fg_max = fg_min

                enable_second = st.checkbox("Filtra anche il SECONDO gol", value=False, key="pat_enable_second")
                if enable_second:
                    d1,d2,d3 = st.columns(3)
                    with d1:
                        second_team = st.selectbox("Secondo gol di", ["Qualsiasi","Home","Away"], index=0, key="pat_second_team")
                    with d2:
                        sg_min = st.number_input("Secondo gol da min", min_value=0, max_value=120, value=1, step=1, key="pat_sg_min")
                    with d3:
                        sg_max = st.number_input("Secondo gol a min", min_value=0, max_value=120, value=90, step=1, key="pat_sg_max")
                    if sg_max < sg_min: sg_max = sg_min

                score_opts = ["Qualsiasi"] + [f"{h}-{a}" for h in range(0,6) for a in range(0,6)]
                colm1, colm2 = st.columns([1,2])
                with colm1:
                    score_now = st.selectbox("Risultato attuale", score_opts, index=0, key="pat_score_now")
                with colm2:
                    minute_now = st.slider("Minuto corrente", min_value=0, max_value=120, value=30, step=1, key="pat_minute_now")

                _work = _df_base.copy()
                _work["_home_times"] = _work[home_times_col].apply(_parse_goal_times)
                _work["_away_times"] = _work[away_times_col].apply(_parse_goal_times)

                def _first_second(row):
                    events = [(m,"Home") for m in row["_home_times"]] + [(m,"Away") for m in row["_away_times"]]
                    events.sort(key=lambda x: x[0])
                    fg = events[0] if events else (None, None)
                    sg = events[1] if len(events) > 1 else (None, None)
                    return pd.Series({"_FG_min": fg[0], "_FG_team": fg[1], "_SG_min": sg[0], "_SG_team": sg[1]})

                _work = pd.concat([_work, _work.apply(_first_second, axis=1)], axis=1)

                _mask = pd.Series([True]*len(_work))
                _mask &= _work["_FG_min"].notna()
                _mask &= (_work["_FG_min"] >= fg_min) & (_work["_FG_min"] <= fg_max)
                if first_team != "Qualsiasi":
                    _mask &= (_work["_FG_team"] == first_team)

                if enable_second:
                    _mask &= _work["_SG_min"].notna()
                    _mask &= (_work["_SG_min"] >= sg_min) & (_work["_SG_min"] <= sg_max)
                    if second_team != "Qualsiasi":
                        _mask &= (_work["_SG_team"] == second_team)

                def _score_at(row, m):
                    h = sum(1 for x in row["_home_times"] if x <= m)
                    a = sum(1 for x in row["_away_times"] if x <= m)
                    return f"{h}-{a}"

                if score_now != "Qualsiasi":
                    _mask &= (_work.apply(lambda r: _score_at(r, minute_now), axis=1) == score_now)

                df_pat = _work[_mask].copy()

                if df_pat.empty:
                    st.info("Nessuna partita corrisponde ai pattern selezionati.")
                else:
                    goal_h_col = "Gol_Home_FT" if "Gol_Home_FT" in df_pat.columns else None
                    goal_a_col = "Gol_Away_FT" if "Gol_Away_FT" in df_pat.columns else None

                    if goal_h_col and goal_a_col:
                        df_pat[goal_h_col] = pd.to_numeric(df_pat[goal_h_col], errors="coerce")
                        df_pat[goal_a_col] = pd.to_numeric(df_pat[goal_a_col], errors="coerce")
                        df_pat = df_pat.dropna(subset=[goal_h_col, goal_a_col])
                        df_pat["Gol_Totali"] = df_pat[goal_h_col] + df_pat[goal_a_col]
                        df_pat["Esito_1X2"] = df_pat.apply(lambda r: ("1" if r[goal_h_col] > r[goal_a_col] else ("2" if r[goal_a_col] > r[goal_h_col] else "X")), axis=1)
                        df_pat["BTTS_SI"] = ((df_pat[goal_h_col] > 0) & (df_pat[goal_a_col] > 0)).astype(int)

                        lines = [0.5,1.5,2.5,3.5,4.5,5.5,6.5]
                        for ln in lines:
                            keyo=f"Over_{str(ln).replace('.','_')}"; keyu=f"Under_{str(ln).replace('.','_')}"
                            df_pat[keyo] = (df_pat["Gol_Totali"] > ln).astype(int)
                            df_pat[keyu] = (df_pat["Gol_Totali"] < ln).astype(int)

                        total = len(df_pat)
                        def pct(n): return round(100.0*n/total, 2) if total>0 else 0.0

                        import pandas as _pd
                        rows = []
                        for es in ["1","X","2"]:
                            c = df_pat["Esito_1X2"].eq(es).sum()
                            rows.append({"Metrica": f"Esito {es}", "Valore": int(c), "Percentuale %": pct(c)})
                        c_btts = int(df_pat["BTTS_SI"].sum())
                        rows.append({"Metrica": "BTTS SI", "Valore": c_btts, "Percentuale %": pct(c_btts)})
                        for ln in lines:
                            keyo=f"Over_{str(ln).replace('.','_')}"; keyu=f"Under_{str(ln).replace('.','_')}"
                            co = int(df_pat[keyo].sum()); cu = int(df_pat[keyu].sum())
                            rows.append({"Metrica": f"Over {ln}", "Valore": co, "Percentuale %": pct(co)})
                            rows.append({"Metrica": f"Under {ln}", "Valore": cu, "Percentuale %": pct(cu)})
                        df_summary = _pd.DataFrame(rows)

                        with st.expander("Riepilogo risultati (esiti finali) per Pattern selezionato"):
                            st.dataframe(df_summary)
                    else:
                        st.info("Colonne gol FT non trovate (richieste: 'Gol_Home_FT' e 'Gol_Away_FT').")

                    with st.expander("Mostra partite filtrate (Pattern selezionato)"):
                        st.dataframe(df_pat)
                        try:
                            csv6 = df_pat.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "Scarica CSV (Sezione 6 - Pattern live)",
                                csv6,
                                file_name="sezione6_pattern_live.csv",
                                mime="text/csv"
                            )
                        except Exception as _e6dl:
                            st.info(f"CSV non disponibile: {_e6dl}")

except Exception as _e6_outer:
    st.warning(f"Errore Analisi Pattern (Sezione 6): {_e6_outer}")
