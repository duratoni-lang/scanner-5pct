"""
Árboles de decisión sobre eventos de resultados (S&P 500 + S&P 400, 2016-2025).

Para cada publicación de resultados:
  día de señal S = 5 sesiones antes de la última sesión previa al evento
  entrada = apertura de S+1
  objetivo 'antes'     = cierre de la última sesión previa al evento / entrada - 1
  objetivo 'atraviesa' = cierre de la primera sesión posterior al evento / entrada - 1
Variables: solo con información disponible al cierre de S.
Entrenamiento 2016-2022, prueba 2023-2025.
Salidas: arbol_resultados.xlsx, arbol_antes.png, arbol_atraviesa.png, eventos_dataset.csv.gz
"""
import io
import time

import numpy as np
import pandas as pd

INICIO_EVENTOS, FIN_EVENTOS = "2016-01-01", "2025-12-31"
CORTE = "2023-01-01"
SESIONES_ANTES = 5
COSTE = 0.005


# ---------------- Universo ----------------
def universo():
    import requests
    fuentes = {
        "S&P 500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "S&P 400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    }
    filas = []
    for nombre, url in fuentes.items():
        try:
            html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
            for t in pd.read_html(io.StringIO(html)):
                cols = {c.lower(): c for c in t.columns.astype(str)}
                sym = next((cols[c] for c in cols if c in ("symbol", "ticker symbol", "ticker")), None)
                sec = next((cols[c] for c in cols if "sector" in c), None)
                if sym and sec and len(t) > 100:
                    filas.append(pd.DataFrame({"ticker": t[sym].astype(str).str.replace(".", "-", regex=False),
                                               "sector": t[sec], "indice": nombre}))
                    break
        except Exception as e:
            print(f"  no se pudo leer {nombre}: {e}")
    if not filas:
        print("  usando watchlists locales")
        wl = pd.concat([pd.read_csv("watchlist.csv"), pd.read_csv("watchlist_mid.csv")])
        return wl.assign(indice="local").drop_duplicates("ticker")
    u = pd.concat(filas).drop_duplicates("ticker")
    print(f"  universo: {len(u)} valores")
    return u


# ---------------- Descarga ----------------
def descargar(tickers):
    import yfinance as yf
    precios = {}
    todos = tickers + ["^GSPC", "^VIX"]
    for k in range(0, len(todos), 100):
        lote = todos[k:k + 100]
        for _ in range(3):
            try:
                df = yf.download(lote, start="2015-01-01", end="2026-02-15", auto_adjust=True,
                                 progress=False, group_by="ticker", threads=True)
                break
            except Exception as e:
                print(f"  error lote {k}: {e}")
                time.sleep(10)
        for t in lote:
            try:
                sub = df[t][["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(sub) > 300:
                    precios[t] = sub
            except Exception:
                pass
        print(f"  precios {min(k + 100, len(todos))}/{len(todos)}")
    eventos = {}
    for n, t in enumerate(tickers):
        if t not in precios:
            continue
        for _ in range(3):
            try:
                ev = yf.Ticker(t).get_earnings_dates(limit=48)
                if ev is not None and len(ev):
                    ev = ev.copy()
                    ev.index = pd.DatetimeIndex([pd.Timestamp(d).tz_localize(None).normalize() for d in ev.index])
                    col = next((c for c in ev.columns if "Surprise" in c), None)
                    eventos[t] = pd.DataFrame({"fecha": ev.index,
                                               "sorpresa": ev[col].values if col else np.nan}
                                              ).drop_duplicates("fecha").sort_values("fecha").reset_index(drop=True)
                    break
            except Exception:
                time.sleep(2)
        time.sleep(0.3)
        if n % 100 == 0:
            print(f"  resultados {n}/{len(tickers)}")
    return precios, eventos


# ---------------- Dataset ----------------
def construir(precios, eventos, sectores):
    spx, vix = precios["^GSPC"], precios["^VIX"]
    spx_c = spx.Close
    spx_sma50, spx_sma200 = spx_c.rolling(50).mean(), spx_c.rolling(200).mean()
    filas = []
    for t, ev in eventos.items():
        d = precios[t]
        c, h, l, v, o = d.Close, d.High, d.Low, d.Volume, d.Open
        sma20, sma50, sma200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean() / c
        vol20 = c.pct_change().rolling(20).std()
        dvol = (c * v)
        idx = d.index
        # pasadas: subida previa y reacción de cada evento (para historial)
        hist = []
        for _, e in ev.iterrows():
            antes = idx[idx < e.fecha]
            desp = idx[idx > e.fecha]
            if len(antes) < SESIONES_ANTES + 2 or len(desp) == 0:
                hist.append((e.fecha, e.sorpresa, np.nan, np.nan, None))
                continue
            p = idx.get_loc(antes[-1])
            pre = c.iloc[p] / c.iloc[p - SESIONES_ANTES] - 1
            reac = c.loc[desp[0]] / c.iloc[p] - 1
            hist.append((e.fecha, e.sorpresa, pre, reac, desp[0]))
        hist = pd.DataFrame(hist, columns=["fecha", "sorpresa", "pre", "reac", "conocido"])
        for _, e in hist.iterrows():
            if not (INICIO_EVENTOS <= str(e.fecha.date()) <= FIN_EVENTOS) or e.conocido is None:
                continue
            antes = idx[idx < e.fecha]
            p_last = idx.get_loc(antes[-1])
            s = p_last - SESIONES_ANTES
            if s < 260 or s + 1 > p_last:
                continue
            S = idx[s]
            ent = o.iloc[s + 1]
            if not ent or np.isnan(ent):
                continue
            pasado = hist[(hist.fecha < e.fecha) & hist.conocido.notna() & (hist.conocido <= S)]
            ult4 = pasado.dropna(subset=["sorpresa"]).tail(4)
            ult8 = pasado.tail(8)
            spx_s = spx_c.loc[:S]
            vix_s = vix.Close.loc[:S]
            if len(spx_s) < 200 or len(vix_s) == 0:
                continue
            rs = lambda n: (c.iloc[s] / c.iloc[s - n] - 1) - (spx_s.iloc[-1] / spx_s.iloc[-1 - n] - 1)
            filas.append(dict(
                ticker=t, sector=sectores.get(t, "Otro"), evento=e.fecha, senal=S,
                anio=e.fecha.year, mes=e.fecha.month,
                r5=c.iloc[s] / c.iloc[s - 5] - 1, r10=c.iloc[s] / c.iloc[s - 10] - 1,
                r20=c.iloc[s] / c.iloc[s - 20] - 1, r60=c.iloc[s] / c.iloc[s - 60] - 1,
                r250=c.iloc[s] / c.iloc[s - 250] - 1,
                rs20=rs(20), rs60=rs(60),
                dist_sma20=c.iloc[s] / sma20.iloc[s] - 1, dist_sma50=c.iloc[s] / sma50.iloc[s] - 1,
                dist_sma200=c.iloc[s] / sma200.iloc[s] - 1,
                dist_max52=c.iloc[s] / h.iloc[s - 250:s + 1].max() - 1,
                atr14=atr.iloc[s], vol20=vol20.iloc[s],
                vol_rel=v.iloc[s - 4:s + 1].mean() / v.iloc[s - 49:s + 1].mean(),
                log_dvol=np.log10(dvol.iloc[s - 19:s + 1].mean() + 1),
                beats_ult4=(ult4.sorpresa > 0).sum() if len(ult4) == 4 else np.nan,
                sorpresa_ult=ult4.sorpresa.iloc[-1] if len(ult4) else np.nan,
                sorpresa_media4=ult4.sorpresa.clip(-100, 100).mean() if len(ult4) else np.nan,
                pre_hist8=ult8.pre.mean() if len(ult8) >= 4 else np.nan,
                reac_hist8=ult8.reac.mean() if len(ult8) >= 4 else np.nan,
                reac_ult=pasado.reac.iloc[-1] if len(pasado) else np.nan,
                spx_r5=spx_s.iloc[-1] / spx_s.iloc[-6] - 1, spx_r20=spx_s.iloc[-1] / spx_s.iloc[-21] - 1,
                spx_sobre_sma50=float(spx_s.iloc[-1] > spx_sma50.loc[:S].iloc[-1]),
                spx_sobre_sma200=float(spx_s.iloc[-1] > spx_sma200.loc[:S].iloc[-1]),
                vix=vix_s.iloc[-1],
                ret_antes=c.iloc[p_last] / ent - 1,
                ret_atraviesa=c.loc[e.conocido] / ent - 1,
            ))
    return pd.DataFrame(filas)


VARIABLES = ["r5", "r10", "r20", "r60", "r250", "rs20", "rs60", "dist_sma20", "dist_sma50", "dist_sma200",
             "dist_max52", "atr14", "vol20", "vol_rel", "log_dvol", "beats_ult4", "sorpresa_ult",
             "sorpresa_media4", "pre_hist8", "reac_hist8", "reac_ult", "spx_r5", "spx_r20",
             "spx_sobre_sma50", "spx_sobre_sma200", "vix", "mes"]


# ---------------- Modelos ----------------
def analizar(df, objetivo, nombre_png):
    from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score

    d = df.dropna(subset=[objetivo]).copy()
    d[objetivo] = d[objetivo].clip(-0.3, 0.3)
    X = pd.get_dummies(d[VARIABLES + ["sector"]], columns=["sector"], dtype=float)
    X = X.fillna(X.median(numeric_only=True))
    y = d[objetivo]
    tr = d.evento < CORTE
    te = ~tr
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

    arbol = DecisionTreeRegressor(max_depth=4, min_samples_leaf=max(200, int(len(Xtr) * 0.01)), random_state=0)
    arbol.fit(Xtr, ytr)
    reglas = export_text(arbol, feature_names=list(X.columns), decimals=3)
    hoja_tr, hoja_te = arbol.apply(Xtr), arbol.apply(Xte)
    hojas = []
    for h in np.unique(hoja_tr):
        a, b_ = ytr[hoja_tr == h], yte[hoja_te == h]
        hojas.append(dict(hoja=int(h), n_entreno=len(a), media_entreno=a.mean(), acierto_entreno=(a > COSTE).mean(),
                          n_prueba=len(b_), media_prueba=b_.mean() if len(b_) else np.nan,
                          acierto_prueba=(b_ > COSTE).mean() if len(b_) else np.nan,
                          neta_prueba=b_.mean() - COSTE if len(b_) else np.nan))
    hojas = pd.DataFrame(hojas).sort_values("media_entreno", ascending=False)
    # camino legible de cada hoja
    caminos = describir_hojas(arbol, list(X.columns))
    hojas["reglas"] = hojas.hoja.map(caminos)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(26, 11))
    plot_tree(arbol, feature_names=list(X.columns), filled=True, impurity=False, precision=3, fontsize=8)
    plt.title(f"Árbol de decisión — rentabilidad '{objetivo}' (entrenado 2016-2022)")
    plt.savefig(nombre_png, dpi=110, bbox_inches="tight")
    plt.close(fig)

    ycls_tr, ycls_te = (ytr > COSTE).astype(int), (yte > COSTE).astype(int)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=100, max_features="sqrt",
                                n_jobs=-1, random_state=0)
    rf.fit(Xtr, ycls_tr)
    prob = rf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(ycls_te, prob)
    pi = permutation_importance(rf, Xte, ycls_te, scoring="roc_auc", n_repeats=5, random_state=0, n_jobs=-1)
    imp = pd.DataFrame({"variable": X.columns, "importancia_auc": pi.importances_mean,
                        "desviacion": pi.importances_std}).sort_values("importancia_auc", ascending=False)
    # rentabilidad por quintil de probabilidad en prueba
    q = pd.qcut(prob, 5, labels=["Q1 (peor)", "Q2", "Q3", "Q4", "Q5 (mejor)"])
    quint = pd.DataFrame({"quintil": q, "ret": yte.values}).groupby("quintil", observed=True).ret.agg(
        n="count", media="mean", acierto=lambda s: (s > COSTE).mean()).reset_index()
    quint["neta"] = quint.media - COSTE
    resumen = dict(objetivo=objetivo, eventos_entreno=len(Xtr), eventos_prueba=len(Xte),
                   media_entreno=ytr.mean(), media_prueba=yte.mean(),
                   acierto_base_prueba=(yte > COSTE).mean(), auc_bosque_prueba=auc)
    return resumen, reglas, hojas, imp, quint


def describir_hojas(arbol, nombres):
    t = arbol.tree_
    caminos = {}

    def rec(nodo, cond):
        if t.children_left[nodo] == -1:
            caminos[nodo] = " Y ".join(cond) if cond else "(todas)"
            return
        f, u = nombres[t.feature[nodo]], t.threshold[nodo]
        rec(t.children_left[nodo], cond + [f"{f} <= {u:.3f}"])
        rec(t.children_right[nodo], cond + [f"{f} > {u:.3f}"])
    rec(0, [])
    return caminos


def main():
    u = universo()
    tickers = u.ticker.tolist()
    sectores = dict(zip(u.ticker, u.sector))
    print("Descargando datos...")
    precios, eventos = descargar(tickers)
    print(f"  con precios: {len(precios)}, con resultados: {len(eventos)}")
    df = construir(precios, eventos, sectores)
    print(f"  eventos en el dataset: {len(df)}")
    df.to_csv("eventos_dataset.csv.gz", index=False, compression="gzip")
    salidas = {}
    for obj in ["ret_antes", "ret_atraviesa"]:
        salidas[obj] = analizar(df, obj, f"arbol_{obj.split('_')[1]}.png")
    with pd.ExcelWriter("arbol_resultados.xlsx") as xw:
        pd.DataFrame([s[0] for s in salidas.values()]).to_excel(xw, sheet_name="Resumen", index=False)
        for obj, (res, reglas, hojas, imp, quint) in salidas.items():
            k = obj.split("_")[1]
            hojas.to_excel(xw, sheet_name=f"Hojas_{k}", index=False)
            imp.to_excel(xw, sheet_name=f"Importancia_{k}", index=False)
            quint.to_excel(xw, sheet_name=f"Quintiles_{k}", index=False)
            pd.DataFrame({"reglas": reglas.splitlines()}).to_excel(xw, sheet_name=f"Arbol_{k}", index=False)
        df.groupby("anio")[["ret_antes", "ret_atraviesa"]].agg(["count", "mean"]).to_excel(xw, sheet_name="Por_anio")
    for obj, (res, reglas, hojas, imp, quint) in salidas.items():
        print("\n==", obj, res)
        print(imp.head(12).to_string(index=False))
        print(quint.to_string(index=False))
        print(hojas.drop(columns="reglas").to_string(index=False))


if __name__ == "__main__":
    main()
