"""
Backtest 7 Filtros v3 — simulación retrospectiva día a día.

Uso:
    pip install yfinance pandas openpyxl
    python backtest_v3.py                      # 2025 completo con watchlist.csv
    python backtest_v3.py --inicio 2025-01-01 --fin 2025-12-31 --watchlist watchlist.csv

Salida: backtest_v3_resultados.xlsx con las hojas Operaciones, Resumen y Senales_descartadas_por_cartera.

Qué automatiza (partes mecánicas de la v3):
  F0  VIX >30 descarta; VIX >25 marca tamaño al 50%; S&P 500 o Nasdaq < -2% en 5 sesiones descarta.
  F1  Resultados trimestrales entre 7 y 14 días naturales (fechas de yfinance).
  F2  Cierre > SMA50 y fuerza relativa positiva (rentabilidad 20 sesiones > S&P 500).
  F3  5D <= +8% y 10D <= +12%. 5D > +4,5% solo se registra.
  F4  Stop en el soporte más cercano (mínimo de 10 sesiones o SMA20, el más alto por debajo del precio).
      Distancia 1,5%-3,3%; si no cumple, se descarta (no se mueve el stop). R/B >= 1,5.
  F5  Volumen negociado medio de 20 sesiones >= 20 M$.
  Cartera: máx. 2 entradas por día, 1 posición abierta por sector, sin reentrada con el mismo catalizador.
  F7  Entrada a la apertura siguiente. Objetivo +4%. Stop. Salida por tiempo al cierre de la
      sesión anterior a resultados. Si un día toca stop y objetivo, se asume stop (conservador).

Qué NO puede automatizar: la nota 0-10 de los verificadores ni el juicio sobre la calidad del
soporte. El backtest mide el esqueleto mecánico del sistema, no el criterio de los verificadores.

Además calcula la variante v2 (sin stop mínimo del 1,5%) para medir el efecto de ese cambio.
"""

import argparse
import sys
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

# ---------------- Parámetros ----------------
P = dict(
    vix_pausa=30.0,
    vix_mitad=25.0,
    caida_indice_5d=-0.02,
    cat_min_dias=7,
    cat_max_dias=14,
    rs_sesiones=20,
    max_5d=0.08,
    max_10d=0.12,
    obs_5d=0.045,
    stop_min=0.015,
    stop_max=0.033,
    rb_min=1.5,
    objetivo=0.04,          # +4%: límite inferior de la banda, conservador como en el backtest v2
    volumen_min_usd=20e6,
    max_entradas_dia=2,
    coste_ida_vuelta=0.005, # comisión + cambio de divisa, en % del importe
)


# ---------------- Descarga de datos ----------------
def descargar(tickers, inicio, fin):
    import yfinance as yf
    ini_hist = (pd.Timestamp(inicio) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    fin_hist = (pd.Timestamp(fin) + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    precios = {}
    import time
    for t in tickers + ["^GSPC", "^IXIC", "^VIX"]:
        df = pd.DataFrame()
        for intento in range(3):
            try:
                df = yf.download(t, start=ini_hist, end=fin_hist, auto_adjust=True, progress=False)
            except Exception as e:
                print(f"  error {t}: {e}")
            if not df.empty:
                break
            time.sleep(3)
        if df.empty:
            print(f"  sin datos: {t}")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        precios[t] = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    resultados = {}
    for t in tickers:
        resultados[t] = []
        for intento in range(3):
            try:
                ed = yf.Ticker(t).get_earnings_dates(limit=40)
                if ed is not None and len(ed):
                    resultados[t] = sorted({pd.Timestamp(d).tz_localize(None).normalize() for d in ed.index})
                    break
            except Exception as e:
                print(f"  sin fechas de resultados: {t} ({e})")
            time.sleep(3)
        print(f"  {t}: {len(resultados[t])} fechas de resultados")
    return precios, resultados


# ---------------- Señales ----------------
def preparar(df):
    d = df.copy()
    d["sma50"] = d["Close"].rolling(50).mean()
    d["sma20"] = d["Close"].rolling(20).mean()
    d["min10"] = d["Low"].rolling(10).min()
    d["r5"] = d["Close"].pct_change(5)
    d["r10"] = d["Close"].pct_change(10)
    d["r20"] = d["Close"].pct_change(P["rs_sesiones"])
    d["vol_usd"] = (d["Close"] * d["Volume"]).rolling(20).mean()
    return d


def proximo_catalizador(fechas, dia):
    for f in fechas:
        dias = (f - dia).days
        if P["cat_min_dias"] <= dias <= P["cat_max_dias"]:
            return f
        if dias > P["cat_max_dias"]:
            break
    return None


def evaluar(t, d, i, idx, fechas_res, stop_min):
    """Devuelve (dict candidato, motivo de descarte o None)."""
    fila = d.iloc[i]
    dia = d.index[i]
    cat = proximo_catalizador(fechas_res, dia)
    if cat is None:
        return None, None  # sin catalizador en ventana: no es candidato
    base = dict(fecha=dia.date(), ticker=t, catalizador=cat.date(), cierre=round(fila.Close, 2),
                r5=fila.r5, r10=fila.r10, obs_5d_alto=bool(fila.r5 > P["obs_5d"]) if pd.notna(fila.r5) else None)

    # F0
    vix = idx["^VIX"].loc[:dia, "Close"].iloc[-1]
    base["vix"] = round(vix, 1)
    if vix > P["vix_pausa"]:
        return base, "F0 VIX>30"
    for ind in ["^GSPC", "^IXIC"]:
        c = idx[ind].loc[:dia, "Close"]
        if len(c) > 5 and c.iloc[-1] / c.iloc[-6] - 1 < P["caida_indice_5d"]:
            return base, f"F0 {ind} < -2% 5 sesiones"
    base["tamano"] = 0.5 if vix > P["vix_mitad"] else 1.0

    # F2
    if pd.isna(fila.sma50) or fila.Close <= fila.sma50:
        return base, "F2 bajo SMA50"
    spx = idx["^GSPC"].loc[:dia, "Close"]
    spx_r = spx.iloc[-1] / spx.iloc[-1 - P["rs_sesiones"]] - 1
    if fila.r20 <= spx_r:
        return base, "F2 RS negativa"

    # F3
    if fila.r5 > P["max_5d"] or fila.r10 > P["max_10d"]:
        return base, "F3 extendida"

    # F5
    if fila.vol_usd < P["volumen_min_usd"]:
        return base, "F5 liquidez"

    # F4: soporte más cercano por debajo del cierre
    soportes = [s for s in (fila.min10, fila.sma20) if pd.notna(s) and s < fila.Close]
    if not soportes:
        return base, "F4 sin soporte"
    stop = max(soportes)
    dist = (fila.Close - stop) / fila.Close
    base.update(stop=round(stop, 2), dist_stop=dist)
    if dist < stop_min:
        return base, "F4 stop < mínimo"
    if dist > P["stop_max"]:
        return base, "F4 stop > 3,3%"
    if 0.05 / dist < P["rb_min"]:
        return base, "F4 R/B < 1,5"
    return base, None


# ---------------- Simulación de salida ----------------
def simular_salida(d, i_senal, stop, cat, stop_min=0.0):
    if i_senal + 1 >= len(d):
        return None
    i_ent = i_senal + 1
    entrada = d.iloc[i_ent].Open
    # si la apertura cae por debajo del stop o deja el stop fuera de rango, no se entra
    if entrada <= stop or (entrada - stop) / entrada < stop_min or (entrada - stop) / entrada > P["stop_max"]:
        return dict(resultado="NO ENTRA (gap)", rent=0.0, salida=None)
    objetivo = entrada * (1 + P["objetivo"])
    ultimo = d.index[d.index < cat]
    if len(ultimo) == 0:
        return None
    dia_limite = ultimo[-1]
    for j in range(i_ent, len(d)):
        f = d.iloc[j]
        dia = d.index[j]
        if j > i_ent and f.Open <= stop:
            return dict(resultado="STOP (gap)", rent=f.Open / entrada - 1, salida=dia.date(), entrada=entrada)
        if f.Low <= stop:
            return dict(resultado="STOP", rent=stop / entrada - 1, salida=dia.date(), entrada=entrada)
        if f.High >= objetivo:
            px = max(f.Open, objetivo) if j > i_ent else objetivo
            return dict(resultado="OBJETIVO", rent=px / entrada - 1, salida=dia.date(), entrada=entrada)
        if dia >= dia_limite:
            return dict(resultado="TIEMPO", rent=f.Close / entrada - 1, salida=dia.date(), entrada=entrada)
    return None


# ---------------- Motor ----------------
def backtest(precios, resultados, sectores, inicio, fin, stop_min, etiqueta):
    idx = {k: precios[k] for k in ["^GSPC", "^IXIC", "^VIX"]}
    datos = {t: preparar(df) for t, df in precios.items() if not t.startswith("^")}
    dias = idx["^GSPC"].loc[inicio:fin].index
    abiertas = []          # (ticker, sector, fecha_salida)
    usados = set()         # (ticker, catalizador)
    ops, descartes, cartera = [], [], []

    for dia in dias:
        abiertas = [a for a in abiertas if a[2] is None or pd.Timestamp(a[2]) >= dia]
        candidatos = []
        for t, d in datos.items():
            if dia not in d.index:
                continue
            i = d.index.get_loc(dia)
            cand, motivo = evaluar(t, d, i, idx, resultados.get(t, []), stop_min)
            if cand is None:
                continue
            if motivo:
                descartes.append({**cand, "motivo": motivo})
            else:
                candidatos.append((t, i, cand))
        # prioridad: menor subida 5D (menos extendida)
        candidatos.sort(key=lambda x: x[2]["r5"])
        entradas_hoy = 0
        for t, i, cand in candidatos:
            sector = sectores.get(t, t)
            clave = (t, cand["catalizador"])
            if clave in usados:
                cartera.append({**cand, "motivo": "reentrada mismo catalizador"}); continue
            if any(a[0] == t for a in abiertas):
                cartera.append({**cand, "motivo": "ya abierta"}); continue
            if any(a[1] == sector for a in abiertas):
                cartera.append({**cand, "motivo": "sector ocupado"}); continue
            if entradas_hoy >= P["max_entradas_dia"]:
                cartera.append({**cand, "motivo": "límite diario"}); continue
            sal = simular_salida(datos[t], i, cand["stop"], pd.Timestamp(cand["catalizador"]), stop_min)
            if sal is None or sal["resultado"].startswith("NO ENTRA"):
                continue
            usados.add(clave)
            entradas_hoy += 1
            abiertas.append((t, sector, sal["salida"]))
            ops.append({**cand, "sector": sector, **sal,
                        "rent_neta": sal["rent"] - P["coste_ida_vuelta"], "variante": etiqueta})
    return pd.DataFrame(ops), pd.DataFrame(descartes), pd.DataFrame(cartera)


def resumen(ops, etiqueta):
    if ops.empty:
        return pd.DataFrame([{"variante": etiqueta, "operaciones": 0}])
    g = ops[ops.rent > 0]
    p = ops[ops.rent <= 0]
    eq = ops.sort_values("fecha").rent_neta.cumsum()
    return pd.DataFrame([{
        "variante": etiqueta,
        "operaciones": len(ops),
        "acierto": len(g) / len(ops),
        "ganancia_media": g.rent.mean() if len(g) else 0,
        "perdida_media": p.rent.mean() if len(p) else 0,
        "esperanza_bruta": ops.rent.mean(),
        "esperanza_neta": ops.rent_neta.mean(),
        "profit_factor": g.rent.sum() / abs(p.rent.sum()) if len(p) and p.rent.sum() else np.nan,
        "max_drawdown_suma": (eq - eq.cummax()).min(),
        "salidas_objetivo": (ops.resultado == "OBJETIVO").sum(),
        "salidas_stop": ops.resultado.str.startswith("STOP").sum(),
        "salidas_tiempo": (ops.resultado == "TIEMPO").sum(),
        "acierto_5d_alto": ops[ops.obs_5d_alto == True].rent.gt(0).mean(),
        "acierto_5d_normal": ops[ops.obs_5d_alto == False].rent.gt(0).mean(),
    }])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inicio", default="2025-01-01")
    ap.add_argument("--fin", default="2025-12-31")
    ap.add_argument("--watchlist", default="watchlist.csv")
    ap.add_argument("--coste", type=float, default=P["coste_ida_vuelta"])
    a = ap.parse_args()
    P["coste_ida_vuelta"] = a.coste

    wl = pd.read_csv(a.watchlist)
    col_t = next((c for c in wl.columns if c.lower() in ("ticker", "symbol", "simbolo", "símbolo")), wl.columns[0])
    col_s = next((c for c in wl.columns if c.lower() in ("sector",)), None)
    tickers = wl[col_t].astype(str).str.strip().tolist()
    sectores = dict(zip(wl[col_t], wl[col_s])) if col_s else {}
    if not col_s:
        print("Aviso: watchlist sin columna 'sector'; la regla de 1 posición por sector no se aplicará.")

    print(f"Descargando {len(tickers)} valores...")
    precios, resultados = descargar(tickers, a.inicio, a.fin)
    ejecutar(precios, resultados, sectores, a.inicio, a.fin, "backtest_v3_resultados.xlsx")


def ejecutar(precios, resultados, sectores, inicio, fin, salida):
    v3, desc3, cart3 = backtest(precios, resultados, sectores, inicio, fin, P["stop_min"], "v3")
    v2, _, _ = backtest(precios, resultados, sectores, inicio, fin, 0.0, "v2 (sin stop mínimo)")
    res = pd.concat([resumen(v3, "v3"), resumen(v2, "v2 (sin stop mínimo)")])
    with pd.ExcelWriter(salida) as xw:
        res.to_excel(xw, sheet_name="Resumen", index=False)
        pd.concat([v3, v2]).to_excel(xw, sheet_name="Operaciones", index=False)
        if not desc3.empty:
            desc3.motivo.value_counts().rename_axis("motivo").reset_index(name="n").to_excel(
                xw, sheet_name="Descartes_por_filtro", index=False)
        cart3.to_excel(xw, sheet_name="Senales_descartadas_por_cartera", index=False)
    print(res.T.to_string())
    print(f"\nGuardado en {salida}")


if __name__ == "__main__":
    main()
