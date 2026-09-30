"""
Anticipar el evento: ¿qué señales previas a resultados predicen la subida?

Base: reglas v3 con entrada a 3-7 días del evento y objetivo +3%.
Filtros predictivos (solo con información disponible el día de la señal):
  H  Historial propio: en los últimos 8 resultados, la subida de las 5 sesiones previas
     fue positiva en media y en al menos el 60% de los casos (mínimo 4 eventos).
  S  Racha de sorpresas: batió la estimación de BPA en los 4 últimos trimestres publicados.
  L  Lectura sectorial: los valores del mismo sector que publicaron en los 15 días previos
     reaccionaron al alza de media (cierre día siguiente vs. cierre día anterior).
  M  Mercado: S&P 500 por encima de su SMA50.
Diseño = 2023-2024, validación = 2025.
Salida: backtest_pre_resultados.xlsx
"""
import itertools
import time

import numpy as np
import pandas as pd

import backtest_v3 as b

INICIO, FIN = "2023-01-01", "2025-12-31"
HIST_INICIO = "2020-09-01"   # historia suficiente para 8 trimestres antes de 2023
COMBOS = ["", "H", "S", "L", "M", "HS", "HM", "SM", "LM", "HSM", "HSLM"]


def descargar(tickers):
    import yfinance as yf
    precios, eventos = {}, {}
    for t in tickers + ["^GSPC", "^IXIC", "^VIX"]:
        df = pd.DataFrame()
        for _ in range(3):
            try:
                df = yf.download(t, start=HIST_INICIO, end="2026-02-15", auto_adjust=True, progress=False)
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
    for t in tickers:
        ev = pd.DataFrame()
        for _ in range(3):
            try:
                ev = yf.Ticker(t).get_earnings_dates(limit=40)
                if ev is not None and len(ev):
                    break
            except Exception as e:
                print(f"  sin resultados {t}: {e}")
            time.sleep(3)
        if ev is None or ev.empty:
            eventos[t] = pd.DataFrame(columns=["fecha", "sorpresa"])
            continue
        ev = ev.copy()
        ev.index = pd.DatetimeIndex([pd.Timestamp(d).tz_localize(None).normalize() for d in ev.index])
        col = next((c for c in ev.columns if "Surprise" in c), None)
        eventos[t] = pd.DataFrame({"fecha": ev.index,
                                   "sorpresa": ev[col].values if col else np.nan}
                                  ).drop_duplicates("fecha").sort_values("fecha").reset_index(drop=True)
        print(f"  {t}: {len(eventos[t])} eventos")
    return precios, eventos


def reaccion(df, fecha):
    """Rentabilidad cierre del día hábil siguiente al evento vs. cierre del día anterior."""
    antes = df.index[df.index < fecha]
    despues = df.index[df.index > fecha]
    if len(antes) == 0 or len(despues) == 0:
        return np.nan, None
    return df.loc[despues[0], "Close"] / df.loc[antes[-1], "Close"] - 1, despues[0]


def subida_previa(df, fecha, sesiones=5):
    antes = df.index[df.index < fecha]
    if len(antes) <= sesiones:
        return np.nan
    return df.loc[antes[-1], "Close"] / df.loc[antes[-1 - sesiones], "Close"] - 1


def preparar_predictores(precios, eventos, sectores):
    """Precalcula por evento: subida previa, reacción y día en que la reacción es conocida."""
    filas = []
    for t, ev in eventos.items():
        if t not in precios:
            continue
        for _, e in ev.iterrows():
            r, dia_conocido = reaccion(precios[t], e.fecha)
            filas.append(dict(ticker=t, sector=sectores.get(t), fecha=e.fecha, sorpresa=e.sorpresa,
                              subida_previa=subida_previa(precios[t], e.fecha),
                              reaccion=r, conocido=dia_conocido))
    return pd.DataFrame(filas)


def pasa_filtros(combo, t, dia, cat, pred, spx):
    hist = pred[(pred.ticker == t) & (pred.fecha < cat) & pred.conocido.notna() & (pred.conocido <= dia)]
    if "H" in combo:
        h = hist.dropna(subset=["subida_previa"]).tail(8)
        if len(h) < 4 or h.subida_previa.mean() <= 0 or (h.subida_previa > 0).mean() < 0.6:
            return False
    if "S" in combo:
        s = hist.dropna(subset=["sorpresa"]).tail(4)
        if len(s) < 4 or not (s.sorpresa > 0).all():
            return False
    if "L" in combo:
        sector = pred.loc[pred.ticker == t, "sector"].iloc[0] if (pred.ticker == t).any() else None
        pares = pred[(pred.sector == sector) & (pred.ticker != t) & pred.conocido.notna()
                     & (pred.conocido <= dia) & (pred.conocido > dia - pd.Timedelta(days=15))]
        if len(pares) == 0 or pares.reaccion.mean() <= 0:
            return False
    if "M" in combo:
        c = spx.loc[:dia, "Close"]
        if len(c) < 50 or c.iloc[-1] <= c.iloc[-50:].mean():
            return False
    return True


def backtest(precios, eventos, sectores, pred, combo):
    idx = {k: precios[k] for k in ["^GSPC", "^IXIC", "^VIX"]}
    datos = {t: b.preparar(df) for t, df in precios.items() if not t.startswith("^")}
    fechas_res = {t: sorted(ev.fecha) for t, ev in eventos.items()}
    dias = idx["^GSPC"].loc[INICIO:FIN].index
    abiertas, usados, ops = [], set(), []
    for dia in dias:
        abiertas = [a for a in abiertas if a[2] is None or pd.Timestamp(a[2]) >= dia]
        candidatos = []
        for t, d in datos.items():
            if dia not in d.index:
                continue
            i = d.index.get_loc(dia)
            cand, motivo = b.evaluar(t, d, i, idx, fechas_res.get(t, []), b.P["stop_min"])
            if cand is None or motivo:
                continue
            if not pasa_filtros(combo, t, dia, pd.Timestamp(cand["catalizador"]), pred, idx["^GSPC"]):
                continue
            candidatos.append((t, i, cand))
        candidatos.sort(key=lambda x: x[2]["r5"])
        entradas = 0
        for t, i, cand in candidatos:
            sector = sectores.get(t, t)
            clave = (t, cand["catalizador"])
            if clave in usados or any(a[0] == t or a[1] == sector for a in abiertas):
                continue
            if entradas >= b.P["max_entradas_dia"]:
                break
            sal = b.simular_salida(datos[t], i, cand["stop"], pd.Timestamp(cand["catalizador"]), b.P["stop_min"])
            if sal is None or sal["resultado"].startswith("NO ENTRA"):
                continue
            usados.add(clave)
            entradas += 1
            abiertas.append((t, sector, sal["salida"]))
            ops.append({**cand, "sector": sector, **sal, "rent_neta": sal["rent"] - b.P["coste_ida_vuelta"]})
    return pd.DataFrame(ops)


def metricas(ops):
    if ops.empty:
        return dict(operaciones=0)
    g, p = ops[ops.rent > 0], ops[ops.rent <= 0]
    return dict(operaciones=len(ops), acierto=len(g) / len(ops),
                esperanza_bruta=ops.rent.mean(), esperanza_neta=ops.rent_neta.mean(),
                profit_factor=g.rent.sum() / abs(p.rent.sum()) if p.rent.sum() else None,
                pct_objetivo=(ops.resultado == "OBJETIVO").mean(),
                pct_stop=ops.resultado.str.startswith("STOP").mean())


def main():
    wl = pd.concat([pd.read_csv("watchlist.csv").assign(universo="Grandes"),
                    pd.read_csv("watchlist_mid.csv").assign(universo="Medianas")]).drop_duplicates("ticker")
    tickers = wl.ticker.astype(str).str.strip().tolist()
    sectores = dict(zip(wl.ticker, wl.sector))
    universo = dict(zip(wl.ticker, wl.universo))
    b.P.update(cat_min_dias=3, cat_max_dias=7, objetivo=0.03, rb_objetivo_real=False)
    print(f"Descargando {len(tickers)} valores...")
    precios, eventos = descargar(tickers)
    pred = preparar_predictores(precios, eventos, sectores)
    filas, todas = [], []
    for combo in COMBOS:
        ops = backtest(precios, eventos, sectores, pred, combo)
        nombre = combo or "Base (sin predictores)"
        if not ops.empty:
            ops = ops.assign(filtros=nombre, anio=pd.to_datetime(ops.fecha).dt.year,
                             universo=ops.ticker.map(universo))
            todas.append(ops)
        partes = {"Diseño 2023-24": ops[ops.anio <= 2024] if not ops.empty else ops,
                  "Validación 2025": ops[ops.anio == 2025] if not ops.empty else ops,
                  "Total": ops}
        for per, sub in partes.items():
            filas.append({"filtros": nombre, "periodo": per, **metricas(sub)})
        m = metricas(ops)
        print(f"  {nombre}: n={m.get('operaciones')} esp_bruta={m.get('esperanza_bruta', 0):+.3%}")
    res = pd.DataFrame(filas)
    # capacidad predictiva cruda de cada variable sobre todos los eventos (sin reglas de entrada)
    pred = pred.sort_values(["ticker", "fecha"])
    pred["hist_media_previa"] = pred.groupby("ticker").subida_previa.transform(
        lambda s: s.shift(1).rolling(8, min_periods=4).mean())
    pred["racha_beats"] = pred.groupby("ticker").sorpresa.transform(
        lambda s: (s.shift(1) > 0).rolling(4, min_periods=4).sum())
    ev = pred[(pred.fecha >= INICIO) & (pred.fecha <= FIN)].dropna(subset=["subida_previa"])
    diag = pd.DataFrame([
        dict(grupo="Todos los eventos", n=len(ev), subida_previa_media=ev.subida_previa.mean(),
             pct_positivos=(ev.subida_previa > 0).mean()),
        dict(grupo="Historial previo positivo", n=(ev.hist_media_previa > 0).sum(),
             subida_previa_media=ev[ev.hist_media_previa > 0].subida_previa.mean(),
             pct_positivos=(ev[ev.hist_media_previa > 0].subida_previa > 0).mean()),
        dict(grupo="Historial previo negativo", n=(ev.hist_media_previa <= 0).sum(),
             subida_previa_media=ev[ev.hist_media_previa <= 0].subida_previa.mean(),
             pct_positivos=(ev[ev.hist_media_previa <= 0].subida_previa > 0).mean()),
        dict(grupo="4 beats seguidos", n=(ev.racha_beats == 4).sum(),
             subida_previa_media=ev[ev.racha_beats == 4].subida_previa.mean(),
             pct_positivos=(ev[ev.racha_beats == 4].subida_previa > 0).mean()),
        dict(grupo="Menos de 4 beats", n=(ev.racha_beats < 4).sum(),
             subida_previa_media=ev[ev.racha_beats < 4].subida_previa.mean(),
             pct_positivos=(ev[ev.racha_beats < 4].subida_previa > 0).mean()),
    ])
    with pd.ExcelWriter("backtest_pre_resultados.xlsx") as xw:
        res.to_excel(xw, sheet_name="Resumen", index=False)
        diag.to_excel(xw, sheet_name="Poder_predictivo", index=False)
        if todas:
            pd.concat(todas).to_excel(xw, sheet_name="Operaciones", index=False)
    print(res.to_string())
    print(diag.to_string())


if __name__ == "__main__":
    main()
