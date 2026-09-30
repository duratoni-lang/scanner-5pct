"""
Escáner diario v4 — VIX + evento. Registro en papel.

Cada día laborable tras el cierre de EE. UU.:
  1. Régimen: VIX entre 22 y 35 y S&P 500 sin subir más de 8,4% en 20 sesiones.
     Se anota en regimen_log.csv. Si no está activo, no busca candidatos.
  2. Si está activo, candidatos del S&P 500 + S&P 400:
       - precio < SMA50, volumen negociado medio 20 sesiones >= 20 M$
       - resultados dentro de 5 a 7 sesiones
       - batió la estimación de BPA en los 4 últimos trimestres
     Orden: más por debajo de su SMA50. Máximo 8 posiciones abiertas de 12,5%.
  3. Registro en papel (registro_v4.csv): entrada a la apertura siguiente a la señal,
     salida al cierre de la primera sesión posterior a resultados. Sin stop.
  4. Aviso: escribe aviso.md si hay novedades (régimen que se activa, señales o cierres).
Uso: python escaner_v4.py [--forzar]   (--forzar ignora el régimen, solo para pruebas)
"""
import argparse
import io
import os
import time
from datetime import timedelta

import numpy as np
import pandas as pd

VIX_MIN, VIX_MAX, SPX20_MAX = 22.0, 35.0, 0.084
LIQ_MIN_USD = 20e6
PESO, MAX_POS = 0.125, 8
CAPITAL_REF = 6000
REGISTRO, LOG_REG, AVISO = "registro_v4.csv", "regimen_log.csv", "aviso.md"
COLS = ["fecha_senal", "ticker", "sector", "evento", "dist_sma50", "sorpresas_4", "cierre_senal",
        "fecha_entrada", "entrada", "fecha_salida", "salida", "rentabilidad", "estado", "nota"]


def yf_hist(tickers, periodo="9mo"):
    import yfinance as yf
    out = {}
    for k in range(0, len(tickers), 100):
        lote = tickers[k:k + 100]
        df = pd.DataFrame()
        for _ in range(3):
            try:
                df = yf.download(lote, period=periodo, auto_adjust=True, progress=False,
                                 group_by="ticker", threads=True)
                break
            except Exception as e:
                print(f"  error lote: {e}")
                time.sleep(10)
        for t in lote:
            try:
                sub = df[t] if len(lote) > 1 else df
                if isinstance(sub.columns, pd.MultiIndex):
                    sub.columns = sub.columns.get_level_values(-1)
                sub = sub[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(sub):
                    out[t] = sub
            except Exception:
                pass
    return out


def universo():
    import requests
    filas = []
    for url in ["https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"]:
        try:
            html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
            for t in pd.read_html(io.StringIO(html)):
                cols = {c.lower(): c for c in t.columns.astype(str)}
                sym = next((cols[c] for c in cols if c in ("symbol", "ticker symbol", "ticker")), None)
                sec = next((cols[c] for c in cols if "sector" in c), None)
                if sym and sec and len(t) > 100:
                    filas.append(pd.DataFrame({"ticker": t[sym].astype(str).str.replace(".", "-", regex=False),
                                               "sector": t[sec]}))
                    break
        except Exception as e:
            print(f"  universo: {e}")
    if not filas:
        wl = pd.concat([pd.read_csv(f) for f in ("watchlist.csv", "watchlist_mid.csv") if os.path.exists(f)])
        return wl.drop_duplicates("ticker")
    return pd.concat(filas).drop_duplicates("ticker")


def sorpresas(t):
    import yfinance as yf
    for _ in range(3):
        try:
            ev = yf.Ticker(t).get_earnings_dates(limit=12)
            if ev is None or ev.empty:
                return None
            ev = ev.copy()
            ev.index = pd.DatetimeIndex([pd.Timestamp(d).tz_localize(None).normalize() for d in ev.index])
            col = next((c for c in ev.columns if "Surprise" in c), None)
            return pd.DataFrame({"fecha": ev.index, "sorpresa": ev[col].values if col else np.nan}
                                ).drop_duplicates("fecha").sort_values("fecha")
        except Exception:
            time.sleep(2)
    return None


def cargar(ruta, cols):
    if os.path.exists(ruta):
        return pd.read_csv(ruta, dtype=str)
    return pd.DataFrame(columns=cols)


def actualizar_registro(reg, hoy, precios_fn):
    """Rellena entradas y salidas pendientes con precios reales."""
    avisos = []
    abiertas = reg[reg.estado.isin(["pendiente entrada", "abierta"])]
    if abiertas.empty:
        return reg, avisos
    precios = precios_fn(sorted(abiertas.ticker.unique()))
    for i, r in abiertas.iterrows():
        d = precios.get(r.ticker)
        if d is None or d.empty:
            continue
        senal = pd.Timestamp(r.fecha_senal)
        if r.estado == "pendiente entrada":
            post = d[d.index > senal]
            if len(post):
                reg.loc[i, ["fecha_entrada", "entrada", "estado"]] = [str(post.index[0].date()),
                                                                      f"{post.Open.iloc[0]:.4f}", "abierta"]
                r = reg.loc[i]
        if reg.loc[i, "estado"] == "abierta":
            evento = pd.Timestamp(r.evento)
            post_ev = d[d.index > evento]
            if len(post_ev):
                salida = post_ev.Close.iloc[0]
                ret = salida / float(reg.loc[i, "entrada"]) - 1
                reg.loc[i, ["fecha_salida", "salida", "rentabilidad", "estado"]] = [
                    str(post_ev.index[0].date()), f"{salida:.4f}", f"{ret:.4f}", "cerrada"]
                avisos.append(f"- Cerrada **{r.ticker}**: {ret:+.2%} (entrada {float(reg.loc[i, 'entrada']):.2f}, "
                              f"salida {salida:.2f})")
    return reg, avisos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forzar", action="store_true")
    a = ap.parse_args()

    idx = yf_hist(["^VIX", "^GSPC"], "3mo")
    vix, spx = idx["^VIX"].Close, idx["^GSPC"].Close
    hoy = spx.index[-1]
    vix_hoy = float(vix.loc[:hoy].iloc[-1])
    spx_r20 = float(spx.iloc[-1] / spx.iloc[-21] - 1)
    activo = VIX_MIN < vix_hoy <= VIX_MAX and spx_r20 <= SPX20_MAX
    print(f"{hoy.date()} VIX={vix_hoy:.1f} S&P500 20 sesiones={spx_r20:+.1%} régimen={'ACTIVO' if activo else 'inactivo'}")

    log = cargar(LOG_REG, ["fecha", "vix", "spx_r20", "activo"])
    previo = log.activo.iloc[-1] == "True" if len(log) else False
    if str(hoy.date()) not in set(log.fecha):
        log = pd.concat([log, pd.DataFrame([dict(fecha=str(hoy.date()), vix=f"{vix_hoy:.2f}",
                                                 spx_r20=f"{spx_r20:.4f}", activo=str(activo))])])
    log.to_csv(LOG_REG, index=False)

    reg = cargar(REGISTRO, COLS)
    reg, avisos = actualizar_registro(reg, hoy, lambda ts: yf_hist(ts, "3mo"))
    if activo and not previo:
        avisos.insert(0, f"## Régimen VIX ACTIVADO\nVIX {vix_hoy:.1f}, S&P 500 {spx_r20:+.1%} en 20 sesiones.")
    if not activo and previo:
        avisos.insert(0, f"## Régimen VIX desactivado\nVIX {vix_hoy:.1f}. No se abren nuevas posiciones.")

    nuevas = []
    if activo or a.forzar:
        u = universo()
        sectores = dict(zip(u.ticker, u.sector))
        precios = yf_hist(u.ticker.tolist(), "9mo")
        filas = []
        for t, d in precios.items():
            if len(d) < 60 or d.index[-1] != hoy:
                continue
            c = d.Close
            dist = c.iloc[-1] / c.rolling(50).mean().iloc[-1] - 1
            liq = (c * d.Volume).rolling(20).mean().iloc[-1]
            if dist < 0 and liq >= LIQ_MIN_USD:
                filas.append((t, dist, float(c.iloc[-1])))
        print(f"  castigadas y líquidas: {len(filas)}")
        sesiones = pd.bdate_range(hoy + timedelta(days=1), periods=10)
        ventana = (sesiones[4], sesiones[6])  # resultados entre 5 y 7 sesiones
        abiertas_ahora = reg[reg.estado.isin(["pendiente entrada", "abierta"])]
        ya = set(zip(reg.ticker, reg.evento))
        for t, dist, cierre in sorted(filas, key=lambda x: x[1]):
            ev = sorpresas(t)
            time.sleep(0.3)
            if ev is None:
                continue
            fut = ev[(ev.fecha >= ventana[0]) & (ev.fecha <= ventana[1])]
            if fut.empty:
                continue
            pas = ev[ev.fecha < hoy].dropna(subset=["sorpresa"]).tail(4)
            if len(pas) < 4 or not (pas.sorpresa > 0).all():
                continue
            evento = str(fut.fecha.iloc[0].date())
            if (t, evento) in ya or t in set(abiertas_ahora.ticker):
                continue
            hueco = len(abiertas_ahora) + len(nuevas) < MAX_POS
            nuevas.append(dict(fecha_senal=str(hoy.date()), ticker=t, sector=sectores.get(t, ""), evento=evento,
                               dist_sma50=f"{dist:.4f}", sorpresas_4="4/4", cierre_senal=f"{cierre:.2f}",
                               estado="pendiente entrada" if hueco else "sin hueco",
                               nota="FORZADO (prueba)" if a.forzar and not activo else ""))
        if nuevas:
            reg = pd.concat([reg, pd.DataFrame(nuevas)], ignore_index=True)
            lineas = [f"- **{n['ticker']}** ({n['sector']}): resultados {n['evento']}, "
                      f"{float(n['dist_sma50']):+.1%} bajo SMA50, cierre {n['cierre_senal']} — {n['estado']}"
                      + (f" · posición de referencia {PESO * CAPITAL_REF:.0f} €" if n["estado"] == "pendiente entrada" else "")
                      for n in nuevas]
            avisos.append("## Señales de hoy\n" + "\n".join(lineas) +
                          "\n\nEntrada a la apertura de mañana; salida al cierre de la sesión siguiente a resultados.")
    reg[COLS].to_csv(REGISTRO, index=False)

    cerr = reg[reg.estado == "cerrada"]
    if len(cerr):
        r = cerr.rentabilidad.astype(float)
        avisos.append(f"\nAcumulado en papel: {len(r)} cerradas, acierto {(r > 0).mean():.0%}, "
                      f"media {r.mean():+.2%} por operación (antes de costes).")
    if avisos and (activo or previo or a.forzar or any("Cerrada" in x for x in avisos)):
        with open(AVISO, "w") as f:
            f.write(f"# Escáner v4 — {hoy.date()}\n\nVIX {vix_hoy:.1f} · régimen "
                    f"{'ACTIVO' if activo else 'inactivo'}\n\n" + "\n".join(avisos) + "\n")
        print(open(AVISO).read())


if __name__ == "__main__":
    main()
