"""
Comparativa final de estrategias de régimen VIX (2016-2025), como cartera.

Régimen el día S: VIX entre 22 y 35 y S&P 500 sin subir más de 8,4% en 20 sesiones.
Opción 'anti-bajista': además S&P 500 por encima de su SMA200.
Cartera: capital 1 (=100%), posiciones de 25% (12,5% en B), sin solapar más del 100%.
Entrada: apertura de S+1. Costes 0,5% y 0,2% por operación.

  A  VIX + evento: resultados en 5 sesiones, precio < SMA50 y 4 sorpresas positivas seguidas,
     salida la sesión anterior a resultados.
  B  Igual que A pero manteniendo durante resultados (cierre sesión posterior), medio tamaño.
  C  VIX + cualquier evento con resultados en 5 sesiones, prioridad a las más castigadas, salida antes.
  D  VIX sin evento: las acciones líquidas más por debajo de su SMA50, 5 sesiones.
  E  Mantener SPY todo el año (referencia).
Salida: combo_resultados.xlsx
"""
import numpy as np
import pandas as pd

import arbol_decision as ad
from vix_estrategia import regimen, simular_trade

LIQ_MIN = np.log10(20e6)


def cartera(candidatos, precios, dias, etiqueta):
    """candidatos: dict senal -> lista ordenada de (ticker, fecha_salida, peso)."""
    usado, abiertas, ops = 0.0, [], []
    for s in dias:
        if s not in candidatos:
            continue
        abiertas = [a for a in abiertas if a[1] > s]
        usado = sum(a[2] for a in abiertas)
        en_cartera = {a[0] for a in abiertas}
        for t, salida, peso in candidatos[s]:
            if usado + peso > 1.0001:
                break
            if t in en_cartera:
                continue
            r = simular_trade(precios[t], s, salida, None)
            if r is None:
                continue
            ret, dia_sal, _ = r
            abiertas.append((t, dia_sal, peso))
            usado += peso
            en_cartera.add(t)
            ops.append(dict(estrategia=etiqueta, ticker=t, senal=s, salida=dia_sal, peso=peso, ret=ret,
                            anio=pd.Timestamp(s).year))
    return pd.DataFrame(ops)


def main():
    u = ad.universo()
    tickers = u.ticker.tolist()
    sectores = dict(zip(u.ticker, u.sector))
    precios, eventos = ad.descargar(tickers + ["SPY"])
    spy = precios["SPY"]
    idx = precios["^GSPC"].index
    df = ad.construir(precios, {t: e for t, e in eventos.items() if t != "SPY"}, sectores)
    df["salida_antes"] = df.evento.map(lambda f: idx[idx < f][-1] if (idx < f).any() else pd.NaT)
    df["salida_atr"] = df.evento.map(lambda f: idx[idx > f][0] if (idx > f).any() else pd.NaT)

    reg = regimen(precios["^VIX"], precios["^GSPC"])
    spx = precios["^GSPC"].Close
    alcista = (spx > spx.rolling(200).mean()).reindex(reg.index).fillna(False)
    dias = idx[(idx >= pd.Timestamp(ad.INICIO_EVENTOS)) & (idx <= pd.Timestamp(ad.FIN_EVENTOS))]

    # paneles para la estrategia sin evento
    cl = pd.DataFrame({t: d.Close for t, d in precios.items() if not t.startswith("^") and t != "SPY"}).reindex(idx)
    vol = pd.DataFrame({t: d.Volume for t, d in precios.items() if not t.startswith("^") and t != "SPY"}).reindex(idx)
    dist50 = cl / cl.rolling(50).mean() - 1
    liq = np.log10((cl * vol).rolling(20).mean() + 1) >= LIQ_MIN

    resultados, todas = [], []
    for filtro in ["sin filtro", "anti-bajista"]:
        activo = reg & (alcista if filtro == "anti-bajista" else True)
        dias_on = set(activo[activo].index)
        ev = df[df.senal.isin(dias_on) & (df.log_dvol >= LIQ_MIN)].sort_values(["senal", "dist_sma50"])
        sel = ev[(ev.dist_sma50 < 0) & (ev.beats_ult4 == 4)]
        cands = {
            "A VIX+evento, sale antes": {s: [(r.ticker, r.salida_antes, 0.25) for r in g.itertuples()]
                                         for s, g in sel.groupby("senal")},
            "B VIX+evento, atraviesa (medio tamaño)": {s: [(r.ticker, r.salida_atr, 0.125) for r in g.itertuples()]
                                                      for s, g in sel.groupby("senal")},
            "C VIX+cualquier evento, sale antes": {s: [(r.ticker, r.salida_antes, 0.25) for r in g.itertuples()]
                                                   for s, g in ev.groupby("senal")},
        }
        d_c = {}
        for s in sorted(dias_on):
            if s not in dist50.index:
                continue
            p = idx.get_loc(s)
            if p + 5 >= len(idx):
                continue
            fila = dist50.loc[s][liq.loc[s].fillna(False)].dropna().sort_values()
            d_c[s] = [(t, idx[p + 5], 0.25) for t in fila.index[:12]]
        cands["D VIX sin evento (más castigadas)"] = d_c

        for nombre, c in cands.items():
            o = cartera(c, precios, dias, nombre)
            if o.empty:
                continue
            o["filtro"] = filtro
            todas.append(o)
            for coste in [0.005, 0.002]:
                o2 = o.assign(aporte=(o.ret - coste) * o.peso)
                anual = o2.groupby("anio").aporte.sum().reindex(range(2016, 2026), fill_value=0.0)
                eq = o2.sort_values("salida").aporte.cumsum()
                resultados.append(dict(
                    estrategia=nombre, filtro=filtro, coste=coste, operaciones=len(o2),
                    ops_por_anio=len(o2) / 10, acierto=(o2.ret - coste > 0).mean(),
                    media_bruta=o2.ret.mean(), peor_operacion=o2.ret.min(),
                    rent_anual_media=anual.mean(), rent_2016_22=anual.loc[2016:2022].mean(),
                    rent_2023_25=anual.loc[2023:2025].mean(), anios_negativos=(anual < 0).sum(),
                    peor_anio=anual.min(), max_drawdown=(eq - eq.cummax()).min(),
                    **{f"a{y}": anual.loc[y] for y in range(2016, 2026)}))

    anual_spy = spy.Close.resample("YE").last().pct_change()
    anual_spy.index = anual_spy.index.year
    anual_spy = anual_spy.reindex(range(2016, 2026))
    eq_spy = spy.Close.loc["2016":"2025"]
    resultados.append(dict(
        estrategia="E Mantener SPY", filtro="-", coste=0.0, operaciones=1, ops_por_anio=0.1,
        rent_anual_media=anual_spy.mean(), rent_2016_22=anual_spy.loc[2016:2022].mean(),
        rent_2023_25=anual_spy.loc[2023:2025].mean(), anios_negativos=(anual_spy < 0).sum(),
        peor_anio=anual_spy.min(), max_drawdown=(eq_spy / eq_spy.cummax() - 1).min(),
        **{f"a{y}": anual_spy.loc[y] for y in range(2016, 2026)}))
    res = pd.DataFrame(resultados)
    with pd.ExcelWriter("combo_resultados.xlsx") as xw:
        res.to_excel(xw, sheet_name="Comparativa", index=False)
        pd.concat(todas).to_excel(xw, sheet_name="Operaciones", index=False)
    print(res.round(4).to_string())


if __name__ == "__main__":
    main()
