"""
Estrategia de régimen VIX: ¿es algo propio de los resultados o un rebote general?

Señal de régimen el día S: VIX entre 22 y 35 y S&P 500 sin subir más de 8,4% en 20 sesiones.
Comparaciones (2016-2025, prueba separada 2023-2025):
  1. Eventos de resultados: rentabilidad y exceso sobre SPY en la misma ventana.
  2. Acciones SIN resultados en los 20 días siguientes, mismos días de señal y misma ventana (5 sesiones).
  3. Cartera: máx. 4 posiciones, 25% cada una, entrada apertura S+1, salida antes / atraviesa,
     sin stop o con stop 8%; costes 0,2% y 0,5%. Prioridad: más por debajo de su SMA50.
  4. Alternativa ETF: comprar SPY en la apertura siguiente a la señal y mantener 5 sesiones.
Salida: vix_resultados.xlsx
"""
import numpy as np
import pandas as pd

import arbol_decision as ad

VIX_MIN, VIX_MAX, SPX20_MAX = 22.0, 35.0, 0.084
MAX_POS, PESO = 4, 0.25
LIQ_MIN = np.log10(20e6)


def regimen(vix, spx):
    c = spx.Close
    r20 = c / c.shift(20) - 1
    v = vix.Close.reindex(c.index).ffill()
    return ((v > VIX_MIN) & (v <= VIX_MAX) & (r20 <= SPX20_MAX)).rename("regimen")


def panel(precios, campo):
    return pd.DataFrame({t: df[campo] for t, df in precios.items() if not t.startswith("^")})


def ret_spy(spy, senal, fin_fecha):
    idx = spy.index
    ent = idx[idx > senal]
    if len(ent) == 0:
        return np.nan
    salida = idx[idx <= fin_fecha]
    if len(salida) == 0 or salida[-1] < ent[0]:
        return np.nan
    return spy.Close.loc[salida[-1]] / spy.Open.loc[ent[0]] - 1


def simular_trade(d, senal, salida_fecha, stop):
    idx = d.index
    ent_dias = idx[idx > senal]
    if len(ent_dias) == 0:
        return None
    e0 = ent_dias[0]
    if pd.isna(salida_fecha) or salida_fecha < e0:
        return None
    entrada = d.Open.loc[e0]
    tramo = d.loc[e0:salida_fecha]
    if len(tramo) == 0 or not entrada:
        return None
    if stop:
        nivel = entrada * (1 - stop)
        for k, (dia, f) in enumerate(tramo.iterrows()):
            if k > 0 and f.Open <= nivel:
                return f.Open / entrada - 1, dia, "stop gap"
            if f.Low <= nivel:
                return nivel / entrada - 1, dia, "stop"
    return tramo.Close.iloc[-1] / entrada - 1, tramo.index[-1], "salida"


def main():
    u = ad.universo()
    tickers = u.ticker.tolist()
    sectores = dict(zip(u.ticker, u.sector))
    precios, eventos = ad.descargar(tickers + ["SPY"])
    spy = precios["SPY"]
    df = ad.construir(precios, {t: e for t, e in eventos.items() if t != "SPY"}, sectores)
    reg = regimen(precios["^VIX"], precios["^GSPC"])
    df["regimen"] = df.senal.map(reg).fillna(False).astype(bool)
    df["periodo"] = np.where(df.evento < ad.CORTE, "2016-22", "2023-25")

    # fechas de salida
    idx_all = precios["^GSPC"].index
    df["salida_antes"] = df.evento.map(lambda f: idx_all[idx_all < f][-1] if (idx_all < f).any() else pd.NaT)
    df["salida_atr"] = df.evento.map(lambda f: idx_all[idx_all > f][0] if (idx_all > f).any() else pd.NaT)

    # 1. exceso sobre SPY
    df["spy_antes"] = [ret_spy(spy, s, f) for s, f in zip(df.senal, df.salida_antes)]
    df["spy_atr"] = [ret_spy(spy, s, f) for s, f in zip(df.senal, df.salida_atr)]
    df["exc_antes"] = df.ret_antes - df.spy_antes
    df["exc_atr"] = df.ret_atraviesa - df.spy_atr
    df["semana"] = df.senal.dt.to_period("W").astype(str)

    def resumen(g):
        s = g.groupby("semana")
        return pd.Series(dict(eventos=len(g), semanas=g.semana.nunique(),
                              ret_antes=s.ret_antes.mean().mean(), spy_antes=s.spy_antes.mean().mean(),
                              exceso_antes=s.exc_antes.mean().mean(),
                              ret_atr=s.ret_atraviesa.mean().mean(), spy_atr=s.spy_atr.mean().mean(),
                              exceso_atr=s.exc_atr.mean().mean(),
                              pct_semanas_exceso_pos_antes=(s.exc_antes.mean() > 0).mean()))
    t1 = df.groupby(["regimen", "periodo"]).apply(resumen).reset_index()

    # 2. acciones sin resultados, mismos días de señal en régimen, ventana 5 sesiones
    op, cl = panel(precios, "Open"), panel(precios, "Close")
    op, cl = op.reindex(idx_all), cl.reindex(idx_all)
    fechas_ev = {t: np.array(e.fecha.values, dtype="datetime64[ns]") for t, e in eventos.items()}
    filas = []
    dias_senal = sorted(df.loc[df.regimen, "senal"].unique())
    for s in dias_senal:
        p = idx_all.get_loc(s)
        if p + 5 >= len(idx_all):
            continue
        ent, sal = idx_all[p + 1], idx_all[p + 5]
        r = cl.loc[sal] / op.loc[ent] - 1
        lim = np.datetime64(s + pd.Timedelta(days=20))
        sin_res = [t for t in r.index if t in fechas_ev and t != "SPY" and not (
            ((fechas_ev[t] > np.datetime64(s - pd.Timedelta(days=3))) & (fechas_ev[t] <= lim)).any())]
        rr = r[sin_res].dropna()
        ev_dia = df[df.regimen & (df.senal == s)]
        filas.append(dict(senal=s, semana=str(pd.Timestamp(s).to_period("W")),
                          periodo="2016-22" if s < pd.Timestamp(ad.CORTE) else "2023-25",
                          sin_resultados=rr.mean(), n_sin=len(rr),
                          con_resultados=ev_dia.ret_antes.mean(), n_con=len(ev_dia),
                          spy=spy.Close.loc[sal] / spy.Open.loc[ent] - 1))
    t2d = pd.DataFrame(filas)
    t2 = t2d.groupby("periodo")[["con_resultados", "sin_resultados", "spy"]].mean().reset_index()
    t2["dias"] = t2d.groupby("periodo").size().values

    # 3. cartera
    cand = df[df.regimen & (df.log_dvol >= LIQ_MIN)].copy()
    cand = cand.sort_values(["senal", "dist_sma50"])
    carteras, ops_all = [], []
    for modo, col_sal in [("antes", "salida_antes"), ("atraviesa", "salida_atr")]:
        for stop in [None, 0.08]:
            abiertas, ops = [], []
            for s, grupo in cand.groupby("senal"):
                abiertas = [a for a in abiertas if a > s]
                for _, e in grupo.iterrows():
                    if len(abiertas) >= MAX_POS:
                        break
                    r = simular_trade(precios[e.ticker], s, e[col_sal], stop)
                    if r is None:
                        continue
                    ret, dia_sal, motivo = r
                    abiertas.append(dia_sal)
                    ops.append(dict(modo=modo, stop="8%" if stop else "sin stop", ticker=e.ticker,
                                    senal=s, salida=dia_sal, motivo=motivo, ret=ret,
                                    anio=pd.Timestamp(s).year, periodo=e.periodo))
            o = pd.DataFrame(ops)
            if o.empty:
                continue
            ops_all.append(o)
            for coste in [0.002, 0.005]:
                o2 = o.assign(neta=o.ret - coste)
                o2["aporte"] = o2.neta * PESO
                anual = o2.groupby("anio").aporte.sum()
                eq = o2.sort_values("salida").aporte.cumsum()
                for per, sub in [("2016-22", o2[o2.periodo == "2016-22"]), ("2023-25", o2[o2.periodo == "2023-25"]),
                                 ("Total", o2)]:
                    carteras.append(dict(modo=modo, stop="8%" if stop else "sin stop", coste=coste, periodo=per,
                                         operaciones=len(sub), acierto=(sub.neta > 0).mean(),
                                         media_bruta=sub.ret.mean(), media_neta=sub.neta.mean(),
                                         peor=sub.ret.min(),
                                         rent_capital_total=sub.aporte.sum(),
                                         anios=sub.anio.nunique(),
                                         rent_capital_anual_media=sub.aporte.sum() / {"2016-22": 7, "2023-25": 3, "Total": 10}[per]))
                carteras[-1]["max_drawdown"] = (eq - eq.cummax()).min()
    t3 = pd.DataFrame(carteras)
    anual_tbl = pd.concat(ops_all).assign(aporte_05=lambda x: (x.ret - 0.005) * PESO).pivot_table(
        index="anio", columns=["modo", "stop"], values="aporte_05", aggfunc="sum").fillna(0)

    # 4. ETF: comprar SPY al día siguiente de la señal, 5 sesiones, sin solapar
    etf, libre = [], None
    for s in reg[reg].index:
        if s < pd.Timestamp(ad.INICIO_EVENTOS) or (libre is not None and s < libre):
            continue
        p = spy.index.get_loc(s) if s in spy.index else None
        if p is None or p + 5 >= len(spy):
            continue
        ent, sal = spy.index[p + 1], spy.index[p + 5]
        etf.append(dict(senal=s, anio=s.year, ret=spy.Close.loc[sal] / spy.Open.loc[ent] - 1))
        libre = sal
    etf = pd.DataFrame(etf)
    etf["neta"] = etf.ret - 0.001
    t4 = etf.groupby("anio").agg(operaciones=("ret", "size"), media=("ret", "mean"),
                                 acierto=("neta", lambda s: (s > 0).mean()),
                                 rent_capital=("neta", lambda s: (1 + s).prod() - 1)).reset_index()
    bh = spy.Close.resample("YE").last().pct_change().rename("spy_comprar_y_mantener")
    bh.index = bh.index.year
    t4 = t4.merge(bh, left_on="anio", right_index=True, how="left")

    with pd.ExcelWriter("vix_resultados.xlsx") as xw:
        t1.to_excel(xw, sheet_name="1_Exceso_vs_SPY", index=False)
        t2.to_excel(xw, sheet_name="2_Sin_resultados", index=False)
        t3.to_excel(xw, sheet_name="3_Cartera", index=False)
        anual_tbl.to_excel(xw, sheet_name="3_Cartera_por_anio")
        t4.to_excel(xw, sheet_name="4_ETF_SPY", index=False)
        pd.concat(ops_all).to_excel(xw, sheet_name="Operaciones", index=False)
    for nombre, t in [("1", t1), ("2", t2), ("3", t3), ("3b", anual_tbl), ("4", t4)]:
        print(f"\n== {nombre}\n{t.round(4).to_string()}")


if __name__ == "__main__":
    main()
