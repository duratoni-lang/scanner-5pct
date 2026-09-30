"""
Más riesgo a la baja y objetivo +5%.

Variantes (entrada a 3-7 días de resultados, apertura del día siguiente a la señal):
  stop: técnico v3 (1,5-3,3%) | fijo 5% | fijo 8%
  objetivo: +3% (referencia) | +5%
  salida: 'antes' = cierre de la sesión previa a resultados
          'atraviesa' = se mantiene durante resultados, cierre máximo 5 sesiones después
  filtros: base | S+M (4 sorpresas seguidas + S&P 500 > SMA50)
Rentabilidad sobre capital: tamaño fijado para perder como máximo el 3% del capital en el stop
(posición = 3% / distancia del stop, máximo 100% del capital).
Diseño = 2023-2024, validación = 2025. Salida: backtest_riesgo_resultados.xlsx
"""
import itertools

import pandas as pd

import backtest_v3 as b
import backtest_pre as bp

INICIO, FIN = bp.INICIO, bp.FIN


def salida(d, i_senal, stop_pct, objetivo, cat, modo, stop_tecnico=None):
    if i_senal + 1 >= len(d):
        return None
    i_ent = i_senal + 1
    entrada = d.iloc[i_ent].Open
    if stop_tecnico is not None:
        stop = stop_tecnico
        dist = (entrada - stop) / entrada
        if dist < b.P["stop_min"] or dist > b.P["stop_max"]:
            return None
    else:
        stop = entrada * (1 - stop_pct)
        dist = stop_pct
    tgt = entrada * (1 + objetivo)
    if modo == "antes":
        lim = d.index[d.index < cat]
        if len(lim) == 0:
            return None
        dia_lim = lim[-1]
    else:
        post = d.index[d.index >= cat]
        if len(post) < 6:
            return None
        dia_lim = post[5]
    for j in range(i_ent, len(d)):
        f, dia = d.iloc[j], d.index[j]
        if j > i_ent and f.Open <= stop:
            return dict(resultado="STOP (gap)", rent=f.Open / entrada - 1, dist=dist, salida=dia.date())
        if j > i_ent and f.Open >= tgt:
            return dict(resultado="OBJETIVO (gap)", rent=f.Open / entrada - 1, dist=dist, salida=dia.date())
        if f.Low <= stop:
            return dict(resultado="STOP", rent=stop / entrada - 1, dist=dist, salida=dia.date())
        if f.High >= tgt:
            return dict(resultado="OBJETIVO", rent=objetivo, dist=dist, salida=dia.date())
        if dia >= dia_lim:
            return dict(resultado="TIEMPO", rent=f.Close / entrada - 1, dist=dist, salida=dia.date())
    return None


def backtest(precios, eventos, sectores, pred, filtros, stop_cfg, objetivo, modo):
    idx = {k: precios[k] for k in ["^GSPC", "^IXIC", "^VIX"]}
    datos = {t: b.preparar(df) for t, df in precios.items() if not t.startswith("^")}
    fechas_res = {t: sorted(ev.fecha) for t, ev in eventos.items()}
    dias = idx["^GSPC"].loc[INICIO:FIN].index
    tecnico = stop_cfg == "tecnico"
    abiertas, usados, ops = [], set(), []
    for dia in dias:
        abiertas = [a for a in abiertas if pd.Timestamp(a[2]) >= dia]
        cands = []
        for t, d in datos.items():
            if dia not in d.index:
                continue
            i = d.index.get_loc(dia)
            cand, motivo = b.evaluar(t, d, i, idx, fechas_res.get(t, []),
                                     b.P["stop_min"] if tecnico else 0.0)
            if cand is None:
                continue
            if motivo and (tecnico or not motivo.startswith("F4")):
                continue
            if not bp.pasa_filtros(filtros, t, dia, pd.Timestamp(cand["catalizador"]), pred, idx["^GSPC"]):
                continue
            cands.append((t, i, cand))
        cands.sort(key=lambda x: x[2]["r5"] if pd.notna(x[2]["r5"]) else 0)
        n = 0
        for t, i, cand in cands:
            sector = sectores.get(t, t)
            clave = (t, cand["catalizador"])
            if clave in usados or any(a[0] == t or a[1] == sector for a in abiertas):
                continue
            if n >= b.P["max_entradas_dia"]:
                break
            s = salida(datos[t], i, None if tecnico else stop_cfg, objetivo,
                       pd.Timestamp(cand["catalizador"]), modo,
                       cand.get("stop") if tecnico else None)
            if s is None:
                continue
            usados.add(clave)
            n += 1
            abiertas.append((t, sector, s["salida"]))
            peso = min(1.0, 0.03 / s["dist"])
            ops.append(dict(fecha=cand["fecha"], ticker=t, catalizador=cand["catalizador"], sector=sector,
                            **s, peso=peso,
                            rent_neta=s["rent"] - b.P["coste_ida_vuelta"],
                            rent_capital=(s["rent"] - b.P["coste_ida_vuelta"]) * peso))
    return pd.DataFrame(ops)


def metricas(o):
    if o.empty:
        return dict(operaciones=0)
    g, p = o[o.rent > 0], o[o.rent <= 0]
    return dict(operaciones=len(o), acierto=len(g) / len(o),
                ganancia_media=g.rent.mean() if len(g) else 0,
                perdida_media=p.rent.mean() if len(p) else 0,
                peor=o.rent.min(),
                esperanza_bruta=o.rent.mean(), esperanza_neta=o.rent_neta.mean(),
                rent_capital_media=o.rent_capital.mean(),
                profit_factor=g.rent.sum() / abs(p.rent.sum()) if p.rent.sum() else None,
                pct_objetivo=o.resultado.str.startswith("OBJETIVO").mean(),
                pct_stop=o.resultado.str.startswith("STOP").mean(),
                pct_gap_stop=(o.resultado == "STOP (gap)").mean())


def main():
    wl = pd.concat([pd.read_csv("watchlist.csv"), pd.read_csv("watchlist_mid.csv")]).drop_duplicates("ticker")
    tickers = wl.ticker.astype(str).str.strip().tolist()
    sectores = dict(zip(wl.ticker, wl.sector))
    b.P.update(cat_min_dias=3, cat_max_dias=7, rb_objetivo_real=False)
    print(f"Descargando {len(tickers)} valores...")
    precios, eventos = bp.descargar(tickers)
    pred = bp.preparar_predictores(precios, eventos, sectores)
    variantes = [("tecnico", 0.03), ("tecnico", 0.05), (0.05, 0.05), (0.08, 0.05)]
    filas, todas = [], []
    for (stop_cfg, obj), modo, filtros in itertools.product(variantes, ["antes", "atraviesa"], ["", "SM"]):
        o = backtest(precios, eventos, sectores, pred, filtros, stop_cfg, obj, modo)
        et = dict(stop="técnico 1,5-3,3%" if stop_cfg == "tecnico" else f"fijo {stop_cfg:.0%}",
                  objetivo=obj, resultados="sale antes" if modo == "antes" else "atraviesa",
                  filtros=filtros or "base")
        if not o.empty:
            o = o.assign(**et, anio=pd.to_datetime(o.fecha).dt.year)
            todas.append(o)
        for per, sub in {"Diseño 2023-24": o[o.anio <= 2024] if not o.empty else o,
                         "Validación 2025": o[o.anio == 2025] if not o.empty else o,
                         "Total": o}.items():
            filas.append({**et, "periodo": per, **metricas(sub)})
        m = metricas(o)
        print(f"  {et}: n={m.get('operaciones')} esp={m.get('esperanza_bruta', 0):+.3%}")
    res = pd.DataFrame(filas)
    with pd.ExcelWriter("backtest_riesgo_resultados.xlsx") as xw:
        res.to_excel(xw, sheet_name="Resumen", index=False)
        if todas:
            pd.concat(todas).to_excel(xw, sheet_name="Operaciones", index=False)
    print(res.to_string())


if __name__ == "__main__":
    main()
