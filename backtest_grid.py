"""
Rejilla de variantes del backtest 7 Filtros v3 (2023-2025).

Prueba, para cada universo (grandes / medianas):
  - ventana de entrada: 7-14 días antes de resultados (actual) y 3-7 días
  - objetivo: +4% (actual), +3%, +2,5%
  - R/B: calculado contra +5% (regla actual) o contra el objetivo real
Diseño = 2023-2024, validación = 2025.
Salida: backtest_grid_resultados.xlsx
"""
import itertools
import pandas as pd
import backtest_v3 as b

INICIO, FIN = "2023-01-01", "2025-12-31"
UNIVERSOS = {"Grandes": "watchlist.csv", "Medianas": "watchlist_mid.csv"}
VENTANAS = {"7-14 d": (7, 14), "3-7 d": (3, 7)}
OBJETIVOS = [0.04, 0.03, 0.025]
RB_REAL = [False, True]
BASE = dict(b.P)


def metricas(ops):
    if ops.empty:
        return dict(operaciones=0)
    g, p = ops[ops.rent > 0], ops[ops.rent <= 0]
    eq = ops.sort_values("fecha").rent_neta.cumsum()
    return dict(
        operaciones=len(ops),
        acierto=len(g) / len(ops),
        esperanza_bruta=ops.rent.mean(),
        esperanza_neta=ops.rent_neta.mean(),
        profit_factor=g.rent.sum() / abs(p.rent.sum()) if p.rent.sum() else None,
        pct_objetivo=(ops.resultado == "OBJETIVO").mean(),
        pct_stop=ops.resultado.str.startswith("STOP").mean(),
        pct_tiempo=(ops.resultado == "TIEMPO").mean(),
        rent_media_tiempo=ops[ops.resultado == "TIEMPO"].rent.mean(),
        drawdown_suma=(eq - eq.cummax()).min(),
    )


def main():
    filas, todas = [], []
    for uni, fichero in UNIVERSOS.items():
        wl = pd.read_csv(fichero)
        tickers = wl.ticker.astype(str).str.strip().tolist()
        sectores = dict(zip(wl.ticker, wl.sector))
        print(f"== {uni}: descargando {len(tickers)} valores")
        precios, resultados = b.descargar(tickers, INICIO, FIN)
        for (vn, (vmin, vmax)), obj, rbr in itertools.product(VENTANAS.items(), OBJETIVOS, RB_REAL):
            if obj == 0.04 and rbr:
                continue  # con +4% ambos cálculos de R/B son casi iguales
            b.P.update(BASE)
            b.P.update(cat_min_dias=vmin, cat_max_dias=vmax, objetivo=obj, rb_objetivo_real=rbr)
            ops, _, _ = b.backtest(precios, resultados, sectores, INICIO, FIN, b.P["stop_min"], "grid")
            etiqueta = dict(universo=uni, ventana=vn, objetivo=obj,
                            rb="objetivo real" if rbr else "contra +5%")
            if not ops.empty:
                ops = ops.assign(**etiqueta, anio=pd.to_datetime(ops.fecha).dt.year)
                todas.append(ops)
            periodos = {"Diseño 2023-24": ops[ops.anio <= 2024] if not ops.empty else ops,
                        "Validación 2025": ops[ops.anio == 2025] if not ops.empty else ops}
            if not ops.empty:
                for a in (2023, 2024, 2025):
                    periodos[str(a)] = ops[ops.anio == a]
            for per, sub in periodos.items():
                filas.append({**etiqueta, "periodo": per, **metricas(sub)})
            m = metricas(ops)
            print(f"  {vn} obj {obj:.1%} rb {etiqueta['rb']}: n={m.get('operaciones')} "
                  f"esp_bruta={m.get('esperanza_bruta', 0):+.3%}")
    res = pd.DataFrame(filas)
    with pd.ExcelWriter("backtest_grid_resultados.xlsx") as xw:
        res.to_excel(xw, sheet_name="Resumen", index=False)
        if todas:
            pd.concat(todas).to_excel(xw, sheet_name="Operaciones", index=False)
    print(res[res.periodo.isin(["Diseño 2023-24", "Validación 2025"])].to_string())


if __name__ == "__main__":
    main()
