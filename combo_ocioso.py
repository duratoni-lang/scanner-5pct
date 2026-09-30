"""
¿Qué hacer con el capital parado? Estrategia B/A + resto en SPY o en monetario.

Usa las operaciones ya simuladas (combo_resultados.xlsx, hoja Operaciones, filtro 'sin filtro').
Cada día: fracción invertida en la estrategia = suma de pesos de las posiciones abiertas.
El resultado de cada operación (en euros) se reparte linealmente entre sus sesiones.
Resto del capital: 0% (liquidez), monetario (letras del Tesoro EE. UU. a 3 meses, ^IRX) o SPY.
Costes: 0,5% por operación de la estrategia; 0,05% por cada euro que entra o sale de SPY.
Salida: ocioso_resultados.xlsx
"""
import numpy as np
import pandas as pd
import yfinance as yf

COSTE_OP, COSTE_SPY = 0.005, 0.0005


def descarga(t):
    df = yf.download(t, start="2015-06-01", end="2026-01-05", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def main():
    spy = descarga("SPY").Close
    irx = descarga("^IRX").Close.reindex(spy.index).ffill() / 100
    dias = spy.loc["2016-01-01":"2025-12-31"].index
    r_spy = spy.pct_change().reindex(dias).fillna(0)
    r_mon = (irx.reindex(dias).fillna(0) / 252)

    ops = pd.read_excel("combo_resultados.xlsx", sheet_name="Operaciones", parse_dates=["senal", "salida"])
    ops = ops[ops.filtro == "sin filtro"]
    filas = []
    for estrategia in ["A VIX+evento, sale antes", "B VIX+evento, atraviesa (medio tamaño)"]:
        o = ops[ops.estrategia == estrategia]
        peso = pd.Series(0.0, index=dias)
        aporte = pd.Series(0.0, index=dias)
        for _, t in o.iterrows():
            tramo = dias[(dias > t.senal) & (dias <= t.salida)]
            if len(tramo) == 0:
                continue
            diario = (t.ret - COSTE_OP) / len(tramo)  # reparto lineal: el total en euros de la operación se conserva
            peso.loc[tramo] += t.peso
            aporte.loc[tramo] += t.peso * diario
        peso = peso.clip(upper=1.0)
        libre = 1 - peso
        for resto, r_resto in [("liquidez 0%", pd.Series(0.0, index=dias)),
                               ("monetario", r_mon), ("SPY", r_spy)]:
            coste_rot = peso.diff().abs().fillna(0) * COSTE_SPY if resto == "SPY" else 0.0
            r = aporte + libre.shift(1).fillna(1) * r_resto - coste_rot
            filas.append((f"{estrategia[:1]} + {resto}", r))
    filas.append(("SPY mantener", r_spy))
    filas.append(("Monetario", r_mon))

    res, anual_tbl = [], {}
    for nombre, r in filas:
        eq = (1 + r).cumprod()
        anual = (1 + r).groupby(r.index.year).prod() - 1
        anual_tbl[nombre] = anual
        cagr = eq.iloc[-1] ** (252 / len(r)) - 1
        dd = (eq / eq.cummax() - 1).min()
        vol = r.std() * np.sqrt(252)
        res.append(dict(estrategia=nombre, rent_anual_compuesta=cagr, max_caida=dd, volatilidad=vol,
                        anios_negativos=int((anual < 0).sum()), peor_anio=anual.min(), mejor_anio=anual.max(),
                        capital_final_6000=6000 * eq.iloc[-1]))
    res = pd.DataFrame(res)
    anual_tbl = pd.DataFrame(anual_tbl)
    with pd.ExcelWriter("ocioso_resultados.xlsx") as xw:
        res.to_excel(xw, sheet_name="Resumen", index=False)
        anual_tbl.to_excel(xw, sheet_name="Por_anio")
    print(res.round(4).to_string())
    print(anual_tbl.round(3).to_string())


if __name__ == "__main__":
    main()
