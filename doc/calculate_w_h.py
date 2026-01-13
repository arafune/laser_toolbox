import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import sympy as sp

    # from sympy import symbols, solve, simplify, tan, cos, sin, trigsimp, expand_trig
    from sympy import (
        symbols,
        solve,
        simplify,
        tan,
        cos,
        sin,
        trigsimp,
        expand_trig,
    )
    return cos, sin, solve, symbols, tan


@app.cell
def _(symbols):
    alpha, theta3, l1, l2, W, H, L = symbols(
        "alpha theta3 l1 l2 W H L", real=True, positive=True
    )
    return H, L, W, alpha, l1, l2, theta3


@app.cell
def _(H, L, W, alpha, cos, l1, l2, sin, tan, theta3):
    eq1 = tan(alpha / 2 - theta3) - (
        (-H + l2 * cos(alpha / 2) + l1 * cos(alpha / 2))
        / (W - l2 * sin(alpha / 2) + l1 * sin(alpha / 2))
    )
    eq2 = L - (W * cos(alpha / 2) + H * sin(alpha / 2))
    return eq1, eq2


@app.cell
def _(H, W, eq1, eq2, solve):
    sols = solve([eq1, eq2], [W, H])
    return (sols,)


@app.cell
def _(sols):
    sols
    return


@app.cell
def _():
    # sp.latex(clean_sols)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
