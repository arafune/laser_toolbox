import marimo

__generated_with = "0.19.0"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""
    # Demonstration of PrismPair class to calculate GDD
    """)
    return


@app.cell
def _():
    import marimo as mo

    import numpy as np
    import matplotlib.pyplot as plt

    from laser_toolbox.prism_pair import (
        PrismPair,
        brewster_angle_deg,
        ideal_apex_deg,
    )
    return PrismPair, mo, np, plt


app._unparsable_cell(
    r"""
    material = "SF11"
    wavelength_nm = 800
    brewster_angle_deg_ = brewster_angle_deg(wavelength_nm, material=material)
    ideal_apex_deg_ = ideal_apex_deg(wavelength_nm, material)
    mo.md(
        f"In the ideal situation, the apex angle for 800 nm light is **{ideal_apex_deg_}** degrees, because the Brewster angle of SF11 is **{brewster_angle_deg_}takn1224
    
        ** degrees.  Here we set the apex angle to be **59** degrees as  our prism, and incident angle is set **60** degrees."
    )
    """,
    name="_"
)


@app.cell
def _(mo):
    l1 = mo.ui.slider(0, 20, value=5, step=0.1, label="Prism 1 Insert (mm)")
    l2 = mo.ui.slider(0, 20, value=5, step=0.1, label="Prism 2 Insert (mm)")
    l = mo.ui.slider(0, 800, value=300, step=1, label="length L (mm)")
    l1, l2, l
    return l, l1, l2


@app.cell
def _(PrismPair, l, l1, l2):
    pp_variable_800 = PrismPair(
        incident_angle_deg=60,
        apex_deg=59,
        material="SF11",
        prism_insert_mm=(l1.value, l2.value),
        separation_mm=l.value,
        wavelength_nm=800,
    )
    return (pp_variable_800,)


@app.cell
def _(PrismPair, l, l1, l2, np, plt, pp_variable_800):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    wavelengths = np.linspace(700, 900, 500)
    pp_variable = PrismPair(
        incident_angle_deg=60,
        material="SF11",
        prism_insert_mm=(l1.value, l2.value),
        separation_mm=l.value,
        wavelength_nm=wavelengths,
        apex_deg=59,
    )
    proerty_str = f"GDD @ 800 nm: {pp_variable_800.gdd:.2f} fs$^2$\n (insert :({l1.value}, {l2.value}) mm, separation: {l.value} mm)"
    ax.plot(wavelengths, pp_variable.gdd)
    ax.set_xlabel("Wavelength  ( nm )")
    ax.set_ylabel("GDD  ( fs$^2$ )")
    ax.set_title(proerty_str)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
