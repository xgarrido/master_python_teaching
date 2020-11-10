import pandas as pd
data = pd.read_csv("./data/higgs-gg.csv")
print(data.head())

import matplotlib.pyplot as plt
x, y, yerr = data["energy"], data["events"], data["sigma"]
plt.errorbar(x, y, yerr=yerr, fmt=".k", label="ATLAS data")
plt.ylabel(r"Nombre d'événements $H\to\gamma\gamma$")
plt.xlabel(r"$m_{\gamma\gamma}$ [MeV]");

import numpy as np
# "Theoritical model" = 4th order polynomial
def model(x, *parameters):
    y = 0.0
    for i, p in enumerate(parameters):
        y += p*np.power(x, i)
    return y

# Polynom order
n = 4
from scipy.optimize import curve_fit
popt, pcov = curve_fit(model, x, y, sigma=yerr, p0=np.full(n, 1.0))
print(popt)

plt.errorbar(x, y, yerr=yerr, fmt=".k", label="ATLAS data")
xmodel = np.linspace(105, 160, 100)
plt.plot(xmodel, model(xmodel, *popt), "-r", label="modèle")
plt.ylabel(r"Nombre d'événements $H\to\gamma\gamma$")
plt.xlabel(r"$m_{\gamma\gamma}$ [MeV]")
chi2 = np.sum((y - model(x, *popt))**2/yerr**2)
plt.legend(title=r"$\chi2$/ndf = {:.2f}".format(chi2/(len(y)-n)));

grid = plt.GridSpec(4, 1, hspace=0, wspace=0)

main = plt.subplot(grid[0:3], xticklabels=[])
main.errorbar(x, y, yerr=yerr, fmt=".k", label="ATLAS data")
main.set_ylabel(r"Nombre d'événements $H\to\gamma\gamma$")

xmodel = np.linspace(105, 160, 100)
popt, pcov = curve_fit(model, x, y, sigma=yerr, p0=np.full(4, 1.0))
main.plot(xmodel, model(xmodel, *popt), "-r", label="modèle polynomial")

# Plot deviation
sub = plt.subplot(grid[3])
dev = (y - model(x, *popt))/yerr
sub.errorbar(x, dev, fmt=".k")
sub.set_ylabel(r"$\frac{y-y_\mathrm{modèle}}{\sigma}$ [$\sigma$]")
sub.set_xlabel(r"$m_{\gamma\gamma}$ [MeV]")

main.legend()

mask = (dev > 3)
sub.scatter(x[mask], dev[mask], edgecolors="red", c="none", s=100)

print("Masse du boson de Higgs = {} GeV".format(*x[mask].values))
print("Best parameters : {}".format(popt))
