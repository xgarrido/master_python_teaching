import pandas as pd
data = pd.read_csv("data/planets.csv")
print(data.dropna().describe())

by_method = data.groupby("method")
print(by_method.mean())

by_year = data.groupby("year")
by_year.count().number.plot.bar();

by_method_year = data.groupby(["method", "year"])
print(by_method_year.count().number.unstack())

by_method_year.count().number.unstack().T.plot.bar(stacked=True);

subdata = data.set_index("method").loc[["Radial Velocity", "Transit"]]
print(subdata.head())

import numpy as np
colors = np.where(subdata.index == "Transit", "tab:blue", "tab:orange")
subdata.plot.scatter(x="distance", y="orbital_period", c=colors, alpha=0.5, loglog=True);

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
grid = plt.GridSpec(4, 4, hspace=0, wspace=0)
main = plt.subplot(grid[:-1, 1:], xticklabels=[], yticklabels=[],
                   xscale="log", yscale="log")

selected_methods = ["Radial Velocity", "Transit"]
methods = {k: None for k in selected_methods}
for method in selected_methods:
  methods[method] = {"x": subdata.loc[method].distance,
                     "y": subdata.loc[method].orbital_period}

for method, xy in methods.items():
  main.plot(xy["x"], xy["y"], "o", alpha=0.5, label=method)
main.legend(ncol=2, bbox_to_anchor=(0.5, 1.05), loc="center")

xlims = main.get_xlim()
x_hist = plt.subplot(grid[-1, 1:], yticklabels=[],
                     xlim=xlims, xscale="log", xlabel="distance [light years]")
x_hist.invert_yaxis()

ylims = main.get_ylim()
y_hist = plt.subplot(grid[:-1, 0], xticklabels=[],
                     ylim=ylims, yscale="log", ylabel="orbital period [days]")
y_hist.invert_xaxis()

kwargs = dict(alpha=0.5, histtype="stepfilled")
for method, xy in methods.items():
  x, y = xy["x"], xy["y"]
  x_hist.hist(x, orientation="vertical", **kwargs,
              bins=np.logspace(np.log10(xlims[0]), np.log10(xlims[1]), 50))
  y_hist.hist(y, orientation="horizontal", **kwargs,
              bins=np.logspace(np.log10(ylims[0]), np.log10(ylims[1]), 50))
