import numpy as np
import pandas as pd
df = pd.read_csv("./data/iris.csv")

sepal_length = df["sepal.length"]
sepal_width = df["sepal.width"]
petal_length = df["petal.length"]
petal_width = df["petal.width"]
variety = df["variety"]
df["colors"] = np.where(variety == "Setosa", "C0",
                        np.where(variety == "Versicolor", "C1", "C2"))
# Version pure python
# variety2int = {k: i for i, k in enumerate(variety.unique())}
# colors = [variety2int[k] for k in variety]

# Distributions des longueurs
import matplotlib.pyplot as plt

kwargs = dict(histtype="stepfilled", alpha=0.5, bins=20)

species = variety.unique()
labels = {"longueur des sépales [cm]" : sepal_length,
          "largeur des sépales [cm]"  : sepal_width,
          "longueur des pétales [cm]" : petal_length,
          "largeur des pétales [cm]"  : petal_width}

for xlabel, data in labels.items():
    # Determine best range and bin probability
    r = (np.min(data), np.max(data))
    plt.figure()
    for s in species:
        plt.hist(data[variety == s], **kwargs, label=s, range=r)
    plt.xlabel(xlabel)
    plt.legend()

# Diagrammes longueur vs. largeur sépales
plt.figure()
plt.scatter(sepal_length, sepal_width, s=100*petal_width,
            c=df["colors"], alpha=0.2)
plt.xlabel("longueur des sépales [cm]")
plt.ylabel("largeur des sépales [cm]")

# Création d'une légende à partir d'un scatter plot vide
for i, v in enumerate(variety.unique()):
    plt.scatter([], [], c="C{}".format(i), alpha=0.2, label=v)
plt.legend()

# Changement de taille de police uniquement pour cette figure
with plt.rc_context({"font.size": 5}):
      # Définition d'une grille de sous-figures
      fig, ax = plt.subplots(len(labels), len(labels),
                             sharex="col", sharey="row",
                             figsize=(1.5*len(labels), 1.5*len(labels)))

      for l1, d1 in labels.items():
            i1 = list(labels.keys()).index(l1)
            for l2, d2 in labels.items():
                  i2 = list(labels.keys()).index(l2)
                  for v in variety.unique():
                        sc = (variety == v)
                        if l1 == l2:
                              ax[i1, i2].hist(d1[sc], alpha=0.5, bins=10, density=True)
                        else:
                              ax[i1, i2].scatter(d2[sc], d1[sc], s=5, alpha=0.5)
                              ax[-1, i1].set_xlabel(l1)
                              ax[i1, 0].set_ylabel(l1)

      # # Création d'une légende à partir d'un scatter plot vide
      # for key, name in iris.items():
      #       plt.scatter([], [], label=name)
      #       plt.legend(title="iris", bbox_to_anchor=(1, len(iris)/2+1), loc="upper left")
      #       fig.subplots_adjust(right=0.9)
plt.show()
