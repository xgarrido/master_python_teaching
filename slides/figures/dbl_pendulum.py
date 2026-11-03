"""Reproduction de la figure du double pendule (dbl_pendulum.pdf).

Double pendule composé : deux barres rouges de longueur l, de centres de
masse (x1, y1) et (x2, y2), avec leurs poids mg, les angles theta1/theta2
mesurés par rapport à la verticale, les cotes l, et quelques positions
« fantômes » de la seconde barre.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Polygon

# plt.rcParams.update(
#     {
#         "font.family": "serif",
#         "mathtext.fontset": "stix",
#         "font.serif": ["STIXGeneral", "DejaVu Serif"],
#     }
# )

# ------------------------------------------------------------------ paramètres
l = 1.0  # longueur d'une barre
w = 0.085  # largeur d'une barre
th1, th2 = np.deg2rad(20), np.deg2rad(45)

# positions « fantômes » (theta1, theta2) en degrés, de la plus pâle à la plus foncée
ghosts = [(-16, -38, 0.10), (-7, -17, 0.20), (2, 3, 0.28), (11, 23, 0.34)]

color = "#4c72b0"
LW = 2.6


def unit(theta):
    """Vecteur unitaire le long d'une barre (y vers le haut, theta / verticale)."""
    return np.array([np.sin(theta), -np.cos(theta)])


def normal(theta):
    """Normale côté gauche/bas, utilisée pour les cotes."""
    return np.array([-np.cos(theta), -np.sin(theta)])


def rod(ax, a, theta, fc=color, ec="k", alpha=1.0, lw=LW, z=2):
    """Barre rectangulaire de longueur l partant du point a."""
    u, n = unit(theta), normal(theta)
    b = a + l * u
    pts = [a + n * w / 2, b + n * w / 2, b - n * w / 2, a - n * w / 2]
    ax.add_patch(
        Polygon(pts, closed=True, fc=fc, ec=ec, lw=lw, alpha=alpha, joinstyle="miter", zorder=z)
    )
    return b


def pivot(ax, p, ec="k", alpha=1.0, r=0.024, lw=2.0, z=6):
    ax.add_patch(Circle(p, r, fc="white", ec=ec, lw=lw, alpha=alpha, zorder=z))


def dimension(ax, a, theta, d=0.57, frac=0.355, label_pos=None, lab_off=(0, 0)):
    """Cote de longueur l parallèle à la barre, décalée de d côté gauche."""
    u, n = unit(theta), normal(theta)
    b = a + l * u
    # traits de rappel
    for p in (a, b):
        ax.plot(*zip(p + 0.05 * n, p + (d + 0.04) * n), "k-", lw=1.2, zorder=1)
    # deux flèches vers les extrémités, avec un blanc au milieu pour le "l"
    pa, pb = a + d * n, b + d * n
    for tip, tail in ((pa, pa + frac * l * u), (pb, pb - frac * l * u)):
        ax.add_patch(
            FancyArrowPatch(
                tail,
                tip,
                arrowstyle="-|>",
                mutation_scale=34,
                color="k",
                lw=2.0,
                shrinkA=0,
                shrinkB=0,
                zorder=3,
            )
        )
    mid = (pa + pb) / 2 + np.array(lab_off)
    ax.text(
        *mid,
        r"$\ell$",
        fontsize=26,
        fontweight="bold",
        fontstyle="italic",
        ha="center",
        va="center"
    )


def angle(ax, c, theta, r, ref_len=0.5):
    """Verticale de référence + arc d'angle theta, centrés en c."""
    ax.plot(*zip(c, c + np.array([0, -ref_len])), "k-", lw=1.5, zorder=1)
    ax.add_patch(Arc(c, 2 * r, 2 * r, theta1=-90, theta2=-90 + np.rad2deg(theta), lw=1.8, zorder=1))


# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(figsize=(5.6, 5.2))
ax.set_aspect("equal")
ax.axis("off")
ax.set_xlim(-1.0, 1.26)
ax.set_ylim(-2.07, 0.27)

O = np.array([0.0, 0.0])

# support
ax.plot([-1.0, 1.0], [0, 0], "k-", lw=2.2, zorder=1)

# positions fantômes
for t1, t2, al in ghosts:
    t1, t2 = np.deg2rad(t1), np.deg2rad(t2)
    j = rod(ax, O, t1, fc=color, ec="0.45", alpha=al, lw=2.0, z=1)
    rod(ax, j, t2, fc=color, ec="0.45", alpha=al, lw=2.0, z=1)
    pivot(ax, j, ec="0.7", alpha=al * 2.2, r=0.02, lw=1.6, z=1)

# barres principales
J = rod(ax, O, th1, z=3)
E = rod(ax, J, th2, z=3)
pivot(ax, O)
pivot(ax, J)

# centres de masse + poids
G1, G2 = O + 0.5 * l * unit(th1), J + 0.5 * l * unit(th2)
for G in (G1, G2):
    ax.add_patch(Circle(G, 0.027, fc="k", ec="k", zorder=7))
    ax.add_patch(
        FancyArrowPatch(
            G,
            G + np.array([0, -0.38]),
            arrowstyle="-|>",
            mutation_scale=34,
            color="k",
            lw=2.0,
            shrinkA=0,
            shrinkB=0,
            zorder=7,
        )
    )

# angles
angle(ax, O, th1, r=0.45, ref_len=0.52)
angle(ax, J, th2, r=0.47, ref_len=0.55)

# cotes
dimension(ax, O, th1)
dimension(ax, J, th2)

# étiquettes
fs = 23
ax.text(0.195, 0.09, "(0, 0)", fontsize=fs, ha="center", va="center")
ax.text(0.44, -0.43, r"$(x_1,y_1)$", fontsize=fs, ha="center", va="center")
ax.text(0.915, -1.20, r"$(x_2,y_2)$", fontsize=fs, ha="center", va="center")
ax.text(0.05, -0.60, r"$\theta_1$", fontsize=fs, ha="center", va="center")
ax.text(0.44, -1.25, r"$\theta_2$", fontsize=fs, ha="center", va="center")
ax.text(-0.03, -0.78, r"$mg$", fontsize=fs, ha="center", va="center")
ax.text(0.53, -1.60, r"$mg$", fontsize=fs, ha="center", va="center")

fig.subplots_adjust(0, 0, 1, 1)
plt.show()
