import numpy as np
import matplotlib.pyplot as plt

img = np.random.rand(500, 500)
plt.imshow(img, cmap="gray");

img = plt.imread("data/balloon.png")
plt.imshow(img)
print("Taille de l'image {}x{} pixels".format(*img.shape[:2]))

img1 = np.dot(img, [0.2126, 0.7152, 0.0722])
# ou img1 = img @ [0.2126, 0.7152, 0.0722] depuis Python 3.5
fig, axes = plt.subplots(1, 2)
axes[0].imshow(img)
axes[1].imshow(img1, cmap="gray");

import numpy as np
img = np.zeros((600, 600, 3), dtype=np.float)
plt.imshow(img);

img[:400, :400, 0] = 1.0
img[200:, 200:, 1] = 1.0
plt.imshow(img);

import numpy as np
img = np.zeros((600, 600, 3), dtype=np.float)
iy, ix = np.ogrid[:600, :600]

def get_mask(center=(300, 300), radius=150):
    dist2center = np.sqrt((ix - center[0])**2 + (iy - center[1])**2)
    mask = dist2center < radius
    return dist2center, mask

dist2center, mask = get_mask()

plt.imshow(dist2center)
cbar = plt.colorbar()
cbar.set_label("distance to disk center")

dist2center, mask = get_mask(center=(200, 200))
img[mask, 0] = 1
dist2center, mask = get_mask(center=(400, 200))
img[mask, 1] = 1
dist2center, mask = get_mask(center=(300, 400))
img[mask, 2] = 1

plt.imshow(img);
