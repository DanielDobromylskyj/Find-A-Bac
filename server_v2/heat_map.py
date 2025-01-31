import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

OPENSLIDE_PATH = os.path.abspath(r'openslide-win64\bin')
if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


def generate_heatmap(path, output_path, locations, size=None):
    if not size:
        size = (1024, 1024)


    slide = openslide.open_slide(path)
    grayscale_image = slide.get_thumbnail(size).convert("L")

    slide_size = slide.dimensions

    scale_x = size[0] / slide_size[0]
    scale_y = size[1] / slide_size[1]

    heatmap = np.zeros(size, dtype=np.float32)
    radius = 10
    for loc in locations:
        x, y = round(loc[0] * scale_x), round(loc[1] * scale_y)
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                nx, ny = x + dx, y + dy
                distance = np.sqrt(dx ** 2 + dy ** 2)
                if 0 <= nx < size[0] and 0 <= ny < size[1] and distance <= radius:
                    heatmap[ny, nx] += (1 - distance / radius)

    # Normalize the heatmap
    normalized_heatmap = Normalize(0, 255)(heatmap)

    # Overlay heatmap on slide
    plt.imshow(grayscale_image, cmap="gray")
    plt.imshow(normalized_heatmap, cmap="Reds", alpha=0.5, interpolation='none')
    plt.axis("off")

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == "__main__":
    import random
    generate_heatmap("uploads/12e2cc96-841e-448c-8a83-792a44466e24.tiff", [(random.randint(100, 120000), random.randint(100, 120000)) for i in range(50)])
