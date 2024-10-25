from PIL import Image
import numpy as np
import io
from typing import Tuple
import sys
import time


python_version = sys.version_info
if (python_version.major != 3) or (python_version.minor != 7):
    exit(f"You must use python 3.7, Not {python_version.major}.{python_version.minor}.{python_version.micro}")


# Must import in this order, else missing DLL errors will occur
import pixelengine
import softwarerendercontext
import softwarerenderbackend

"""
As A Warning, This class is designed to be interchangeable with a open-slide instance.
However, I have only implemented basic functionality here, as its what I need, for a full
solution, Feel free to contact me or modify this class as required (Good luck with the SDK)!
"""


class Isyntax:
    def __init__(self, path):
        self.path = path

        # noinspection PyUnresolvedReferences
        self.__engine = pixelengine.PixelEngine(
            softwarerenderbackend.SoftwareRenderBackend(),
            softwarerendercontext.SoftwareRenderContext()
        )

        self.__engine["in"].open(self.path)
        self.__wsi = self.__engine["in"]["WSI"]
        self.__view = self.__wsi.source_view

        self.dimensions = self.get_dimensions_at_level(0)

    def get_dimensions_at_level(self, level: int) -> Tuple[int, int]:
        """ Returns the size of a level given the 'level' param"""
        x, y, component = self.__view.dimension_ranges(level)
        return x[2], y[2]

    def read_region(self, location: Tuple[int, int], level: int, dimensions: Tuple[int, int], as_image: bool = False):
        view_ranges = [[
            location[0],
            location[0] + dimensions[0] - (2 ** level),
            location[1],
            location[1] + dimensions[1] - (2 ** level),
            level
        ]]

        data_envelopes = self.__view.data_envelopes(level)
        region = self.__view.request_regions(
                region=view_ranges,
                data_envelopes=data_envelopes,
                enable_async_rendering=False,
                background_color=[254, 254, 254]
            )[0]

        if as_image:
            pixels = np.empty((dimensions[0], dimensions[1], 3), dtype=np.uint8)
            region.get(pixels)
            return Image.fromarray(pixels)

        else:
            pixels = np.empty((dimensions[0] * dimensions[1] * 3), dtype=np.uint8)
            region.get(pixels)
            return pixels

    def get_thumbnail(self, size: Tuple[int, int]) -> Image.Image:
        return Image.open(
            io.BytesIO(self.__engine["MACROIMAGE"].image_data)
        ).resize(size, Image.LANCZOS)

    def test_region_read_speed(self):
        start = time.time()

        for i in range(500):
            data = self.read_region((9000, 9000), 0, (100, 100))

        elapsed = time.time() - start

        print(f"Time per Read (100x100, x500): {elapsed/500*1000:.2f}ms")


if __name__ == '__main__':
    file = Isyntax("39335_3.isyntax")
    file.test_region_read_speed()

