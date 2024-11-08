import os
import numpy as np
import sys
from PIL import ImageDraw

sys.setrecursionlimit(3000)

OPENSLIDE_PATH = os.path.abspath(r'openslide-win64\bin')

if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


# todo - check to see if we need to fill the internals of a "group" incase we missed any

def AverageBox(data, x, y, size):
    r, g, b = 0, 0, 0

    for offsetX in range(size):
        for offsetY in range(size):
            pixel = data[min(x + offsetX, data.shape[0] - 1)][min(y + offsetY, data.shape[1] - 1)]
            r += pixel[0]
            g += pixel[1]
            b += pixel[2]

    pixel_count = size ** 2
    return r // pixel_count, g // pixel_count, b // pixel_count


def IsBoxTissue(data, x, y, size) -> int:
    avg_colour = AverageBox(data, x, y, size)
    return int((avg_colour[0] < 232) or (avg_colour[1] < 232) or (avg_colour[2] < 248))


def FindImportantRegions(data, detail):
    width, height, colours = data.shape
    return [[IsBoxTissue(data, x, y, detail) for x in range(0, width, detail)] for y in range(0, height, detail)]


def GroupFromPoint(data, data_map, x, y, value):
    x = int(x)
    y = int(y)

    if data_map[x][y] == 1:
        return 0

    count = 1
    data_width = len(data)
    data_height = len(data[0])
    data_map[x][y] = value

    for sx in range(-1, 2):
        for sy in range(-1, 2):
            if (sx == 0) and (sy == 0):
                continue

            if (data_width > (x + sx) > 0) and (data_height > (y + sy) > 0):
                if (data[x + sx][y + sy] == 1) and (data_map[x + sx][y + sy] == 0):
                    count += GroupFromPoint(data, data_map, x + sx, y + sy, value)

    return count


def GroupBoxData(data):
    data_map = [[0 for i in range(len(data[0]))] for j in range(len(data))]
    groups_found = 1
    groups = {}

    for x in range(len(data)):
        for y in range(len(data[0])):
            if (data[x][y] == 1) and (data_map[x][y] == 0):
                group_size = GroupFromPoint(data, data_map, x, y, groups_found)
                groups[groups_found] = group_size
                groups_found += 1

    return data_map, groups


def RemoveRedundantData(data, groups):
    for x in range(len(data)):
        for y in range(len(data[0])):
            if data[x][y] != 0:
                if groups[data[x][y]] < 8:
                    data[x][y] = 0


def OptimiseBoxData(data):
    grouped_data, groups = GroupBoxData(data)
    RemoveRedundantData(grouped_data, groups)
    return grouped_data


def OptimiseBoxes(matrix):
    cols = len(matrix)  # Number of columns (x-axis length)
    rows = len(matrix[0]) if cols > 0 else 0  # Number of rows (y-axis length)
    visited = [[False for _ in range(rows)] for _ in range(cols)]
    boxes = []

    def get_box_dimensions(x, y):
        width, height = 0, 0

        # Calculate width
        while x + width < cols and matrix[x + width][y] != 0 and not visited[x + width][y]:
            width += 1

        # Calculate height
        while y + height < rows and all(
                matrix[x + k][y + height] != 0 and not visited[x + k][y + height] for k in range(width)):
            height += 1

        # Mark as visited
        for i in range(x, x + width):
            for j in range(y, y + height):
                visited[i][j] = True

        return width, height

    for x in range(cols):
        for y in range(rows):
            if matrix[x][y] != 0 and not visited[x][y]:
                width, height = get_box_dimensions(x, y)
                boxes.append((x, y, width, height))

    return boxes


def SelectTissue(path, DEBUG=False):
    slide = openslide.OpenSlide(path)

    smallImageSize = [1024, 1024]
    WholeSlideRegion = slide.get_thumbnail(smallImageSize)

    if DEBUG:
        print(f"[DEBUG] Created thumbnail with target size {smallImageSize}, Real Size: {WholeSlideRegion.size}")

    # Balance: 30
    # Best: 7
    # Max: 300
    box_size = 15

    if DEBUG:
        print(f"[DEBUG] Finding important regions with box_size={box_size}")

    boxed_data = FindImportantRegions(np.asarray(WholeSlideRegion), detail=box_size)

    if DEBUG:
        print("[DEBUG] Removing Redundant Boxes")

    optimised_data = OptimiseBoxData(boxed_data)

    if DEBUG:
        print("[DEBUG] Optimising Box Layout")

    boxes = OptimiseBoxes(optimised_data)

    if DEBUG:
        print(f"[DEBUG] Found important regions. Optimisations complete")

    x_ratio = slide.dimensions[0] / WholeSlideRegion.width
    y_ratio = slide.dimensions[1] / WholeSlideRegion.height

    if DEBUG:
        draw = ImageDraw.Draw(WholeSlideRegion)
        WholeSlideRegion.save("DEBUG.OUTPUT.RAW.png")

    regions_to_search = []
    area_to_search = 0

    for [x1, y1, w, h] in boxes:
        x2, y2 = x1 + w, y1 + h

        px1 = round(x1 * box_size * x_ratio)
        px2 = round(x2 * box_size * x_ratio)
        py1 = round(y1 * box_size * y_ratio)
        py2 = round(y2 * box_size * y_ratio)

        regions_to_search.append((px1, py1, px2, py2))
        area_to_search += (x2 - x1) * (y2 - y1)

        if DEBUG:
            draw.rectangle((x1 * box_size, y1 * box_size, x2 * box_size, y2 * box_size), outline=(255, 0, 0))

    if DEBUG:
        total_area = len(boxed_data) * len(boxed_data[0])
        print(
            f"[DEBUG][INFO] Regions Collected. Trimmed Search Area By {round((1 - (area_to_search / total_area)) * 100, 1)}%")
        print(
            f"[DEBUG][INFO] Whole Image: {round((slide.dimensions[0] * slide.dimensions[1]) / 1000000, 2)}M pixels, {round((slide.dimensions[0] * slide.dimensions[1]) / 10000000)}k regions")
        print(
            f"[DEBUG][INFO] Trimmed: {round(((area_to_search / total_area) * slide.dimensions[0] * slide.dimensions[1]) / 1000000, 2)}M pixels, {round(((area_to_search / total_area) * slide.dimensions[0] * slide.dimensions[1]) / 10000000)}k regions")
        print("[DEBUG] Writing output to 'DEBUG.OUTPUT.ANNOTATED.png'")
        WholeSlideRegion.save("DEBUG.OUTPUT.ANNOTATED.png")
        print("[DEBUG] Write Complete")

    return regions_to_search


def convert_regions_to_horizontal_strips(regions: list):
    """
    Takes in a list of regions (x1, y1, x2, y2) and returns a list of regions (x1, y1, x2, y2) but
    returns them in a format so that when looped over it goes from left to right, top to bottom
    """
    return sorted(regions, key=lambda r: (r[1], r[0]))


def convert_into_blocks(regions: list, block_size=1000):
    """
    Takes in a list of regions (x1, y1, x2, y2) and returns a list of regions (x1, y1, x2, y2)
    formatted to fit into blocks of the specified block_size. Each returned region will correspond
    to a block that covers part or all of the input region.

    :param regions: List of tuples representing regions in the format (x1, y1, x2, y2)
    :param block_size: Size of each block (width and height)
    :return: List of tuples representing regions divided into blocks of the specified size
    """
    output_blocks = []

    for (x1, y1, x2, y2) in regions:
        start_block_x = x1 // block_size
        end_block_x = (x2 - 1) // block_size

        for bx in range(start_block_x, end_block_x + 1):
            block_x1 = max(x1, bx * block_size)
            block_y1 = y1  # max(y1, by * block_size)
            block_x2 = min(x2, (bx + 1) * block_size)
            block_y2 = y2  # min(y2, (by + 1) * block_size)

            output_blocks.append((block_x1, block_y1, block_x2, block_y2))

    return output_blocks


# INFO: To search 27k regions in 20 minutes, each region, must take less than 44ms

if __name__ == "__main__":
    for file in os.listdir("trainingData/"):
        path = f"trainingData/{file}"
        regions = SelectTissue(path, DEBUG=True)
