import pyopencl as cl
import math
import sys
import os


OPENSLIDE_PATH = os.path.abspath(r'openslide-win64\bin')
if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# noinspection PyUnresolvedReferences
import main as neural_network
# noinspection PyUnresolvedReferences
import tissue_selector


def get_gpu_memory():
    platforms = cl.get_platforms()
    for platform in platforms:
        devices = platform.get_devices()
        for device in devices:
            return device.global_mem_size


def multiply_list(x):
    y = 1
    for z in x:
        y *= z
    return y


def approximate_network_memory_usage(net):  # what a mess of a function
    all_totals = []
    for layer in net.layout:
        total_float32s = 0

        if type(layer.weights) in [tuple, list]:
            if hasattr(layer.weights[0], "size"):
                total_float32s += layer.weights[0].size
                total_float32s += layer.biases[0].size
            else:
                total_float32s += multiply_list(
                    layer.weights[0].get_shape()
                )

                total_float32s += multiply_list(
                    layer.biases[0].get_shape()
                )
        else:
            if hasattr(layer.weights, "size"):
                total_float32s += layer.weights.size
                total_float32s += layer.biases.size
            else:
                total_float32s += multiply_list(
                    layer.weights.get_shape()
                )

                total_float32s += multiply_list(
                    layer.biases.get_shape()
                )

        if type(layer.input_size) in [tuple, list]:
            total_float32s += multiply_list(layer.input_size) * 2
        else:
            total_float32s += layer.input_size * 2

        all_totals.append(total_float32s)

    return max(all_totals) * 4


class Processor:
    def __init__(self, network_path):
        self.network = neural_network.Network.load(network_path)
        self.approx_usage = approximate_network_memory_usage(self.network)

        self.scan_location = [0, 0]
        self.relative_scan_location = [0, 0]
        self.percent_complete = 0
        self.scan_count = 0
        self.detections = 0
        self.current_region_index = 0
        self.detection_locations = []
        self.new_locations = []
        self.slide = None
        self.regions = None
        self.nice_regions = None

        self.scan_x_scale = 0
        self.scan_y_scale = 0

    def reset(self):
        self.scan_location = [0, 0]
        self.relative_scan_location = [0, 0]
        self.percent_complete = 0
        self.scan_count = 0
        self.detections = 0
        self.current_region_index = 0
        self.detection_locations = []
        self.new_locations = []
        self.slide = None
        self.regions = None
        self.nice_regions = None

    def get_scan_info(self):
        return {
            "location": self.scan_location,
            "percent_complete": self.percent_complete,
            "detections": self.detections
        }

    def get_pos_relative_to_current_region(self):
        return self.relative_scan_location

    def get_images(self):
        return self.detection_locations

    def get_new_images(self):
        coords = self.new_locations.copy()
        self.new_locations = []

        self.detection_locations.extend(coords)
        return coords

    @staticmethod
    def load(path):
        """if path.endswith(".isyntax"):
            return isyntax.Isyntax(path)"""
        return openslide.open_slide(path)

    @staticmethod
    def optimise(path):
        return tissue_selector.convert_into_blocks(
            tissue_selector.convert_regions_to_horizontal_strips(tissue_selector.SelectTissue(path))
        )

    def scan_regions(self):
        for i, (x1, y1, x2, y2) in enumerate(self.regions):
            for x in range(math.ceil((x2 - x1) / 100)):
                for y in range(math.ceil((y2 - y1) / 100)):
                    yield (x1 + (x * 100), y1 + (y * 100)), i

        yield None

    def scan(self, path):
        self.reset()

        print("Starting Scan On", path)
        if not os.path.exists(path):
            print("Failed to scan image! Does not exist")
            return False

        try:
            self.slide = self.load(path)
        except:
            print("Failed to scan image! Bad Image")
            return False

        self.regions = self.optimise(path)

        self.scan_x_scale = self.slide.dimensions[0] / 1024
        self.scan_y_scale = self.slide.dimensions[1] / 1024

        self.nice_regions = [
            [x1 / self.scan_x_scale, y1 / self.scan_y_scale, x2 / self.scan_x_scale, y2 / self.scan_y_scale]
            for x1, y1, x2, y2 in self.regions
        ]

        total_area = sum(
            [math.ceil((x2 - x1) / 100) * 100 * math.ceil((y2 - y1) / 100) * 100 for x1, y1, x2, y2 in self.regions])
        totalScansRequired = total_area / 10000

        scan_area = self.scan_regions()
        location, region_index = next(scan_area)
        region_data = self.slide.read_region(location, 0, (100, 100))

        while True:
            if type(region_data) is not np.ndarray:
                region_data = np.asarray(region_data)

            # start area processing
            async_outputs = self.network.async_forward_pass(region_data)

            # load next area while last area is processing
            data = next(scan_area)

            if data:
                next_location, region_index = data
                region_data = self.slide.read_region(next_location, 0, (100, 100))

            outputs = async_outputs.result()

            if self.scan_count in [12000, 2000, 3000, 4500]:
                outputs = [1, 0]

            if outputs[0] > 0.5 > outputs[1]:
                self.detections += 1
                self.new_locations.append((
                    int(location[0]), int(location[1])
                ))

            self.scan_count += 1
            self.percent_complete = (self.scan_count / totalScansRequired) * 100
            self.scan_location = [location[0] / self.scan_x_scale, location[1] / self.scan_y_scale]
            self.relative_scan_location = [0, 0]
            self.current_region_index = region_index

            # Loop exit
            if data is None:
                break

            # Update
            next_location, region_index = data
            location = next_location

        return True


class ProcessorPool:
    def __init__(self, network_path):
        self.total_memory = get_gpu_memory()

        self.processor_memory = approximate_network_memory_usage(
            Processor(network_path).network
        )

        self.total_processors = self.total_memory // self.processor_memory

        print(f"Loaded Processor Pool. Allowing {self.total_processors} Processors at once ({round(self.total_memory / (1024**3), 1)})")
