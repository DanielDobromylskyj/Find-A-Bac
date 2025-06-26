import pyopencl as cl
import numpy as np
import threading
import math
import uuid
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
import gpu_info


class Results:
    def __init__(self, status, locations):
        self.status = status
        self.detection_locations = locations


class Processor:
    def __init__(self, network_path, device):
        self.network = neural_network.Network.load(network_path, device)  # todo add device support

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

                self.detection_locations.append((
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
        self.gpus = gpu_info.get_gpus()
        self.processors = []
        self.processor_tasks = {}

        self.total_max_networks = 0
        self.total_running_networks = 0

        print("[PROCESSOR] Getting Specs For GPUs")
        for gpu in self.gpus:
            prc = Processor(network_path, gpu.device)
            prc_uuid = str(uuid.uuid4())
            gpu.set_network(prc.network)

            max_networks = gpu.get_max_concurrent_networks()
            print(f"\n>>> GPU '{gpu.get_name()}' <<<")
            print("- Max Usage:", max_networks)
            print("- Card UUID:", prc_uuid)

            self.total_max_networks += max_networks
            for i in range(max_networks):
                self.processors.append({
                    "processor": prc,
                    "uuid": prc_uuid,
                    "in_use": False,
                    "complete": False,
                    "queue_item": None,
                    "result": None,
                })

        if self.total_max_networks == 0 and len(self.gpus) > 0:
            print("\n[PROCESSOR][WARNING] Forcing Single GPU - Your GPU(s) may not be powerful enough")
            gpu = self.gpus[0]

            prc = Processor(network_path, gpu.device)
            prc_uuid = str(uuid.uuid4())
            gpu.set_network(prc.network)

            self.processors.append({
                "processor": prc,
                "uuid": prc_uuid,
                "in_use": False,
                "complete": False,
                "queue_item": None,
                "result": None,
            })

        elif len(self.gpus) == 0:
            raise Exception("no gpus detected")

        print(f"\n[PROCESSOR] Loaded GPU devices. Total Networks: {self.total_max_networks}")

    def __get_free_prc(self, task_id):
        for i, prc in enumerate(self.processors):
            if prc["in_use"] is False:
                self.total_running_networks += 1
                self.processor_tasks[task_id] = prc
                prc["in_use"] = True
                prc["complete"] = False
                return self.processors[i], i

    def __release_prc(self, task_id, prc):
        for processor in self.processors:
            if processor["uuid"] == prc["uuid"]:
                self.total_running_networks -= 1
                self.processor_tasks.pop(task_id)
                processor["in_use"] = False

    def _run(self, queue_item, task_id):
        prc, i = self.__get_free_prc(task_id)
        prc = self.processors[i]

        processor = prc["processor"]
        prc["queue_item"] = queue_item
        result = processor.scan(queue_item.file_path)

        if result is False:
            prc["result"] = Results(-1, "")
        else:
            prc["result"] = Results(1, "|".join([str(pos) for pos in processor.detection_locations]))

        prc["complete"] = True

    def release_task(self, task_id):
        self.__release_prc(task_id, self.processor_tasks[task_id])

    def get_processor(self, task_id):
        if task_id in self.processor_tasks:
            return self.processor_tasks[task_id]

    def is_free_processors(self):
        return self.total_running_networks != self.total_max_networks

    def run(self, queue_item, task_id):
        if self.is_free_processors() is False:
            raise Exception("No free prc")

        threading.Thread(target=self._run, args=(queue_item, task_id), daemon=True).start()
