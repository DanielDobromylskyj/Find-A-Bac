import os
import sqlite3
import threading
import sys
import math
import time
import numpy as np
import uuid
from PIL import ImageDraw

import web_server

# load main.py module
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import main as neural_network


# noinspection PyUnresolvedReferences
import tissue_selector


OPENSLIDE_PATH = os.path.abspath(r'openslide-win64\bin')

if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


# Server.py stuff
def create_db():
    conn = sqlite3.connect('server.db')
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS queue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        file_path TEXT,
        detections INTIGER,
        locations TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        shared INTIGER DEFAULT 0
    )''')

    conn.commit()
    conn.close()


def save_image(img):
    img_id = str(uuid.uuid4()) + ".png"
    path = "generated_images/" + img_id
    img.save(path)
    return img_id


def draw_region(slide, x, y):
    scale_x = slide.dimensions[0] / 1024
    scale_y = slide.dimensions[1] / 1024

    img = slide.get_thumbnail((1024, 1024))
    draw = ImageDraw.Draw(img)

    x, y = x / scale_x, y / scale_y
    r = 10
    bounding_box = [(x - r, y - r), (x + r, y + r)]

    draw.ellipse(bounding_box, outline="red", width=3)
    return img


class Processor:
    def __init__(self, path):
        self.path = path
        self.network = neural_network.Network.load(self.path)

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
    def __load(path):
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
            self.slide = self.__load(path)
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
            if type(region_data) != np.ndarray:
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


class Server:
    def __init__(self):
        self.web_server = web_server.WebServer(self)
        self.loop_thread = threading.Thread(target=self.server_loop)

        self.processor = Processor("../ZF.pyn") if os.path.exists("../ZF.pyn") else Processor(
            "../ZF2.pyn")  # testing only
        self.running = True

        if os.path.exists("queue_pos.txt"):
            with open("queue_pos.txt", "r") as f:
                self.current_queue_pos = int(f.read())
        else:
            self.current_queue_pos = 1

        self.currently_processing = 1
        self.current_task = None

    def increment_queue(self):
        self.current_queue_pos += 1
        self.current_task = None

        with open("queue_pos.txt", "w") as f:
            f.write(str(self.current_queue_pos))

    def get_queue(self, user_id):
        db = sqlite3.connect('server.db')
        c = db.cursor()

        c.execute("SELECT * FROM queue WHERE user_id = ?", (user_id,))
        results = c.fetchall()

        db.close()

        return [
            {
                "id": result[2],
                "pos_in_queue": -3 if int(result[0]) < 0 else -2 if self.current_queue_pos == result[0] else (
                    (result[0] - self.current_queue_pos) if (result[0] - self.current_queue_pos) > 0 else -1),
            } for result in results
        ]

    @staticmethod
    def queue_id_to_user_id(queue_path: str):
        db = sqlite3.connect('server.db')
        c = db.cursor()

        c.execute("SELECT user_id FROM queue WHERE file_path = ?", (queue_path,))
        results = c.fetchone()
        db.close()

        if results:
            return results[0]

    @staticmethod
    def task_id_to_queue_id(task_id):
        db = sqlite3.connect('server.db')
        c = db.cursor()

        c.execute("SELECT id FROM queue WHERE file_path = ?", (task_id,))
        results = c.fetchone()
        db.close()

        if results:
            return results[0]

    @staticmethod
    def get_share_status(file_path: str):
        db = sqlite3.connect('server.db')
        c = db.cursor()

        c.execute("SELECT shared FROM queue WHERE file_path = ?", (file_path,))
        results = c.fetchone()
        db.close()

        if results:
            return results[0]

    @staticmethod
    def set_share_status(file_path: str, value: int):
        db = sqlite3.connect('server.db')
        c = db.cursor()

        c.execute("UPDATE queue SET shared = ? WHERE file_path = ?", (value, file_path,))

        db.commit()
        db.close()

    @staticmethod
    def enqueue_file(user_id, file_path):
        db = sqlite3.connect('server.db')
        c = db.cursor()

        c.execute("INSERT INTO queue (user_id, file_path) VALUES (?, ?)", (user_id, file_path))
        db.commit()
        db.close()

    def get_full_queue(self):
        db = sqlite3.connect('server.db')
        c = db.cursor()

        c.execute("SELECT * FROM queue WHERE id > ?", (self.current_queue_pos - 1,))
        results = c.fetchall()

        db.close()

        return [
            {
                "id": result[2],
                "pos_in_queue": -2 if self.current_queue_pos == result[0] else (
                    (result[0] - self.current_queue_pos) if (result[0] - self.current_queue_pos) >= 1 else -1),
            } for result in results if result[0] >= self.current_queue_pos
        ]

    def clear_old_queue(self):
        db = sqlite3.connect('server.db')
        c = db.cursor()

        # Delete entries older than 2 days
        c.execute("DELETE FROM queue WHERE created_at < datetime('now', '-2 days') AND id < ?",
                  (self.current_queue_pos,))
        db.commit()

        db.close()

    def get_task_scan_areas(self, task_id):
        if self.current_task != task_id:
            slide = self.load_slide(f"uploads/{task_id}")
            scan_x_scale = slide.dimensions[0] / 1024
            scan_y_scale = slide.dimensions[1] / 1024
            scan_areas = [
                [x1 / scan_x_scale, y1 / scan_y_scale, x2 / scan_x_scale, y2 / scan_y_scale]
                for x1, y1, x2, y2 in Processor.optimise(f"uploads/{task_id}")
            ]

            return {
                "scan_areas": scan_areas,
                "processing": False
            }

        else:
            return {
                    "scan_areas": self.processor.nice_regions,
                    "processing": True,
                    "region_index": self.processor.current_region_index,
                    "region_xy": self.processor.get_pos_relative_to_current_region()
                }

    @staticmethod
    def load_slide(path: str):
        if not os.path.exists(path):
            return

        """if path.endswith(".isyntax"):
            return isyntax.Isyntax(path)"""
        return openslide.open_slide(path)

    def get_task_image(self, task_id):
        slide = self.load_slide(f"uploads/{task_id}")
        image = slide.get_thumbnail((1024, 1024))

        return {
            "img_url": save_image(image),
            "width": 100 * (1024 / slide.dimensions[0]),
            "height": 100 * (1024 / slide.dimensions[1])
        }

    def get_task_info(self, task_id):
        if task_id == self.current_task:
            coords = self.processor.get_new_images()
            images = [
                (save_image(
                    self.processor.slide.read_region(
                        (
                            x, y
                        ), 1, (100, 100)
                    )
                ),
                 save_image(
                     draw_region(
                         self.processor.slide,
                         x, y
                     )
                 ))
                for x, y in coords
            ]

            return {
                "progress": round(self.processor.percent_complete),
                "integerResult": self.processor.detections,
                "imagePaths": images,
                "complete": False,
                "x": round(self.processor.scan_location[0]),
                "y": round(self.processor.scan_location[1]),
                "scan_area": self.processor.current_region_index,
            }

        else:
            results = self.get_old_results(task_id)

            return {
                "progress": 100,
                "integerResult": results[0],
                "imagePaths": self.load_images(task_id, results[1]),
                "complete": True,
                "x": 0,
                "y": 0,
                "scan_area": -1,
            }

    def full_cleanup(self):
        for sub_path in os.listdir("generated_images"):
            path = os.path.join(f"generated_images", sub_path)

            if time.time() - os.path.getctime(path) > (48 * 60 * 60):
                os.remove(path)

        for sub_path in os.listdir("uploads"):
            path = os.path.join(f"uploads", sub_path)

            if time.time() - os.path.getctime(path) > (48 * 60 * 60):
                os.remove(path)

        self.clear_old_queue()

    def load_images(self, task_id, locations):
        slide = self.load_slide(f"uploads/{task_id}")

        if not locations:
            return []

        images = []
        for location in [eval(pos) for pos in locations.split("|")]:
            img1 = slide.read_region((round(location[0]), round(location[1])), 0, (100, 100))
            img2 = draw_region(slide, round(location[0]), round(location[1]))
            images.append((save_image(img1), save_image(img2)))

        return images

    def get_old_results(self, job_id):
        conn = sqlite3.connect('server.db')
        cursor = conn.cursor()

        cursor.execute('''
        SELECT detections, locations FROM queue WHERE file_path == ?
        ''', (job_id,))

        results = cursor.fetchone()
        conn.close()

        return results

    def save_results_to_db(self, job_id, status):
        conn = sqlite3.connect('server.db')
        cursor = conn.cursor()

        cursor.execute('''UPDATE queue
                          SET detections = ?, locations = ?
                          WHERE file_path == ?;''',
                       (self.processor.detections, "|".join([
                                        str(detection_pos) for detection_pos in self.processor.get_images()
                                    ]), job_id))

        conn.commit()
        if status is False:
            cursor.execute('''UPDATE queue
                              SET id = ?
                              WHERE file_path == ?''', (-self.task_id_to_queue_id(job_id), job_id))

            conn.commit()
        conn.close()

    def start(self):
        self.loop_thread.start()
        self.web_server.run()

    def server_loop(self):
        last_full_cleanup = time.time()

        while self.running:
            queue = self.get_full_queue()

            if last_full_cleanup + 1800 > time.time():
                last_full_cleanup = time.time()
                self.full_cleanup()

            if len(queue) != 0:
                scan_job = queue[0]
                self.current_task = scan_job["id"]

                status = self.processor.scan("uploads/" + scan_job["id"])
                self.save_results_to_db(scan_job["id"], status)

                self.increment_queue()
            else:
                time.sleep(2)


if __name__ == '__main__':
    create_db()
    server = Server()
    server.start()
