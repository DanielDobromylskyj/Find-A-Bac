import lz4.frame
from io import BytesIO
from PIL import Image


class ScanItem:
    def __init__(self, openslide, path, user_id, task_id, is_archived=False):
        self.__detections_locations = []
        self.__detection_images = []
        self.__path = path
        self.__user_id = user_id
        self.__task_id = task_id
        self.__progress = 0
        self.__is_archived = is_archived

        if not is_archived:
            self.__openslide = openslide
            self.__slide = self.__openslide.open_slide(self.__path)

    def set_progress(self, value):
        self.__progress = value

    def get_progress(self):
        return self.__progress

    def get_task_id(self):
        return self.__task_id

    def get_user_id(self):
        return self.__user_id

    def get_path(self):
        return self.__path

    def get_detections_locations(self):
        return self.__detections_locations

    def get_detections_image(self, location) -> Image.Image:
        if location not in self.__detections_locations:
            raise Exception("Invalid location")

        if self.__is_archived:
            location_index = self.__detections_locations.index(location)
            return self.__detection_images[location_index]

        else:
            if self.__detection_images[location] is None:  # Load it first time
                self.__detection_images[location] = self.__slide.read_region(location, 0, (100, 100))

            return self.__detection_images[location]

    def add_detection(self, location):
        self.__detections_locations.append(location)
        self.__detection_images.append(None)

    def to_bytes(self, archive: bool):
        data = bytearray()

        if archive is False and self.__is_archived:
            raise FileNotFoundError("Cannot undo archive on a archived scan")

        data += b"A" if archive else b"R"  # Get from "Archive" or Get from "Raw"
        data += len(self.__path).to_bytes(2, byteorder='little') + self.__path.encode()
        data += len(self.__user_id).to_bytes(2, byteorder='little') + self.__user_id.encode()
        data += len(self.__task_id).to_bytes(2, byteorder='little') + self.__task_id.encode()
        data += len(self.__detections_locations).to_bytes(2, byteorder='little')

        for i, location in enumerate(self.__detections_locations):
            data += location[0].to_bytes(8, byteorder='little') + location[1].to_bytes(8, byteorder='little')

            if archive:
                region = self.get_detections_image(location)

                buffer = BytesIO()
                region.save(buffer, format='PNG')
                image_bytes = buffer.getvalue()
                data += len(image_bytes).to_bytes(8, byteorder='little') + image_bytes

        return lz4.frame.compress(data)

    @staticmethod
    def _load_image_from_bytes(image_bytes):
        buffer = BytesIO(image_bytes)
        return Image.open(buffer)

    def _from_bytes_add_location_and_image(self, location, image):
        """ DO NOT USE -> For loading only"""
        self.__detections_locations.append(location)
        self.__detection_images.append(image)

    @staticmethod
    def from_bytes(openslide, data):
        data = lz4.frame.decompress(data)
        index = 0

        is_archived = data[index] == 65  # b"A"
        index += 1

        path_length = int.from_bytes(data[index:index+2], byteorder='little')
        index += 2

        path = data[index:index+path_length].decode()
        index += path_length

        user_id_length = int.from_bytes(data[index:index+2], byteorder='little')
        index += 2

        user_id = data[index:index+user_id_length].decode()
        index += user_id_length

        task_id_length = int.from_bytes(data[index:index + 2], byteorder='little')
        index += 2

        task_id = data[index:index + task_id_length].decode()
        index += task_id_length

        item = ScanItem(openslide, path, user_id, task_id, is_archived)

        number_of_detections = int.from_bytes(data[index:index+2], byteorder='little')
        index += 2

        for _ in range(number_of_detections):
            location_x = int.from_bytes(data[index:index+8], byteorder='little')
            index += 8

            location_y = int.from_bytes(data[index:index+8], byteorder='little')
            index += 8

            if is_archived:
                image_length = int.from_bytes(data[index:index+8], byteorder='little')
                index += 8

                image_bytes = data[index:index+image_length]
                index += image_length

                image = ScanItem._load_image_from_bytes(image_bytes)
                item._from_bytes_add_location_and_image((location_x, location_y), image)

            else:
                item.add_detection((location_x, location_y))

        return item
