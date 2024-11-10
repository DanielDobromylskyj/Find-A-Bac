from PIL import Image, ImageDraw
import hashlib
import os
import uuid


class ImageHandler:
    def __init__(self):
        self.images, self.image_hashes = self.__load_all_images()

    def __clear_old_cache(self):
        for image in self.images:
            if time.time() - os.path.getatime(image) > 1800:  # 30min
                os.remove(image)

    def __load_all_images(self):
        if not os.path.exists("images"):
            os.mkdir("images")

        return (["images/" + file_name for file_name in os.listdir('images')],
                [self.__hash_data(self.read_file("images/" + file_name)) for file_name in os.listdir('images')])

    def is_image_hash_in_dict(self, image_hash):
        return image_hash in self.image_hashes

    @staticmethod
    def __save_image(image):
        path = "images/" + str(uuid.uuid4()) + ".png"
        image.save(path)
        return path

    def add_image(self, image):
        image_hash = self.__hash_data(image)
        if self.is_image_hash_in_dict(image_hash):
            index = self.image_hashes.index(image_hash)
            return self.images[index]
        else:
            path = self.__save_image(image)
            self.image_hashes.append(image_hash)
            self.images.append(path)
            return path

    @staticmethod
    def read_file(path):
        with open(path, 'rb') as f:
            return f.read()

    @staticmethod
    def __hash_data(data):
        if isinstance(data, Image.Image):
            return hash(data.tobytes())

        else:
            print("Unknown hash", type(data))
            return hash(data)
