from queue import Queue, QueueItem
from web_api import WebInterface
from processor_controller import ProcessorPool

import time


class Server:
    def __init__(self):
        self.queue = Queue()
        self.web_interface = WebInterface(self)
        self.processor_pool = ProcessorPool("../ZF2.pyn")

    def enqueue_files(self, current_user, request):
        for file in request.files.getlist('files'):
            if file.filename != '':  # Check if there's a file selected
                file_id = str(uuid.uuid4())
                extension = file.filename.split('.')[-1]

                if extension == 'tif' or extension == 'tiff' or extension == 'isyntax':
                    file_path = os.path.join(UPLOAD_FOLDER, file_id + "." + extension)
                    file.save(file_path)  # Save the file

                    queue_item = QueueItem(file.filename, file_path, current_user.id)
                    self.queue.enqueue(queue_item)

        return {}, 200

    def process_item(self, queue_item):
        pass

    def start(self):
        self.web_interface.start_threaded()

        while True:
            queue_item = self.queue.dequeue()

            if queue_item:
                self.process_item(queue_item)
            else:
                time.sleep(1)


if __name__ == '__main__':
    server = Server()
    server.start()
