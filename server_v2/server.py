from my_queue import Queue, QueueItem
from web_api import WebInterface
from processor_controller import ProcessorPool

import uuid
import time
import os

UPLOAD_FOLDER = "uploads"


class Server:
    def __init__(self):
        self.queue = Queue(self)
        self.web_interface = WebInterface(self)
        self.processor_pool = ProcessorPool("../ZF.pyn")

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
        self.processor_pool.run(queue_item, queue_item.queue_id)

    def store_queue_item(self, queue_item, result):
        self.queue.push_queue_item_to_storage(queue_item, result)

    def check_for_process_completions(self):
        for task_id in self.processor_pool.processor_tasks:
            prc = self.processor_pool.get_processor(task_id)
            if prc["complete"] is True:
                self.store_queue_item(prc["queue_item"], prc["result"])
                self.processor_pool.release_task(task_id)

    def start(self):
        self.web_interface.start_threaded()

        while True:
            item = self.queue.get()

            if item and self.processor_pool.is_free_processors():
                queue_item = self.queue.dequeue()
                self.process_item(queue_item)
                pass
            else:
                self.check_for_process_completions()
                time.sleep(1)


if __name__ == '__main__':
    server = Server()
    server.start()
