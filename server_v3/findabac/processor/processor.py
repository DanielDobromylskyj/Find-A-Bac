import os
import uuid
import threading
import time

OPENSLIDE_PATH = os.path.abspath(r'openslide-win64\bin')
if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

from .queue import Queue
from .scan_item import ScanItem


class Processor:
    def __init__(self):
        self.queue = Queue(openslide)

        self.shutdown = False

    def enqueue_task(self, path, user_id):
        self.queue.enqueue_task(
            ScanItem(openslide, path, user_id, str(uuid.uuid4()))
        )

    @staticmethod
    def __task_to_simple_dict(task):
        return {
            "task_id": task.get_task_id(),
            "percentage": task.get_progress(),
            "name": os.path.basename(task.get_path())[36:]  # Starts with UUID4 then name
        }

    def get_users_active_tasks(self, user_id):
        tasks = self.queue.get_tasks_by_user(user_id)
        return [self.__task_to_simple_dict(task) for task in tasks]

    def get_users_recently_complete(self, user_id):
        tasks = self.queue.get_recently_processed(user_id)
        return [self.__task_to_simple_dict(task) for task in tasks]

    def _start(self):  # todo - this is just a placeholder function for now
        while not self.shutdown:
            task = self.queue.get_next_task()

            if task:
                print(f"[Server] Processing task: {task.get_task_id()}")
                i = -8
                while i < 100:
                    task.set_progress(max(0, i))
                    i += 1
                    time.sleep(1)

                self.queue.move_to_cold_storage(task.get_task_id())

            else:
                time.sleep(1)


    def start(self):
        processing_thread = threading.Thread(target=self._start)
        processing_thread.start()


