import os
import uuid

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

    def enqueue_task(self, path, user_id):
        self.queue.enqueue_task(
            ScanItem(openslide, path, user_id, str(uuid.uuid4()))
        )

    @staticmethod
    def __task_to_simple_dict(task):
        return {
            "task_id": task.get_task_id(),
            "percentage": task.get_percentage(),
        }

    def get_users_active_tasks(self, user_id):
        tasks = self.queue.get_tasks_by_user(user_id)
        return [self.__task_to_simple_dict(task) for task in tasks]

