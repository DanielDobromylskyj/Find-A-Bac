from db_manager import create_db, does_db_exist, MyDB
import uuid


class QueueItem:
    def __init__(self, task_display_name, file_path, user_id):
        self.task_display_name = task_display_name
        self.file_path = file_path
        self.user_id = user_id
        self.queue_id = str(uuid.uuid4())


class Queue:
    def __init__(self):
        self.__task_storage = init_storage_db()
        self.__queue_db = []

    def push_queue_item_to_storage(self, queue_item, processor_result):
        self.__task_storage.insert(
            "completed_tasks",
            ("task_id", "task_display_name", "user_id", "file_path",
             "detection_locations", "completion_status", "share_status"),
            (queue_item.task_id, queue_item.task_display_name, queue_item.user_id, queue_item.file_path,
             processor_result.detection_locations, processor_result.completion_status, 0)
        )

    def dequeue(self):
        if len(self.__queue_db) != 0:
            return self.__queue_db.pop(0)

    def enqueue(self, queue_item):
        self.__queue_db.append(queue_item)


def init_storage_db():
    if not does_db_exist("task_storage.db"):
        create_db("task_storage.db", [
            {
                "name": "completed_tasks",
                "args": [
                    ("task_id", "TEXT PRIMARY"),
                    ("task_display_name", "TEXT"),
                    ("user_id", "TEXT"),
                    ("file_path", "TEXT"),
                    ("detection_locations", "TEXT"),
                    ("completion_status", "INTEGER"),
                    ("share_status", "INTEGER"),
                    ("upload_time", "DATETIME DEFAULT CURRENT_TIMESTAMP")
                ]
            }
        ])

    return MyDB("task_storage.db")
