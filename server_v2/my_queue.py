from db_manager import create_db, does_db_exist, MyDB
import uuid


class QueueItem:
    def __init__(self, task_display_name, file_path, user_id):
        self.task_display_name = task_display_name
        self.file_path = file_path
        self.user_id = user_id
        self.queue_id = str(uuid.uuid4())


class Queue:
    def __init__(self, server):
        self.server = server
        self.__task_storage = init_storage_db()
        self.__queue_db = []

    def push_queue_item_to_storage(self, queue_item, processor_result):
        self.__task_storage.insert(
            "completed_tasks",
            ("task_id", "task_display_name", "user_id", "file_path",
             "detection_locations", "completion_status", "share_status"),
            (queue_item.queue_id, queue_item.task_display_name, queue_item.user_id, queue_item.file_path,
             processor_result.detection_locations, processor_result.status, 0)
        )

    def get_stored_tasks(self, user_id):
        return self.__task_storage.request(
            "completed_tasks",
            "user_id = ?",
            "task_id, task_display_name, detection_locations, completion_status",
            (user_id,)
        )

    def get_queued_tasks(self, user_id):
        return [
            (pos_in_queue, item.queue_id, item.task_display_name)
            for pos_in_queue, item in enumerate(self.__queue_db)
            if item.user_id == user_id
        ]

    def get_processing_tasks(self, user_id):
        found = []
        for task_id in self.server.processor_pool.processor_tasks:
            prc = self.server.processor_pool.get_processor(task_id)
            queue_item = prc["queue_item"]
            processor = prc["processor"]

            if queue_item.user_id == user_id:
                found.append(
                    (queue_item.task_display_name, processor.percent_complete)
                )

        return found

    def get_all_user_queue_items(self, user_id):
        return [
            self.get_queued_tasks(user_id),
            self.get_processing_tasks(user_id),
            self.get_stored_tasks(user_id)
        ]

    def dequeue(self):
        if len(self.__queue_db) != 0:
            return self.__queue_db.pop(0)

    def get(self):
        return self.__queue_db[0] if len(self.__queue_db) != 0 else None

    def enqueue(self, queue_item):
        self.__queue_db.append(queue_item)


def init_storage_db():
    if not does_db_exist("task_storage.db"):
        create_db("task_storage.db", [
            {
                "name": "completed_tasks",
                "args": [
                    ("task_id", "TEXT PRIMARY KEY"),
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
