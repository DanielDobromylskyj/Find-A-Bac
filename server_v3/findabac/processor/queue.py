import os, time

from .scan_item import ScanItem


class Queue:
    SCAN_PATH = "scans/"
    STORAGE_AWAITING_PATH = "scans/awaiting/"

    def __init__(self, openslide):
        self.openslide = openslide

        self.awaiting_queue = []
        self.processing_queue = []
        self.recently_processed_queue = {}

        self.shutting_down = False
        self.load_after_shutdown()


    def move_to_processing(self, task_id):
        task = self.get_task_by_id(task_id)

        if task is None:
            raise Exception("Task not found")

        self.awaiting_queue.remove(task)
        self.processing_queue.append(task)


    def move_to_cold_storage(self, task_id):
        task = self.get_task_by_id(task_id)

        if task is None:
            raise Exception("Task not found")

        self.processing_queue.remove(task)

        with open(os.path.join(Queue.SCAN_PATH, str(task_id)), "wb") as f:
            f.write(task.to_bytes(archive=False))


        if not task.get_user_id() in self.recently_processed_queue:
            self.recently_processed_queue[task.get_user_id()] = []

        self.recently_processed_queue[task.get_user_id()].append(task)

        if len(self.recently_processed_queue[task.get_user_id()]) > 4:
            self.recently_processed_queue[task.get_user_id()].pop(0)


    def archive_task(self, task_id):
        task = self.get_task_by_id(task_id)

        if task is None:
            raise Exception("Task not found")

        with open(os.path.join(Queue.SCAN_PATH, str(task_id)), "wb") as f:
            f.write(task.to_bytes(archive=True))

    def get_tasks_by_user(self, user_id):
        tasks = []

        for task in self.processing_queue:
            if task.get_user_id() == user_id:
                tasks.append(task)

        for task in self.awaiting_queue:
            if task.get_user_id() == user_id:
                tasks.append(task)

        return tasks

    def get_recently_processed(self, user_id):
        if user_id in self.recently_processed_queue:
            return self.recently_processed_queue[user_id]
        return []

    def get_next_task(self):
        if len(self.awaiting_queue) == 0:
            return None

        task = self.awaiting_queue[0]

        self.move_to_processing(task.get_task_id())
        return task

    def get_task_by_id(self, task_id):
        # Search HOT storage
        for task in self.processing_queue:
            if task.get_task_id() == task_id:
                return task

        # Search Warm Storage
        for task in self.awaiting_queue:
            if task.get_task_id() == task_id:
                return task

        # Search Cold Storage
        if os.path.exists(os.path.join(Queue.SCAN_PATH, str(task_id))):
            with open(os.path.join(Queue.SCAN_PATH, str(task_id)), "rb") as f:
                file_bytes = f.read()

            return ScanItem.from_bytes(self.openslide, file_bytes)

        return None


    def enqueue_task(self, task):
        if self.shutting_down:
            return

        self.awaiting_queue.append(task)


    def shutdown(self):
        if self.shutting_down:
            return

        self.shutting_down = True

        print("[Server] Shutting down -> Saving Awaiting Queue")

        for task in self.awaiting_queue:
            task_bytes = task.to_bytes(False)

            with open(os.path.join(Queue.STORAGE_AWAITING_PATH, str(task.get_task_id())), "wb") as f:
                f.write(task_bytes)

        print(f"[Server] Saved {len(self.awaiting_queue)} tasks from queue to disk")
        self.awaiting_queue = []

        print(f"[Server] Awaiting all currently processing tasks to end...")

        while len(self.processing_queue) > 0:
            time.sleep(1)  # Await to finish processing current items before shutdown

        print("[Server] Queue shutdown")

    def load_after_shutdown(self):
        self.shutting_down = False

        print("[Server] Loading Awaiting Queue From Disk")

        for file in os.listdir(Queue.STORAGE_AWAITING_PATH):
            full_path = os.path.join(Queue.STORAGE_AWAITING_PATH, file)

            with open(full_path, "rb") as f:
                file_bytes = f.read()

            task = ScanItem.from_bytes(self.openslide, file_bytes)
            self.awaiting_queue.append(task)
            print(f"[Server] Loaded {file}")

            os.remove(full_path)

