from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask import Flask, jsonify, request, redirect, url_for, send_file, Response

import json
import uuid
import sqlite3
import bcrypt
import os
import time


def create_db():
    if os.path.exists("login.db"):
        os.remove("login.db")

    conn = sqlite3.connect("login.db")
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS login_data (
            email TEXT PRIMARY KEY,
            display_name TEXT,
            password BLOB
        )
    ''')

    conn.commit()
    conn.close()


def create_account(email, username, password):
    user_db = sqlite3.connect("login.db")
    cursor = user_db.cursor()

    cursor.execute("SELECT * FROM login_data WHERE email = ?", (email,))
    results = cursor.fetchall()

    if len(results) > 0:
        return False

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    cursor.execute("INSERT INTO login_data (email, display_name, password) VALUES (?, ?, ?)", (email, username, hashed_password))
    user_db.commit()

    return True


def validate_login(email, password):
    user_db = sqlite3.connect("login.db")
    cursor = user_db.cursor()

    cursor.execute("SELECT password FROM login_data WHERE email = ?", (email,))
    results = cursor.fetchall()

    if len(results) == 0:
        return False

    return bcrypt.checkpw(password.encode('utf-8'), results[0][0])


def get_displayname(email):
    user_db = sqlite3.connect("login.db")
    cursor = user_db.cursor()

    cursor.execute("SELECT display_name FROM login_data WHERE email = ?", (email,))
    results = cursor.fetchall()

    if len(results) > 0:
        return results[0]
    return "No Name Set"


class User(UserMixin):
    def __init__(self, user_id):
        self.id = user_id
        self.username = get_displayname(user_id)


UPLOAD_FOLDER = "uploads"


class WebServer:
    def __init__(self, server_instance):
        self.server = server_instance

        self.app = Flask(__name__)  # Initialize Flask
        self.app.secret_key = os.urandom(24)
        self.setup_routes()  # Setup routes

        self.login_manager = LoginManager()
        self.login_manager.unauthorized_callback = self.home
        self.login_manager.user_loader(self.user_loader)
        self.login_manager.init_app(self.app)

        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)

    def setup_routes(self):
        self.app.add_url_rule('/', view_func=self.home)
        self.app.add_url_rule('/account', view_func=self.account)
        self.app.add_url_rule('/account_settings', view_func=self.account_settings)

        self.app.add_url_rule('/signup', view_func=self.signup, methods=['POST'])
        self.app.add_url_rule('/login', view_func=self.login, methods=['POST'])
        self.app.add_url_rule('/get_queue', view_func=self.get_queue, methods=['GET'])
        self.app.add_url_rule('/enqueue_files', view_func=self.enqueue_files, methods=['POST'])
        self.app.add_url_rule('/get_process_info', view_func=self.get_current_process, methods=['POST'])
        self.app.add_url_rule('/task/<task_id>', view_func=self.get_task, methods=['GET'])
        self.app.add_url_rule('/task/share/<task_id>', view_func=self.get_task_share, methods=['GET'])
        self.app.add_url_rule('/task/share/enable/<task_id>', view_func=self.enable_task_share, methods=['GET'])
        self.app.add_url_rule('/<path:filename>', view_func=self.serve_static_image)
        self.app.add_url_rule('/task/view_task_progress', view_func=self.view_task_progress)
        self.app.add_url_rule('/task/get_task_image', view_func=self.get_task_image, methods=['GET'])
        self.app.add_url_rule('/task/get_task_scan_areas', view_func=self.get_task_scan_areas, methods=['GET'])

    @staticmethod
    def user_loader(user_id):
        user = User(user_id)
        login_user(user)
        return user

    @staticmethod
    def serve_html(page):
        with open(f"static/html/{page}.html", 'r') as f:
            return f.read()

    def home(self):
        return self.serve_html("index")

    @login_required
    def account(self):
        return self.serve_html("account")

    @login_required
    def account_settings(self):
        return self.serve_html("my_account")

    def signup(self):
        result = create_account(request.form.get('email'), request.form.get('name'), request.form.get('password'))

        if result is True:
            user = User(request.form.get('email'))
            login_user(user)
            return redirect(url_for('account'))
        else:
            return {"error": "Account Creation Failed"}, 200

    def login(self):
        result = validate_login(request.form.get('email'), request.form.get('password'))

        if result is True:
            user = User(request.form.get('email'))
            login_user(user)
            return redirect(url_for('account'), 200)

        else:
            return {"error": "Account Login Failed"}, 200

    @login_required
    def get_queue(self):
        return self.server.get_queue(current_user.id)

    @login_required
    def enqueue_files(self):
        for file in request.files.getlist('files'):
            if file.filename != '':  # Check if there's a file selected
                file_id = str(uuid.uuid4())
                extension = file.filename.split('.')[-1]

                if extension == 'tif' or extension == 'tiff' or extension == 'isyntax':
                    file_path = os.path.join(UPLOAD_FOLDER, file_id + "." + extension)
                    file.save(file_path)  # Save the file

                    self.server.enqueue_file(current_user.id, file_id + "." + extension)

        return {}, 200

    def is_currently_scanning_user(self, user):
        return self.server.queue_id_to_user_id(self.server.current_task) == user.id

    def does_user_own_task(self, user, task_path):
        return self.server.queue_id_to_user_id(task_path) == user.id

    def file_is_shared(self, task_id):
        return self.server.get_share_status(task_id) == 1

    @login_required
    def enable_task_share(self, task_id):
        if self.does_user_own_task(current_user, task_id):
            self.server.set_share_status(task_id, 1)

            if self.file_is_shared(task_id) is False:
                return {}, 500

            return {}, 200

        return {}, 500

    @login_required
    def get_current_process(self):
        if self.is_currently_scanning_user(current_user):
            return self.server.processor.get_scan_info(), 200
        return {}, 401

    @login_required
    def get_task_scan_areas(self):
        if self.does_user_own_task(current_user, request.args.get('id')) or self.file_is_shared(request.args.get('id')):
            return self.server.get_task_scan_areas(request.args.get('id')), 200
        return {}, 401

    @login_required
    def get_task(self, task_id):
        if self.does_user_own_task(current_user, task_id):
            return self.serve_html("task").replace('js_task_id = ""', f'js_task_id = "{task_id}"')

        if self.file_is_shared(task_id):
            return self.serve_html("task").replace('js_task_id = ""', f'js_task_id = "{task_id}"')

        return {}, 401

    @login_required
    def get_task_share(self, task_id):
        if self.does_user_own_task(current_user, task_id):
            return self.serve_html("share_me").replace('js_task_id = ""', f'js_task_id = "{task_id}"')

        return {}, 401

    @login_required
    def get_task_image(self):
        if self.does_user_own_task(current_user, request.args.get('id')) or self.file_is_shared(request.args.get('id')):
            return jsonify(self.server.get_task_image(request.args.get("id"))), 200
        return {}, 401

    @login_required
    def serve_static_image(self, filename):
        if filename == "favicon.ico":
            return send_file("favicon.ico", mimetype='image/ico', as_attachment=True)

        full_static_image = os.path.join(os.getcwd(), f"static/imgs/{filename.split("/")[-1]}")
        if os.path.exists(full_static_image):
            return send_file(full_static_image, mimetype='image/png', as_attachment=True)

        # Set the folder where your static files (e.g., images) are stored
        return send_file("generated_images/" + filename, mimetype='image/png', as_attachment=True)

    def stream_task_progress(self, task_id):
        last_pos = None
        while task_id == self.server.current_task:
            data = self.server.get_task_info(task_id)

            if (data["x"], data["y"]) != last_pos:
                yield f"data: {json.dumps(data)}\n\n"
                last_pos = (data["x"], data["y"])

            else:
                time.sleep(0.01)

        yield f"data: {json.dumps(self.server.get_task_info(task_id))}\n\n"

    @login_required
    def view_task_progress(self):  # Stream Data
        if self.does_user_own_task(current_user, request.args.get('id')) or self.file_is_shared(request.args.get('id')):
            return Response(self.stream_task_progress(request.args.get('id')),
                            content_type='text/event-stream')
        return {}, 401

    def run(self, debug=False):
        ip = "192.168.1.120"
        port = 80

        self.app.run(ip, port, debug=debug)


if __name__ == "__main__":
    create_db()
