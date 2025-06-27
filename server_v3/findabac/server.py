import os
import sqlite3
import uuid

from argon2 import PasswordHasher, exceptions
from importlib.resources import files
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user, login_manager
from flask import Flask, jsonify, request, redirect, url_for, send_file, Response

from .database import setup_database
from .processor.processor import Processor


class User(UserMixin):
    def __init__(self, user_id):
        self.id = user_id


package_root = files('findabac')

class WebServer:
    def __init__(self):
        self.app = Flask(__name__)  # Initialize Flask
        self.app.secret_key = os.urandom(24)

        self.app.config['UPLOAD_FOLDER'] = "uploads"

        self.setup_routes()  # Setup routes

        self.database_path = 'user_data.db'

        if not os.path.exists(self.database_path):
            setup_database(self.database_path)

            with open("donotshare.txt", "r") as donotshare:
                pswd = donotshare.read()

            self.create_user("daniel.dobromylskyj@outlook.com", pswd, 2)

        self.login_manager = LoginManager()
        self.login_manager.unauthorized_callback = self.home
        self.login_manager.user_loader(self.user_loader)
        self.login_manager.init_app(self.app)

        self.processor = Processor()
        self.processor.start()

    @staticmethod
    def user_loader(user_id):
        user = User(user_id)
        login_user(user)
        return user

    def setup_routes(self):
        self.app.add_url_rule('/', view_func=self.home)
        self.app.add_url_rule('/login', view_func=self.login, methods=['POST'])
        self.app.add_url_rule('/dashboard', view_func=self.dashboard)
        self.app.add_url_rule('/queue', view_func=self.queue)

        self.app.add_url_rule('/api/upload', view_func=self.upload_file, methods=['POST'])

        self.app.add_url_rule('/api/tasks', view_func=self.get_users_active_tasks, methods=['GET'])

        self.app.add_url_rule('/static/<path:path>', view_func=self.serve_static)


    def home(self):
        return self.serve_html("login")

    @login_required
    def upload_file(self):
        if 'file' not in request.files:
            return 'No file uploaded.', 400

        files = request.files.getlist('file')
        for file in files:
            if file.filename == '':
                continue

            path = os.path.join(self.app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + str(file.filename))
            file.save(path)

            self.processor.enqueue_task(path, current_user.id)
            print(f"[Server] Saved File '{path}'")

        print("[Debug]", self.processor.get_users_active_tasks(current_user.id))
        return redirect('/dashboard')

    @login_required
    def get_users_active_tasks(self):
        return self.processor.get_users_active_tasks(current_user.id), 200

    @login_required
    def dashboard(self):
        return self.serve_html("dashboard")

    @login_required
    def queue(self):
        return self.serve_html("queue")

    def create_user(self, email, password, auth_level=0):
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        hashed = PasswordHasher().hash(password)
        cursor.execute('''INSERT INTO users (email, pswd_hash, auth_level) VALUES (?, ?, ?)''', (email, hashed, auth_level))

        conn.commit()

        cursor.close()
        conn.close()

        print("[Server] Created user {}".format(email))



    def login(self):
        email = request.json.get('email')
        password = request.json.get('password')

        db = sqlite3.connect(self.database_path)
        cursor = db.cursor()

        cursor.execute('''SELECT pswd_hash FROM users WHERE email = ?''', (email,))
        password_hash = cursor.fetchone()

        if not password_hash:
            return {"error": "Incorrect email or password"}, 200

        cursor.close()
        db.close()

        try:
            PasswordHasher().verify(password_hash[0], password)
            user = User(request.form.get('email'))
            login_user(user)
            return {"redirect_url": "dashboard"}, 200
        except exceptions.VerifyMismatchError:
            return {"error": "Incorrect email or password"}, 200



    @staticmethod
    def serve_static(path):
        full_static_image = "static/" + path

        if os.path.exists(full_static_image):
            return send_file(full_static_image, mimetype='image/png', as_attachment=True), 200

        return None, 404


    def serve_html(self, name):
        path = os.path.join(package_root.name, "html", f"{name}.html")

        if not os.path.exists(path) or not os.path.isfile(path):
            return self.serve_html("error/404")[0], 404

        with open(path, 'r') as f:
            return f.read(), 200


    def run(self, debug=False):
        ip = "192.168.1.120"
        port = 80

        self.app.run(ip, port, debug=debug)

    def shutdown(self):
        self.processor.queue.shutdown()
