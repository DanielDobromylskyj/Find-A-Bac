from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask import Flask, jsonify, request, redirect, url_for, send_file, Response, abort
from user import UserHandler
import threading
import os


class User(UserMixin):
    def __init__(self, user_id, username):
        self.id = user_id
        self.username = username


class WebInterface:
    def __init__(self, server_instance):
        self.server = server_instance

        self.app = Flask(__name__)  # Initialize Flask
        self.app.secret_key = os.urandom(24)
        self.setup_routes()  # Setup routes

        self.user_manager = UserHandler()
        self.login_manager = LoginManager()
        self.login_manager.unauthorized_callback = self.no_auth
        self.login_manager.user_loader(self.user_loader)
        self.login_manager.init_app(self.app)

    def setup_routes(self):
        self.app.add_url_rule('/', view_func=self.home)
        self.app.add_url_rule('/dashboard', view_func=self.dashboard)
        self.app.add_url_rule('/task/<string:task_id>', view_func=self.load_task)
        self.app.add_url_rule('/task/<string:task_id>/info', view_func=self.get_task_info)
        self.app.add_url_rule('/task/<string:task_id>/heatmap', view_func=self.get_heatmap)
        self.app.add_url_rule('/task/<string:task_id>/share', view_func=self.set_share_status)

        self.app.add_url_rule('/enqueue_files', view_func=self.enqueue_files, methods=['POST'])
        self.app.add_url_rule('/get_queue', view_func=self.get_queue)

        self.app.add_url_rule('/static/img/<path:filename>', view_func=self.serve_site_image)
        self.app.add_url_rule('/css/<path:filepath>', view_func=self.get_css)

        self.app.add_url_rule('/signup', view_func=self.signup, methods=['POST'])
        self.app.add_url_rule('/login', view_func=self.login, methods=['POST'])

    @staticmethod
    def user_loader(user_id):
        user = User(user_id, None)
        login_user(user)
        return user

    @staticmethod
    def serve_html(page):
        with open(f"static/html/{page}.html", 'r') as f:
            return f.read()

    @staticmethod
    def no_auth():
        return redirect(url_for("home"))

    def home(self):
        return self.serve_html("index"), 200

    @login_required
    def enqueue_files(self):
        return self.server.enqueue_files(current_user, request)

    @login_required
    def get_queue(self):
        return self.server.queue.get_all_user_queue_items(current_user.id)

    @login_required
    def dashboard(self):
        return self.serve_html("dashboard"), 200

    def signup(self):
        result = self.user_manager.create_account(request.form.get('email'), request.form.get('email'), request.form.get('password'))

        if result is True:
            user = User(request.form.get('email'), self.user_manager.get_display_name(request.form.get('email')))
            login_user(user)
            return redirect(url_for('dashboard'), 200)
        else:
            return {"error": "Account Creation Failed"}, 200

    def login(self):
        result = self.user_manager.validate_login(request.form.get('email'), request.form.get('password'))

        if result is True:
            user = User(request.form.get('email'), self.user_manager.get_display_name(request.form.get('email')))
            login_user(user)
            return redirect(url_for('dashboard'), 200)

        else:
            return {"error": "Account Login Failed"}, 200

    @staticmethod
    def serve_site_image(filename):
        full_static_image = "static/img/" + filename
        if os.path.exists(full_static_image):
            return send_file(full_static_image, mimetype='image/png', as_attachment=True), 200

        return {}, 404

    @staticmethod
    def get_css(filepath):
        full_static_image = "static/css/" + filepath
        if os.path.exists(full_static_image):
            return send_file(full_static_image, mimetype='text/css', as_attachment=True), 200

        return {}, 404

    def load_task(self, task_id):
        task_details = self.server.queue.get_stored_task(current_user.id, task_id)

        if task_details:
            return self.serve_html("task_viewer")
        return abort(403)

    def get_task_info(self, task_id):
        task_details = self.server.queue.get_stored_task(current_user.id, task_id)

        if task_details:
            return task_details, 200
        return abort(403)

    def get_heatmap(self, task_id):  # todo  - Make this check that it is shared OR you are the owner
        full_static_image = f"heatmaps/{task_id}.png"

        if os.path.exists(full_static_image):
            return send_file(full_static_image, mimetype='image/png', as_attachment=True), 200

        return {}, 404

    def set_share_status(self, task_id):
        result = self.server.queue.set_share_status(current_user.id, task_id, request.args.get('status'))

        if result is True:
            return {}, 200
        return abort(403)

    def start_threaded(self):
        threading.Thread(target=self.run, daemon=True).start()

    def run(self, debug=False):
        ip = "127.0.0.1"
        port = 5000

        self.app.run(ip, port, debug=debug)
