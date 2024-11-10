from db_manager import MyDB, create_db, does_db_exist

import bcrypt


class UserHandler:
    def __init__(self):
        self.user_db = init_user_db()

    def create_account(self, email, display_name, password):
        result = self.user_db.request(
            "login_data",
            "email == ?",
            "*",
            (email,)
        )

        if len(result) > 0:
            return False

        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        self.user_db.insert(
            "login_data",
            ("email", "display_name", "password"),
            (email, display_name, password_hash)
        )

        return True

    def validate_login(self, email, password):
        results = self.user_db.request(
            "login_data",
            "email == ?",
            "password",
            (email,)
        )

        if len(results) == 0:
            return False

        return bcrypt.checkpw(password.encode('utf-8'), results[0][0])

    def get_display_name(self, email):
        results = self.user_db.request(
            "login_data",
            "email == ?",
            "display_name",
            (email,)
        )

        if len(results) > 0:
            return results[0]

        return email


def init_user_db():
    if not does_db_exist("users.db"):
        create_db("users.db", [
            {
                "name": "login_data",
                "args": [
                    ("email", "TEXT PRIMARY KEY"),
                    ("display_name", "TEXT"),
                    ("password", "BLOB")
                ]
            }
        ])

    return MyDB("users.db")
