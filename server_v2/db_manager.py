import os
import sqlite3


"""
table_data format:

[
    {
        "name": 'name',
        "args": [("name1", "type"), ...]
    },
    ...
]

"""


def create_db(path, table_data):
    db = sqlite3.connect(path)
    cursor = db.cursor()

    for table in table_data:
        cursor.execute(f"CREATE TABLE {table['name']} ({', '.join([
            f"{col} {datatype}" for col, datatype in table['args']
        ])});")

        db.commit()

    cursor.close()
    db.close()


def does_db_exist(path):
    return os.path.exists(path)


class DatabaseFailure(Exception):
    pass


class Cursor:
    def __init__(self, my_db):
        self.db = my_db.get_db()
        self.cursor = self.db.cursor()

    def execute(self, sql, params):
        self.cursor.execute(sql, params)

    def fetchall(self):
        return self.cursor.fetchall()

    def commit_to_db(self):
        self.db.commit()

    def return_with_data(self):
        results = self.fetchall()
        self.close()
        return results

    def close(self):
        self.cursor.close()
        self.db.close()


class MyDB:
    def __init__(self, path):
        self.path = path

    def get_db(self):
        if does_db_exist(self.path):
            return sqlite3.connect(self.path)
        raise DatabaseFailure("Database does not exist")

    def request(self, table, conditions, columns, param_data):
        """ WARNING table/data/conditions ARE NOT INJECTION SAFE, DO NOT USE USER INPUTS """
        cursor = Cursor(self)
        cursor.execute(f"SELECT {columns} FROM {table} WHERE {conditions}", param_data)
        return cursor.return_with_data()

    def insert(self, table, param_names, param_data):
        cursor = Cursor(self)
        cursor.execute(f"INSERT INTO {table} ({', '.join(param_names)}) VALUES ({", ".join(['?' for i in range(len(param_data))])})", param_data)
        cursor.close()

    def update(self, table, conditions, param_names, param_data):
        cursor = Cursor(self)
        cursor.execute(f"UPDATE {table} SET {' = ?, '.join(param_names)} = ? WHERE {conditions}", param_data)
        cursor.commit_to_db()
        cursor.close()

    def delete(self, table, param_names, param_data):
        cursor = Cursor(self)
        cursor.execute(f"DELETE FROM {table} WHERE {param_names}", param_data)
        cursor.close()

