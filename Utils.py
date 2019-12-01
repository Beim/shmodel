import requests
import json
import pymysql

from config.config_loader import config_loader


class RequestUtils:

    def __init__(self):
        server_config = config_loader.get_config()['server']
        self.protocol = server_config['protocol']
        self.host = server_config['host']
        self.port = server_config['port']

    def post(self, path, data, cookies=None):
        url = '%s://%s:%s/%s' % (self.protocol, self.host, self.port, path)
        data_str = json.dumps(data, ensure_ascii=False).encode('utf-8')
        res = requests.post(url, data=data_str, cookies=cookies)
        return res

    def get(self, path, params, cookies=None):
        url = '%s://%s:%s/%s' % (self.protocol, self.host, self.port, path)
        res = requests.get(url, params=params, cookies=cookies)
        return res


class MysqlUtils:

    def __init__(self):
        mysql_config = config_loader.get_config()['mysql']
        self.db = pymysql.connect(host=mysql_config['host'],
                                  port=mysql_config['port'],
                                  database=mysql_config['database'],
                                  user=mysql_config['username'],
                                  password=mysql_config['password'],
                                  cursorclass=pymysql.cursors.DictCursor)

    def execute(self, sql: str, args: list = None, expect_rows: int = 1) -> bool:
        with self.db.cursor() as cursor:
            affected_rows = cursor.execute(sql, args)
        self.db.commit()
        return affected_rows == expect_rows

    def query(self, sql: str, args: list = None) -> list:
        with self.db.cursor() as cursor:
            cursor.execute(sql, args)
            result = list(cursor.fetchall())
        return result


request_utils = RequestUtils()
mysql_utils = MysqlUtils()