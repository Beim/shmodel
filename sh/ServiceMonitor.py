from flask import Flask, request
from kazoo.client import KazooClient
import json
from config.config_loader import config_loader
from Utils import mysql_utils

app = Flask(__name__)
config = config_loader.get_config()


class ServiceMonitor:

    def __init__(self, zkhost: str):
        zk = KazooClient(zkhost)
        zk.start()
        service_monitor_path = config['service_monitor']['zkpath']
        zk.ensure_path(service_monitor_path)
        data = {
            'host': config['service_monitor']['host'],
            'port': config['service_monitor']['port'],
        }
        zk.create(service_monitor_path + "/service", str.encode(json.dumps(data)), ephemeral=True, sequence=True)

    def record(self, uid: int, service: str, timestamp: str, duration: str, info: str):
        """
        记录调用信息
        :param uid: 用户id
        :param service: 服务名
        :param timestamp: 调用时间戳 (s)
        :param duration: 调用时长 (s)
        :return:
        """
        sql = 'insert into servicemonitorlog (uid, service, timestamp, duration, info) values (%s, %s, %s, %s, %s)'
        mysql_utils.execute(sql, [uid, service, timestamp, duration, info])

    def query(self, uid: int):
        """
        查询调用信息
        :param uid: 用户id
        :return:
        """
        sql = 'select * from servicemonitorlog where uid=%s'
        response = mysql_utils.query(sql, [uid])
        result = []
        for item in response:
            data = {
                'uid': item['uid'],
                'service': item['service'],
                'timestamp': item['timestamp'],
                'duration': item['duration'],
                'info': item['info']
            }
            result.append(data)
        return result

sm = ServiceMonitor(zkhost=config['zkhost'])


@app.route("/report", methods=['POST'])
def report():
    """
    报告调用信息
    body: {
        uid(long),
        service(str),
        timestamp(str),
        duration(str)
    }
    :return:
    """
    args = json.loads(request.data)
    sm.record(args['uid'], args['service'], args['timestamp'], args['duration'], args['info'])
    return json.dumps({'succ': True})


@app.route("/query", methods=['POST'])
def query():
    """
    查询调用信息
    body: {
      uid(long),
    }
    :return:
    """
    args = json.loads(request.data)
    result = sm.query(args['uid'])
    return json.dumps({'succ': True, 'data': result})


if __name__ == '__main__':
    app.run('0.0.0.0', config['service_monitor']['port'])