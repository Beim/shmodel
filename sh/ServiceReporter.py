from kazoo.client import KazooClient
from config.config_loader import config_loader
import json
import requests

config = config_loader.get_config()


class ServiceReporter:

    monitors = []

    def __init__(self, zkhost: str):
        zk = KazooClient(zkhost)
        zk.start()
        service_monitor_path = config['service_monitor']['zkpath']
        children = zk.get_children(service_monitor_path)
        for node in children:
            data, stat = zk.get("%s/%s" % (service_monitor_path, str(node)))
            data = json.loads(data.decode('utf-8'))
            self.monitors.append(data)
        print('monitors: ' + json.dumps(self.monitors))

    def report(self, data: dict):
        monitor_info = self.monitors[0]
        url = '%s://%s:%s/%s' % ('http', monitor_info['host'], monitor_info['port'], 'report')
        data_str = json.dumps(data, ensure_ascii=False).encode('utf-8')
        return requests.post(url, data=data_str)


if __name__ == '__main__':
    sr = ServiceReporter(zkhost=config['zkhost'])