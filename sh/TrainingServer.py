from kazoo.client import KazooClient
from config.config_loader import config_loader
import json
import time
from concurrent import futures
from flask import Flask, request
from sh.TrainJob import TrainJob
from sh.ServiceReporter import ServiceReporter

app = Flask(__name__)
config = config_loader.get_config()
executor = futures.ThreadPoolExecutor(max_workers=1)


class TrainJobRegister:

    zk = None  # zookeeper client
    node_path = None  # the real path of the service node

    def __init__(self, host: str):
        self.zk = KazooClient(host)
        self.zk.start()

    def register(self):
        train_service_path = config['train_service_path']
        zk = self.zk
        zk.ensure_path(train_service_path)
        path = train_service_path + "/service"
        data = {
            'host': config['host'],
            'port': config['port'],
            'gpu': config['gpu'],
            'available': True
        }
        self.node_path = zk.create(path, str.encode(json.dumps(data)), ephemeral=True, sequence=True)

    def set_availability(self, state: bool):
        zk = self.zk
        data = json.loads(bytes.decode(zk.get(self.node_path)[0]))
        data['available'] = state
        zk.set(self.node_path, str.encode(json.dumps(data)))


tjr = TrainJobRegister(host=config['zkhost'])


@app.route("/train", methods=['POST'])
def train_job_run():
    def run(args: dict):
        train_triples = args['trainTriples']
        model_name = args['modelName']
        gspace_id = args['gid']
        uuid = args['uuid']
        uid = args['uid']
        tjr.set_availability(False)
        start_time = time.time()
        # TrainJob(train_triples, model_name, gspace_id, uuid, use_gpu=config_loader.get_config()['gpu']).run()
        time.sleep(10)
        end_time = time.time()
        tjr.set_availability(True)

        sr = ServiceReporter(zkhost=config['zkhost'])
        report_data = {
            'uid': uid,
            'service': config['train_service_path'],
            'timestamp': str(start_time),
            'duration': str(round(end_time - start_time, 2)),
            'info': json.dumps({
                'gpu': config['gpu']
            })
        }
        sr.report(report_data)

    args = json.loads(request.data)
    print({
        'modelName': args['modelName'],
        'gid': args['gid'],
        'trainTriplesLen': len(args['trainTriples'])
    })
    executor.submit(run, args)
    return json.dumps({'succ': True})


if __name__ == '__main__':
    tjr.register()
    app.run('0.0.0.0', config['port'])





