import pika
import json
from pika.adapters.blocking_connection import BlockingChannel, BlockingConnection

from sh.TrainJob import TrainJob
from config.config_loader import config_loader


class TrainJobQueueReceiver:
    connection = None
    channel = None

    def __init__(self, host: str, port: str, username: str, password: str,
                 queue_name: str, durable: bool, auto_ack: bool, prefetch_count: int):
        self.connection, self.channel = self.create_connection(
            host, port, username, password, queue_name, durable, auto_ack, prefetch_count)
        self.channel.start_consuming()

    def create_connection(self, host: str, port: str, username: str, password: str,
                          queue_name: str, durable: bool, auto_ack: bool, prefetch_count: bool) -> (
    BlockingConnection, BlockingChannel):
        """
        建立连接
        :param host:
        :param port:
        :param username:
        :param password:
        :param queue_name:
        :param durable: 持久化
        :param auto_ack: 自动确认
        :param prefetch_count: 每个worker 最多接受的消息数量
        :return: connection, channel
        """
        credential = pika.PlainCredentials(username, password)
        connection_params = pika.ConnectionParameters(host=host, credentials=credential, port=port)
        connection = pika.BlockingConnection(connection_params)
        channel = connection.channel()
        channel.queue_declare(queue=queue_name, durable=durable)
        channel.basic_qos(prefetch_count=prefetch_count)
        channel.basic_consume(queue=queue_name, on_message_callback=self.receive_callback, auto_ack=auto_ack)
        print('[pika] connection established, waiting for messages...')
        return connection, channel

    def receive_callback(self, ch: BlockingChannel, method, properties, body):
        """
        接受消息的回调
        :param ch:
        :param method:
        :param properties:
        :param body:
        :return:
        """
        # {'trainTriples': [[2, 0, 'like'], [2, 1, 'like']], 'modelName': 'TransE', 'gid': 1}
        info = json.loads(body, encoding='utf-8')
        print(info)
        train_triples = info['trainTriples']
        model_name = info['modelName']
        gspace_id = info['gid']
        uuid = info['uuid']
        try:
            TrainJob(train_triples, model_name, gspace_id, uuid, use_gpu=config_loader.get_config()['gpu']).run()
            ch.basic_ack(delivery_tag=method.delivery_tag)
            print('ack %s %d' % (model_name, gspace_id))
        except Exception as e:
            print(e)
            ch.basic_nack(delivery_tag=method.delivery_tag)
            print('nack %s %d' % (model_name, gspace_id))

if __name__ == '__main__':
    rabbitmq_config = config_loader.get_config()['rabbitmq']
    receiver = TrainJobQueueReceiver(rabbitmq_config['host'],
                                     rabbitmq_config['port'],
                                     rabbitmq_config['username'],
                                     rabbitmq_config['password'],
                                     rabbitmq_config['queue_name'],
                                     rabbitmq_config['durable'],
                                     rabbitmq_config['auto_ack'],
                                     rabbitmq_config['prefetch_count'])