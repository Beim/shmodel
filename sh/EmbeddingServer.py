from concurrent import futures
from google.protobuf import wrappers_pb2 as wrappers
import grpc
import time

from protos import embedding_pb2_grpc as embedding_pb2_grpc, embedding_pb2 as embedding_pb2
from config.config_loader import config_loader
from sh.ModelLoader import ModelLoader

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class EmbeddingServicer(embedding_pb2_grpc.GraphEmbeddingServiceServicer):

    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader

    def predictHead(self, request: embedding_pb2.PredictHeadRequest, context) -> embedding_pb2.PredictPartResponse:
        print('[%s] predictHead\n' % time.time(), request)
        model = self.model_loader.get_model(request.gid, request.modelName)
        res = model.predict_head_entity(request.tail, request.relation, request.k)
        return embedding_pb2.PredictPartResponse(val=res)

    def predictTail(self, request: embedding_pb2.PredictTailRequest, context) -> embedding_pb2.PredictPartResponse:
        print('[%s] predictTail\n' % time.time(), request)
        model = self.model_loader.get_model(request.gid, request.modelName)
        res = model.predict_tail_entity(request.head, request.relation, request.k)
        return embedding_pb2.PredictPartResponse(val=res)

    def predictRelation(self, request: embedding_pb2.PredictRelationRequest, context) -> embedding_pb2.PredictPartResponse:
        print('[%s] predictRelation\n' % time.time(), request)
        model = self.model_loader.get_model(request.gid, request.modelName)
        res = model.predict_relation(request.head, request.tail, request.k)
        return embedding_pb2.PredictPartResponse(val=res)

    def predictTriple(self, request: embedding_pb2.PredictTripleRequest, context) -> wrappers.BoolValue:
        print('[%s] predictTriple\n' % time.time(), request)
        model = self.model_loader.get_model(request.gid, request.modelName)
        res = model.predict_triple(request.head, request.tail, request.relation, request.thresh)
        return wrappers.BoolValue(value=res)

    def getEntityEmbedding(self, request: embedding_pb2.GetEmbeddingRequest, context) -> embedding_pb2.GetEmbeddingResponse:
        print('[%s] getEntityEmbedding\n' % time.time(), request)
        model = self.model_loader.get_model(request.gid, request.modelName)
        res = model.get_ent_embedding(request.val)
        return embedding_pb2.GetEmbeddingResponse(val=res)

    def getRelationEmbedding(self, request: embedding_pb2.GetEmbeddingRequest, context) -> embedding_pb2.GetEmbeddingResponse:
        print('[%s] getRelationEmbedding\n' % time.time(), request)
        model = self.model_loader.get_model(request.gid, request.modelName)
        res = model.get_rel_embedding(request.val)
        return embedding_pb2.GetEmbeddingResponse(val=res)


def serve(model_loader: ModelLoader):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    embedding_pb2_grpc.add_GraphEmbeddingServiceServicer_to_server(
        EmbeddingServicer(model_loader), server
    )
    port = config_loader.get_config()['grpc']['port']
    server.add_insecure_port('[::]:%d' % port)
    server.start()
    print('start serve on [::]:%d' % port)
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        print('stop serve...')
        server.stop(0)


def update_model(model_loader: ModelLoader):
    try:
        while True:
            time.sleep(config_loader.get_config()['update_interval'])
            if model_loader.check_update():
                print('[%s] updating model...' % time.time())
                model_loader.model_map = model_loader.load()
    except KeyboardInterrupt:
        print('stop udpate model')


if __name__ == '__main__':
    model_loader = ModelLoader()
    executor = futures.ThreadPoolExecutor(max_workers=2)
    executor.submit(update_model, model_loader)
    executor.submit(serve, model_loader)
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        print('stop main thread...')


