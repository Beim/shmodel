import numpy as np
import torch
import json

from openke.module.model import Model, TransE


class ModelPredictor:

    def __init__(self, model_name: str, paramters_path: str, entity2id_path: str, relation2id_path, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.entity2id_map, self.id2entity_map, self.relation2id_map, self.id2relation_map \
            = self._get_ent_rel_map(entity2id_path, relation2id_path)
        self.ent_tot = len(self.entity2id_map)
        self.rel_tot = len(self.relation2id_map)
        self.ent_embeddings, self.rel_embeddings = self._get_ent_rel_embedding(paramters_path)

        constructor = self._get_model_constructor(model_name)
        self.model = constructor(
            ent_tot=self.ent_tot,
            rel_tot=self.rel_tot,
            dim=200,
            p_norm=1,
            norm_flag=True)
        if use_gpu:
            self.model.cuda()
        self.model.load_parameters(paramters_path)

    def predict_head_entity(self, t: str, r: str, k: int) -> list:
        """
        This method predicts the top k head entities given tail entity and relation.
        :param t: tail entity name
        :param r: relation type
        :param k: top k head entities
        :return: k possible entity names
        """
        t = self.entity2id_map[t]
        r = self.relation2id_map[r]
        res = self._predict_head_entity(t, r, k)
        for idx in range(len(res)):
            res[idx] = self.id2entity_map[res[idx]]
        return res

    def predict_tail_entity(self, h: str, r: str, k: int) -> list:
        """
        This method predicts the top k tail entities given head entity and relation.
        :param h: head entity name
        :param r: relation type
        :param k: top k tail entities
        :return: k possible entity names
        """
        h = self.entity2id_map[h]
        r = self.relation2id_map[r]
        res = self._predict_tail_entity(h, r, k)
        for idx in range(len(res)):
            res[idx] = self.id2entity_map[res[idx]]
        return res

    def predict_relation(self, h: str, t: str, k: int) -> list:
        """
        This methods predict the relation id given head entity and tail entity.
        :param h: head entity name
        :param t: tail entity name
        :param k: top k relations
        :return: k possible relation types
        """
        h = self.entity2id_map[h]
        t = self.entity2id_map[t]
        res = self._predict_relation(h, t, k)
        for idx in range(len(res)):
            res[idx] = self.id2relation_map[res[idx]]
        return res

    def predict_triple(self, h: str, t: str, r: str, thresh: float) -> bool:
        """
        This method tells you whether the given triple (h, t, r) is correct of wrong
        :param h: head entity name
        :param t: tail entity name
        :param r: relation type
        :param thresh: threshold for the triple
        :return:
        """
        h = self.entity2id_map[h]
        t = self.entity2id_map[t]
        r = self.relation2id_map[r]
        return self._predict_triple(h, t, r, thresh)

    def get_ent_embedding(self, ent: str):
        return self.ent_embeddings[self.entity2id_map[ent]]

    def get_rel_embedding(self, rel: str):
        return self.rel_embeddings[self.relation2id_map[rel]]

    def _predict_head_entity(self, t: int, r: int, k: int) -> list:
        """
        This method predicts the top k head entities given tail entity and relation.
        :param t: tail entity id
        :param r: relation id
        :param k: top k head entities
        :return: k possible entity ids
        """
        test_h = self._to_cuda(torch.LongTensor(range(self.ent_tot)), self.use_gpu)
        test_t = self._to_cuda(torch.LongTensor([t] * self.ent_tot), self.use_gpu)
        test_r = self._to_cuda(torch.LongTensor([r] * self.ent_tot), self.use_gpu)
        res = self._predict(test_h, test_t, test_r).reshape(-1).argsort()[:k]
        return list(res)

    def _predict_tail_entity(self, h: int, r: int, k: int) -> list:
        """
        This method predicts the top k tail entities given head entity and relation.
        :param h: head entity id
        :param r: relation id
        :param k: top k tail entities
        :return: k possible entity ids
        """
        test_h = self._to_cuda(torch.LongTensor([h] * self.ent_tot), self.use_gpu)
        test_t = self._to_cuda(torch.LongTensor(range(self.ent_tot)), self.use_gpu)
        test_r = self._to_cuda(torch.LongTensor([r] * self.ent_tot), self.use_gpu)
        res = self._predict(test_h, test_t, test_r).reshape(-1).argsort()[:k]
        return list(res)

    def _predict_relation(self, h: int, t: int, k: int) -> list:
        """
        This methods predict the relation id given head entity and tail entity.
        :param h: head entity id
        :param t: tail entity id
        :param k: top k relations
        :return: k possible relation ids
        """
        test_h = self._to_cuda(torch.LongTensor([h] * self.rel_tot), self.use_gpu)
        test_t = self._to_cuda(torch.LongTensor([t] * self.rel_tot), self.use_gpu)
        test_r = self._to_cuda(torch.LongTensor(range(self.rel_tot)), self.use_gpu)
        res = self._predict(test_h, test_t, test_r).reshape(-1).argsort()[:k]
        return list(res)

    def _predict_triple(self, h: int, t: int, r: int, thresh: float) -> bool:
        """
        This method tells you whether the given triple (h, t, r) is correct of wrong
        :param h: head entity id
        :param t: tail entity id
        :param r: relation id
        :param thresh: threshold for the triple
        :return:
        """
        test_h = self._to_cuda(torch.LongTensor([h]), self.use_gpu)
        test_t = self._to_cuda(torch.LongTensor([t]), self.use_gpu)
        test_r = self._to_cuda(torch.LongTensor([r]), self.use_gpu)
        res = self._predict(test_h, test_t, test_r)[0]
        return res < thresh

    def _to_cuda(self, t: torch.Tensor, use_gpu: bool) -> torch.Tensor:
        if use_gpu:
            return t.cuda()
        else:
            return t

    def _predict(self, batch_h: torch.LongTensor, batch_t: torch.LongTensor, batch_r: torch.LongTensor, mode: str = None) -> np.ndarray:
        """
        :param batch_h: head entity ids
        :param batch_t: tail entity ids
        :param batch_r: relation ids
        :param mode:
        :return: scores
        """
        data = {
            'batch_h': batch_h,
            'batch_t': batch_t,
            'batch_r': batch_r,
            'mode': mode
        }
        return self.model.predict(data)

    def _get_ent_rel_map(self, entity2id_path: str, relation2id_path: str) -> (dict, dict, dict, dict):
        entity2id_map = {}
        id2entity_map = {}
        relation2id_map = {}
        id2relation_map = {}
        with open(entity2id_path, 'r') as f:
            rows = f.read().split('\n')[1:]
            for row in rows:
                items = row.split('\t')
                if len(items) != 2:
                    continue
                entity = str(items[0])
                id = int(items[1])
                entity2id_map[entity] = id
                id2entity_map[id] = entity
        with open(relation2id_path, 'r') as f:
            rows = f.read().split('\n')[1:]
            for row in rows:
                items = row.split('\t')
                if len(items) != 2:
                    continue
                relation = str(items[0])
                id = int(items[1])
                relation2id_map[relation] = id
                id2relation_map[id] = relation
        return entity2id_map, id2entity_map, relation2id_map, id2relation_map

    def _get_model_constructor(self, model_name: str) -> Model:
        models = {
            'transe': TransE,
        }
        if model_name in models:
            return models[model_name]
        else:
            raise NotImplementedError

    def _get_ent_rel_embedding(self, paramters_path: str):
        with open(paramters_path, 'r') as f:
            params = json.load(f)
        ent_embedding = params['ent_embeddings.weight']
        rel_embedding = params['rel_embeddings.weight']
        return ent_embedding, rel_embedding

if __name__ == '__main__':
    import os

    curr_dir = os.path.split(os.path.abspath(__file__))[0]
    predictor = ModelPredictor('transe',
                               '%s/../checkpoint/gspace/1/transe.param' % curr_dir,
                               '%s/../benchmarks/gspace/1/entity2id.txt' % curr_dir,
                               '%s/../benchmarks/gspace/1/relation2id.txt' % curr_dir,
                               False)
    res = predictor.predict_head_entity(440, 13, 10)
    # res = predictor.predict_tail_entity(439, 13, 10)
    # res = predictor.predict_triple(439, 440, 13, 10)
    print(res)