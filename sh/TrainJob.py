import os
import numpy as np
import shutil
import json

from sh.ModelControllers import model_constructor
from Utils import mysql_utils
from config.config_loader import config_loader

curr_dir = os.path.split(os.path.abspath(__file__))[0]


class TrainJob:

    def __init__(self, triples: list, model_name: str, gspace_id: int, uuid: str, use_gpu: bool = True):
        self.triples = triples
        self.model_name = model_name
        self.gspace_id = gspace_id
        self.uuid = uuid
        self.use_gpu = use_gpu

        benchmarks = config_loader.get_config()['path']['benchmarks']
        checkpoint = config_loader.get_config()['path']['checkpoint']

        self.BENCHMARK_DIRPATH = '%s/../%s/%s' % (curr_dir, benchmarks, gspace_id)
        self.ENTITY2ID_PATH = '%s/entity2id.txt' % self.BENCHMARK_DIRPATH
        self.RELATION2ID_PATH = '%s/relation2id.txt' % self.BENCHMARK_DIRPATH
        self.TRAIN2ID_PATH = '%s/train2id.txt' % self.BENCHMARK_DIRPATH
        self.VALID2ID_PATH = '%s/valid2id.txt' % self.BENCHMARK_DIRPATH
        self.TEST2ID_PATH = '%s/test2id.txt' % self.BENCHMARK_DIRPATH
        self.CHECKPOINT_DIRPATH = '%s/../%s/gspace/%s' % (curr_dir, checkpoint, gspace_id)
        
        self.model_constructor = model_constructor(model_name)

    def run(self):
        """
        S1 准备数据
        S2 训练
        S3 测试
        S4 上传结果
        :return:
        """
        # TODO 解决0数据报错的bug
        """
        {'trainTriples': [], 'modelName': 'transe', 'gid': 4, 'uuid': '2f6963fc-59c9-4b2b-933c-1b9714c6120f'}
        train num 1, test num 1
        #train: 0, #valid: 0, #test: 0
        Input Files Path : /root/shmodel_copy/sh/../benchmarks/gspace/4/
        The toolkit is importing datasets.
        The total of relations is 0.
        The total of entities is 0.
        The total of train triples is 1.
        Input Files Path : /root/shmodel_copy/sh/../benchmarks/gspace/4/
        The total of test triples is 0.
        The total of valid triples is 0.
        bin/train_server.sh: line 2: 38132 Segmentation fault      PYTHONPATH=. python sh/TrainJobQueueReceiver.py
        """
        print('in train job')
        self._prepare_data(self.triples, self.gspace_id)
        model = self.model_constructor(self.BENCHMARK_DIRPATH, self.CHECKPOINT_DIRPATH, self.use_gpu)
        model.train()
        model.test()
        self._upload_param(model.parameters_path)
        print('finish trian job')
        return

    def _upload_param(self, param_path: str) -> None:
        # TODO 使用上传文件方式，传入mysql 的param 文件不能过大

        print('prepare upload param...')
        with open(param_path, 'r') as f:
            params = f.read()
        with open(self.ENTITY2ID_PATH, 'r') as f:
            entity2id = f.read()
        with open(self.RELATION2ID_PATH, 'r') as f:
            relation2id = f.read()

        # TODO 解决长时间未使用断开连接的bug
        # mysql_utils.execute('update gspacemodelparam set available=true, params=%s, entity2id=%s, relation2id=%s where gid=%s and modelname=%s',
        #                     [params, entity2id, relation2id, self.gspace_id, self.model_name])

        print('params len = %d' % len(params))
        mysql_utils.execute(
            'update gspacemodelparam set params=%s where gid=%s and modelname=%s',
            [params, self.gspace_id, self.model_name])
        print('upload_param gid[%d] model_name[%s]' % (self.gspace_id, self.model_name))

        mysql_utils.execute(
            'update gspacemodelparam set entity2id=%s where gid=%s and modelname=%s',
            [entity2id,  self.gspace_id, self.model_name])
        print('update entity2id')
        mysql_utils.execute(
            'update gspacemodelparam set relation2id=%s where gid=%s and modelname=%s',
            [relation2id, self.gspace_id, self.model_name])
        print('update relation2id')


        mysql_utils.execute(
            'update gspacemodelparam set available=true where gid=%s and modelname=%s',
            [self.gspace_id, self.model_name])
        print('update available=true')
        # request_path = 'embed/gspace/%d/model/%s' % (self.gspace_id, self.model_name)
        # data = {
        #     'uuid': self.uuid,
        #     'param': param
        # }
        # response = request_utils.post(request_path, data)
        # print('upload_param: ', response)

    def _prepare_data(self, triples: list, gspace_id: int):
        """
        准备训练、测试数据
        在benchmarks/gspace/<gspace_id> 目录下生成：
        entity2id.txt, relation2id.txt, train2id.txt, valid2id.txt, test2id.txt
        type_constraint.txt, 1-1.txt, 1-n.txt, n-1.txt, n-n.txt, test2id_all.txt
        :param triples: triples [[head, tail, relType], ...]
        :param gspace_id: 图空间id
        :return:
        """
        TRAIN_RATIO = 0.8
        TEST_RATIO = 0.1
        TRAIN_NUM = max(int(len(triples) * TRAIN_RATIO), 1)
        TEST_NUM = max(int(len(triples) * TEST_RATIO), 1)
        if os.path.exists(self.BENCHMARK_DIRPATH):
            shutil.rmtree(self.BENCHMARK_DIRPATH)
        os.makedirs(self.BENCHMARK_DIRPATH)

        entities = set()
        rels = set()
        for [head, tail, rel] in triples:
            entities.add(head)
            entities.add(tail)
            rels.add(rel)
        entities = list(entities)
        rels = list(rels)

        entity2idmap = {}
        rel2idmap = {}
        with open(self.ENTITY2ID_PATH, 'w') as f:
            f.write('%d\n' % len(entities))
            for idx in range(len(entities)):
                entity = entities[idx]
                entity2idmap[entity] = idx
                f.write('%s\t%d\n' % (entity, idx))
        del entities
        with open(self.RELATION2ID_PATH, 'w') as f:
            f.write('%d\n' % len(rels))
            for idx in range(len(rels)):
                rel = rels[idx]
                rel2idmap[rel] = idx
                f.write('%s\t%d\n' % (rel, idx))
        del rels

        for idx in range(len(triples)):
            [head, tail, rel] = triples[idx]
            triples[idx] = [
                entity2idmap[head],
                entity2idmap[tail],
                rel2idmap[rel]
            ]
        triples = np.random.permutation(triples)
        print('train num %d, test num %d' % (TRAIN_NUM, TEST_NUM))
        train_triples = triples[: TRAIN_NUM]
        test_triples = triples[TRAIN_NUM : TRAIN_NUM + TEST_NUM]
        valid_triples = triples[-TEST_NUM:]
        print('#train: %d, #valid: %d, #test: %d' % (len(train_triples), len(valid_triples), len(test_triples)))
        del triples

        with open(self.TRAIN2ID_PATH, 'w') as f:
            f.write('%d\n' % len(train_triples))
            for [headid, tailid, relid] in train_triples:
                f.write('%d %d %d\n' % (headid, tailid, relid))
        with open(self.VALID2ID_PATH, 'w') as f:
            f.write('%d\n' % len(valid_triples))
            for [headid, tailid, relid] in valid_triples:
                f.write('%d %d %d\n' % (headid, tailid, relid))
        with open(self.TEST2ID_PATH, 'w') as f:
            f.write('%d\n' % len(test_triples))
            for [headid, tailid, relid] in test_triples:
                f.write('%d %d %d\n' % (headid, tailid, relid))
        del train_triples, valid_triples, test_triples

        self._nn_prepare_data(self.BENCHMARK_DIRPATH)

    @staticmethod
    def _nn_prepare_data(benchmark_gspace_dir_path: str):
        """
        准备测试、验证数据
        已有train2id.txt, valid2id.txt, test2id.txt
        生成 type_constraint.txt, 1-1.txt, 1-n.txt, n-1.txt, n-n.txt, test2id_all.txt
        :param benchmark_gspace_dir_path: 存放上述文件的路径
        :return:
        """
        lef = {}
        rig = {}
        rellef = {}
        relrig = {}

        triple = open("%s/train2id.txt" % benchmark_gspace_dir_path, "r")
        valid = open("%s/valid2id.txt" % benchmark_gspace_dir_path, "r")
        test = open("%s/test2id.txt" % benchmark_gspace_dir_path, "r")

        tot = (int)(triple.readline())
        for i in range(tot):
            content = triple.readline()
            h, t, r = content.strip().split()
            if not (h, r) in lef:
                lef[(h, r)] = []
            if not (r, t) in rig:
                rig[(r, t)] = []
            lef[(h, r)].append(t)
            rig[(r, t)].append(h)
            if not r in rellef:
                rellef[r] = {}
            if not r in relrig:
                relrig[r] = {}
            rellef[r][h] = 1
            relrig[r][t] = 1

        tot = (int)(valid.readline())
        for i in range(tot):
            content = valid.readline()
            h, t, r = content.strip().split()
            if not (h, r) in lef:
                lef[(h, r)] = []
            if not (r, t) in rig:
                rig[(r, t)] = []
            lef[(h, r)].append(t)
            rig[(r, t)].append(h)
            if not r in rellef:
                rellef[r] = {}
            if not r in relrig:
                relrig[r] = {}
            rellef[r][h] = 1
            relrig[r][t] = 1

        tot = (int)(test.readline())
        for i in range(tot):
            content = test.readline()
            h, t, r = content.strip().split()
            if not (h, r) in lef:
                lef[(h, r)] = []
            if not (r, t) in rig:
                rig[(r, t)] = []
            lef[(h, r)].append(t)
            rig[(r, t)].append(h)
            if not r in rellef:
                rellef[r] = {}
            if not r in relrig:
                relrig[r] = {}
            rellef[r][h] = 1
            relrig[r][t] = 1

        test.close()
        valid.close()
        triple.close()

        f = open("%s/type_constrain.txt" % benchmark_gspace_dir_path, "w")
        f.write("%d\n" % (len(rellef)))
        for i in rellef:
            f.write("%s\t%d" % (i, len(rellef[i])))
            for j in rellef[i]:
                f.write("\t%s" % (j))
            f.write("\n")
            f.write("%s\t%d" % (i, len(relrig[i])))
            for j in relrig[i]:
                f.write("\t%s" % (j))
            f.write("\n")
        f.close()

        rellef = {}
        totlef = {}
        relrig = {}
        totrig = {}
        # lef: (h, r)
        # rig: (r, t)
        for i in lef:
            if not i[1] in rellef:
                rellef[i[1]] = 0
                totlef[i[1]] = 0
            rellef[i[1]] += len(lef[i])
            totlef[i[1]] += 1.0

        for i in rig:
            if not i[0] in relrig:
                relrig[i[0]] = 0
                totrig[i[0]] = 0
            relrig[i[0]] += len(rig[i])
            totrig[i[0]] += 1.0

        s11 = 0
        s1n = 0
        sn1 = 0
        snn = 0
        f = open("%s/test2id.txt" % benchmark_gspace_dir_path, "r")
        tot = (int)(f.readline())
        for i in range(tot):
            content = f.readline()
            h, t, r = content.strip().split()
            rign = rellef[r] / totlef[r]
            lefn = relrig[r] / totrig[r]
            if (rign < 1.5 and lefn < 1.5):
                s11 += 1
            if (rign >= 1.5 and lefn < 1.5):
                s1n += 1
            if (rign < 1.5 and lefn >= 1.5):
                sn1 += 1
            if (rign >= 1.5 and lefn >= 1.5):
                snn += 1
        f.close()

        f = open("%s/test2id.txt" % benchmark_gspace_dir_path, "r")
        f11 = open("%s/1-1.txt" % benchmark_gspace_dir_path, "w")
        f1n = open("%s/1-n.txt" % benchmark_gspace_dir_path, "w")
        fn1 = open("%s/n-1.txt" % benchmark_gspace_dir_path, "w")
        fnn = open("%s/n-n.txt" % benchmark_gspace_dir_path, "w")
        fall = open("%s/test2id_all.txt" % benchmark_gspace_dir_path, "w")
        tot = (int)(f.readline())
        fall.write("%d\n" % (tot))
        f11.write("%d\n" % (s11))
        f1n.write("%d\n" % (s1n))
        fn1.write("%d\n" % (sn1))
        fnn.write("%d\n" % (snn))
        for i in range(tot):
            content = f.readline()
            h, t, r = content.strip().split()
            rign = rellef[r] / totlef[r]
            lefn = relrig[r] / totrig[r]
            if (rign < 1.5 and lefn < 1.5):
                f11.write(content)
                fall.write("0" + "\t" + content)
            if (rign >= 1.5 and lefn < 1.5):
                f1n.write(content)
                fall.write("1" + "\t" + content)
            if (rign < 1.5 and lefn >= 1.5):
                fn1.write(content)
                fall.write("2" + "\t" + content)
            if (rign >= 1.5 and lefn >= 1.5):
                fnn.write(content)
                fall.write("3" + "\t" + content)
        fall.close()
        f.close()
        f11.close()
        f1n.close()
        fn1.close()
        fnn.close()

if __name__ == '__main__':
    l = []
    for i in range(1000):
        l.append([i, i + 1, 'like%d' % (i % 20)])
    job = TrainJob(l, 'transe', 1, 'f35a7da8-49e4-43ec-aa75-e67e6d935f69')
    job.run()

