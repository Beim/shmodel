import os

from config.config_loader import config_loader
from Utils import mysql_utils
from sh.ModelPredictors import ModelPredictor

curr_dir = os.path.split(os.path.abspath(__file__))[0]


class ModelLoader:

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.models_dir = '%s/../%s' % (curr_dir, config_loader.get_config()['path']['trainedmodels'])
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        self.model_map = self.load()


    def get_model(self, gid: int, model_name: str) -> ModelPredictor:
        return self.model_map[(str(gid), model_name)]

    def check_update(self) -> bool:
        """
        检查模型是否更新
        :return:
        """
        local_embed_info = self._get_embed_infos()
        remote_embed_info = self._fetch_embed_infos()
        to_download_embed = remote_embed_info - local_embed_info
        return len(to_download_embed) > 0

    def load(self) -> dict:
        """
        {
            gspaceId: Long,
            modelName: String,
            lastModified: int,
            modelRelativePath: String
        }
        返回 {
            (<gid>, <model_name>): <ModelPredictor>,
            ...
        }
        本地存储embed param 的规则为 {modelId}_{modelName}_{lastModified}.param
        S1 找到本地所有embed param 信息
        S2 获取server embed 信息
        S3 diff 记录server 上有，本地没有的embed param
        S5 下载缺少的embed param
        # S6 删除无用的embed param
        S7 load predictor
        :return:
        """
        local_embed_info = self._get_embed_infos()
        remote_embed_info = self._fetch_embed_infos()
        to_download_embed = list(remote_embed_info - local_embed_info)
        # to_delete_embed = list(local_embed_info - remote_embed_info)
        self._download_embed(to_download_embed)
        load_map = {}
        for (gid, modelname, updated) in remote_embed_info:
            paramters_path = '%s/%s' % (self.models_dir, self._make_param_file_name(gid, modelname, updated))
            entity2id_path = '%s/%s' % (self.models_dir, self._make_entity2id_file_name(gid, modelname, updated))
            relation2id_path = '%s/%s' % (self.models_dir, self._make_relation2id_file_name(gid, modelname, updated))
            load_map[(gid, modelname)] = ModelPredictor(modelname,
                                                        paramters_path,
                                                        entity2id_path,
                                                        relation2id_path,
                                                        self.use_gpu)
        return load_map


    def _download_embed(self, to_download_embed: list) -> None:
        """
        下载模型参数
        保存的文件格式为
            param: <gid>_<modelname>_<updated>.param
            entity2id: <gid>_<modelname>_<updated>.entity2id.txt
            relation2id: <gid>_<modelname>_<updated>.relation2id.txt
        :param to_download_embed:
        :return: None
        """
        for (gid, modelname, updated) in to_download_embed:
            response = mysql_utils.query('select params, entity2id, relation2id from gspacemodelparam where gid=%s and modelname=%s',
                                         [gid, modelname])[0]
            paramters_path = '%s/%s' % (self.models_dir, self._make_param_file_name(gid, modelname, updated))
            entity2id_path = '%s/%s' % (self.models_dir, self._make_entity2id_file_name(gid, modelname, updated))
            relation2id_path = '%s/%s' % (self.models_dir, self._make_relation2id_file_name(gid, modelname, updated))
            if response['params'] != None and response['entity2id'] != None and response['relation2id'] != None:
                with open(paramters_path, 'w') as f:
                    f.write(response['params'])
                with open(entity2id_path, 'w') as f:
                    f.write(response['entity2id'])
                with open(relation2id_path, 'w') as f:
                    f.write(response['relation2id'])

    def _get_embed_infos(self) -> set:
        """
        遍历本地embed 目录，获取embed param 信息
        ((<gid>, <modelname>, <updated>), ...)
        :return:
        """
        result = set()
        files = os.listdir(self.models_dir)
        for file in files:
            result.add(self._parse_param_file_name(file))
        return result

    def _fetch_embed_infos(self) -> set:
        """
        获取远程embed 信息
        ((<gid>, <modelname>, <updated>), ...)
        :return:
        """
        response = mysql_utils.query('select gid, modelname, updated from gspacemodelparam where available=true')
        result = set()
        for item in response:
            result.add( (str(item['gid']), item['modelname'], str(int(item['updated'].timestamp()))) )
        return result

    def _parse_param_file_name(self, filename: str) -> (str, str, int):
        [gid, model_name, updated] = filename.split('.')[0].split('_')
        return gid, model_name, updated

    def _make_param_file_name(self, gid: str, model_name: str, updated: int) -> str:
        return '%s_%s_%s.param' % (gid, model_name, updated)

    def _make_entity2id_file_name(self, gid: str, model_name: str, updated: int) -> str:
        return '%s_%s_%s.entity2id.txt' % (gid, model_name, updated)

    def _make_relation2id_file_name(self, gid: str, model_name: str, updated: int) -> str:
        return '%s_%s_%s.relation2id.txt' % (gid, model_name, updated)

