import os
import json

from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

from Utils import RequestUtils

class BaseModelController:

    def __init__(self, benchmark_dir: str, checkpoint_dir: str):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            # shutil.rmtree(checkpoint_dir)
        self.checkpoint_path = '%s/%s.ckpt' % (checkpoint_dir, self.model_name)
        self.parameters_path = '%s/%s.param' % (checkpoint_dir, self.model_name)

    def train(self) -> None:
        raise NotImplementedError

    def test(self) -> None:
        raise NotImplementedError


class TranseController(BaseModelController):

    model_name = 'transe'

    def __init__(self, benchmark_dir: str, checkpoint_dir: str):
        super(TranseController, self).__init__(benchmark_dir, checkpoint_dir)

        self.train_dataloader = TrainDataLoader(
            in_path=benchmark_dir + '/',
            nbatches=100,
            threads=8,
            sampling_mode="normal",
            bern_flag=1,
            filter_flag=1,
            neg_ent=25,
            neg_rel=0)

        self.test_dataloader = TestDataLoader(benchmark_dir + '/', "link")

        self.ent_tot = self.train_dataloader.get_ent_tot()
        self.rel_tot = self.train_dataloader.get_rel_tot()

        self.transe = TransE(
            ent_tot = self.ent_tot,
            rel_tot = self.rel_tot,
            dim = 200,
            p_norm = 1,
            norm_flag = True)

        self.model = NegativeSampling(
            model = self.transe,
            loss = MarginLoss(margin = 5.0),
            batch_size = self.train_dataloader.get_batch_size()
        )

    def train(self) -> None:
        trainer = Trainer(model=self.model, data_loader=self.train_dataloader, train_times=20, alpha=1.0, use_gpu=True)
        trainer.run()
        self.transe.save_checkpoint(self.checkpoint_path)
        self.transe.save_parameters(self.parameters_path)

    def test(self) -> None:
        self.transe.load_checkpoint(self.checkpoint_path)
        tester = Tester(model = self.transe, data_loader = self.test_dataloader, use_gpu = True)
        tester.run_link_prediction(type_constrain = False)


model_controllers = {
    'transe': TranseController
}


def model_constructor(model_name: str) -> BaseModelController:
    if model_name in model_controllers:
        return model_controllers[model_name]
    else:
        raise Exception("model %s not implemented" % model_name)


