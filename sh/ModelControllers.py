import os

from openke.config import Trainer, Tester
from openke.module.model import TransE, TransH, TransD
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
from config.config_loader import config_loader


class BaseModelController:

    def __init__(self, benchmark_dir: str, checkpoint_dir: str, use_gpu: bool = True):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            # shutil.rmtree(checkpoint_dir)
        self.checkpoint_path = '%s/%s.ckpt' % (checkpoint_dir, self.model_name)
        self.parameters_path = '%s/%s.param' % (checkpoint_dir, self.model_name)
        self.use_gpu = use_gpu

    def train(self) -> None:
        raise NotImplementedError

    def test(self) -> None:
        raise NotImplementedError


class TranseController(BaseModelController):

    model_name = 'transe'

    def __init__(self, benchmark_dir: str, checkpoint_dir: str, use_gpu: bool = True):
        super(TranseController, self).__init__(benchmark_dir, checkpoint_dir, use_gpu)

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

        self.transx = TransE(
            ent_tot = self.ent_tot,
            rel_tot = self.rel_tot,
            p_norm = 1,
            norm_flag = True)

        self.model = NegativeSampling(
            model = self.transx,
            loss = MarginLoss(margin = 5.0),
            batch_size = self.train_dataloader.get_batch_size()
        )

    def train(self) -> None:
        trainer = Trainer(model=self.model, data_loader=self.train_dataloader, train_times=100, alpha=1.0, use_gpu=self.use_gpu)
        trainer.run()
        self.transx.save_checkpoint(self.checkpoint_path)
        self.transx.save_parameters(self.parameters_path)

    def test(self) -> None:
        self.transx.load_checkpoint(self.checkpoint_path)
        tester = Tester(model = self.transx, data_loader = self.test_dataloader, use_gpu=self.use_gpu)
        tester.run_link_prediction(type_constrain = False)


class TranshController(BaseModelController):

    model_name = 'transh'

    def __init__(self, benchmark_dir: str, checkpoint_dir: str, use_gpu: bool = True):
        super(TranshController, self).__init__(benchmark_dir, checkpoint_dir, use_gpu)

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

        self.transx = TransH(
            ent_tot = self.ent_tot,
            rel_tot = self.rel_tot,
            p_norm = 1,
            norm_flag = True)

        self.model = NegativeSampling(
            model = self.transx,
            loss = MarginLoss(margin = 4.0),
            batch_size = self.train_dataloader.get_batch_size()
        )

    def train(self) -> None:
        trainer = Trainer(model=self.model, data_loader=self.train_dataloader, train_times=50, alpha=0.5, use_gpu=self.use_gpu)
        trainer.run()
        self.transx.save_checkpoint(self.checkpoint_path)
        self.transx.save_parameters(self.parameters_path)

    def test(self) -> None:
        self.transx.load_checkpoint(self.checkpoint_path)
        tester = Tester(model = self.transx, data_loader = self.test_dataloader, use_gpu=self.use_gpu)
        tester.run_link_prediction(type_constrain = False)


class TransdController(BaseModelController):

    model_name = 'transd'

    def __init__(self, benchmark_dir: str, checkpoint_dir: str, use_gpu: bool = True):
        super(TransdController, self).__init__(benchmark_dir, checkpoint_dir, use_gpu)

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

        self.transx = TransD(
            ent_tot = self.ent_tot,
            rel_tot = self.rel_tot,
            p_norm = 1,
            norm_flag = True)

        self.model = NegativeSampling(
            model = self.transx,
            loss = MarginLoss(margin = 4.0),
            batch_size = self.train_dataloader.get_batch_size()
        )

    def train(self) -> None:
        trainer = Trainer(model=self.model, data_loader=self.train_dataloader, train_times=100, alpha=1.0, use_gpu=self.use_gpu)
        trainer.run()
        self.transx.save_checkpoint(self.checkpoint_path)
        self.transx.save_parameters(self.parameters_path)

    def test(self) -> None:
        self.transx.load_checkpoint(self.checkpoint_path)
        tester = Tester(model = self.transx, data_loader = self.test_dataloader, use_gpu=self.use_gpu)
        tester.run_link_prediction(type_constrain = False)


model_controllers = {
    'transe': TranseController,
    'transh': TranshController,
    'transd': TransdController,
}


def model_constructor(model_name: str) -> BaseModelController:
    if model_name in model_controllers:
        return model_controllers[model_name]
    else:
        raise Exception("model %s not implemented" % model_name)


