import os
import torch
import sys
sys.path.append("/workspace/4D-Diff-RNA_test_1")

import hydra
from omegaconf import DictConfig, OmegaConf

import GPUtil
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from Model.folding_module import FlowModule
from data.pdb_na_datamodule_base import PDBNABaseDataModule
import logging
from pytorch_lightning.utilities.rank_zero import rank_zero_only

# 경로 설정
TEST_DATA_PATH = "./Training_data/"
RNA_METADATA_CSV = "/workspace/4D-Diff-RNA_test_1/Training_data/rna_metadata_debug.csv"

def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger

log = get_pylogger(__name__)
torch.set_float32_matmul_precision('high')


class NodeFeaturesConfig:
    def __init__(self):
        self.c_s = 256
        self.c_pos_emb = 128
        self.c_timestep_emb = 128
        self.embed_diffuse_mask = False
        self.max_num_res = 2000
        self.timestep_int = 1000
        self.Node_param = "/workspace/4D-Diff-RNA_test_1/node_embedder_param.pickle"

class EdgeFeaturesConfig:
    def __init__(self):
        self.single_bias_transition_n = 2
        self.c_s = 256
        self.c_p = 128
        self.relpos_k = 64
        self.use_rbf = True
        self.num_rbf = 32
        self.feat_dim = 64
        self.num_bins = 22
        self.self_condition = True
        self.Edge_param = "/workspace/4D-Diff-RNA_test_1/edge_embedder_param.pickle"

class IPAConfig:
    def __init__(self):
        self.c_s = 256
        self.c_z = 128
        self.c_hidden = 128
        self.no_heads = 4
        self.no_qk_points = 4
        self.no_v_points = 6
        self.seq_tfmr_num_heads = 2
        self.seq_tfmr_num_layers = 1
        self.num_blocks = 3

class ModelConfig:
    def __init__(self):
        self.node_embed_size = 256
        self.edge_embed_size = 128
        self.symmetric = False
        self.node_features = NodeFeaturesConfig()
        self.edge_features = EdgeFeaturesConfig()
        self.ipa = IPAConfig()

class CheckpointerConfig:
    def __init__(self):
        self.dirpath = "./checkpoints"
        self.filename = "test_model"

class SamplingConfig:
    def __init__(self):
        self.num_timesteps = 50

class rot_config:
    def __init__(self):
        self.train_schedule = "linear"
        self.sample_schedule = "exp"
        self.exp_rate = 10

class trans_config:
    def __init__(self):
        self.train_schedule = "linear"
        self.sample_schedule = "linear"

class InterpolantConfig:
    def __init__(self):
        self.min_t = 1e-2
        self.rots = rot_config()
        self.trans = trans_config()
        self.sampling = SamplingConfig()
        self.self_condition = True

class DataConfig:
    def __init__(self):
        self.csv_path = "/workspace/4D-Diff-RNA_test_1/Training_data/rna_metadata_debug.csv"
        self.filtering = {
            "max_len": 1000,
            "min_len": 1
        }
        self.min_t = 0.01
        self.samples_per_eval_length = 5   # 5
        self.num_eval_lengths = 10  #10
        self.batch_size = 2     # 5
        self.max_batch_size =  2 # 28  
        self.max_squared_res = 375_000
        self.max_num_res_squared = 375_000
        self.eval_batch_size = 2   # 5
        self.num_workers = 1   # 4
        self.prefetch_factor = 100

class BatchOTConfig:
    def __init__(self):
        self.enabled = True
        self.cost = "kabsch"
        self.noise_per_sample = 1
        self.permute = False

class TrainingConfig:
    def __init__(self):
        self.min_plddt_mask = None
        self.loss = "se3_vf_loss"
        self.bb_atom_scale = 0.1
        self.trans_scale = 0.1
        self.translation_loss_weight = 2.0
        self.t_normalize_clip = 0.9
        self.rotation_loss_weights = 1.0
        self.aux_loss_weight = 1.0
        self.aux_loss_t_pass = 0.25
        self.tors_loss_scale = 1.0
        self.num_non_frame_atoms = 0

class WandbConfig:
    def __init__(self):
        self.name = "rna-frameflow"
        self.project = "se3-fm"
        self.save_code = False
        self.tags = []
        self.mode = "online"

class OptimizerConfig:
    def __init__(self):
        self.lr = 0.0001

class TrainerConfig:
    def __init__(self):
        self.overfit_batches = 0
        self.min_epochs = 1
        self.max_epochs = 200
        self.accelerator = "gpu"
        self.log_every_n_steps = 1
        self.deterministic = False
        self.strategy = "ddp"
        self.check_val_every_n_epoch = 5 #20
        self.accumulate_grad_batches = 1

class CheckpointerConfig:
    def __init__(self):
        self.dirpath = "ckpt/se3-fm/rna-frameflow/"
        self.save_last = True
        self.save_top_k = 3
        self.monitor = "train/loss"
        self.mode = "min"
        self.every_n_epochs = 40

class ExperimentConfig:
    def __init__(self):
        self.debug = True
        self.seed = 123
        self.num_devices = 1  # 4
        self.warm_start = None
        self.warm_start_cfg_override = True
        self.use_swa = False
        self.batch_ot = BatchOTConfig()
        self.training = TrainingConfig()
        self.wandb = WandbConfig()
        self.optimizer = OptimizerConfig()
        self.trainer = TrainerConfig()
        self.checkpointer = CheckpointerConfig()

class Config:
    def __init__(self):
        self.data = DataConfig()
        self.experiment = ExperimentConfig()
        self.model = ModelConfig()
        self.interpolant = InterpolantConfig()

# 데이터 모듈 초기화
def initialize_data_module():
    cfg = Config()
    data_module = PDBNABaseDataModule(cfg.data)
    return data_module

# FlowModule 초기화
def initialize_flow_module():
    cfg = Config()
    flow_module = FlowModule(cfg)
    return flow_module

def train_flow_module():
    # Config 객체 생성
    cfg = Config()

    # 데이터 모듈 초기화
    data_module = PDBNABaseDataModule(cfg.data)

    # FlowModule 초기화
    flow_module = FlowModule(cfg)

    # 모델 체크포인트 콜백 설정
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    # 조기 종료 콜백 
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=10, mode="min"
    )

    # Trainer 설정
    trainer = Trainer(
        max_epochs=100,  # 최대 100 epoch 실행
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        #devices= GPUtil.getAvailable(order='memory', limit = 8)[:cfg.experiment.num_devices],
        # precision = 16, 
        devices=[ 1],
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    # 모델 학습 실행
    trainer.fit(flow_module, datamodule=data_module)

# 실행
if __name__ == "__main__":
    train_flow_module()