# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for MMDiff (https://github.com/Profluent-Internships/MMDiff):
# -------------------------------------------------------------------------------------------------------------------------------------

import torch, math
from beartype.typing import Any, Dict, Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data.distributed import DistributedSampler, dist
from torch.utils.data import DataLoader

from datasets import pdb_na_dataset_base

set_mode = 2
dataset_mode = ["PDBNABaseDataset", "PDB_NA_MDDataset", "PDBNABaseDatasetMD" ]
dataset = "PDBNABaseDatasetMD" # dataset_mode[set_mode]

class PDBNABaseDataModule(LightningDataModule):
    def __init__(self, data_cfg, inference_cfg=None):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_cfg = data_cfg

        self.data_test = None

        self.data_train = None
        self.data_val = None

        self.sampler_train = None
        self.sampler_val = None

    def prepare_data(self):
        """Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage):
        if dataset == "PDBNABaseDataset" :
            self.data_train = pdb_na_dataset_base.PDBNABaseDataset(
                                self.hparams.data_cfg,
                                is_training=True,
                            )
            self.data_val = pdb_na_dataset_base.PDBNABaseDataset(
                                self.hparams.data_cfg,
                                is_training=False,
                            )
        elif dataset == "PDB_NA_MDDataset":
            self.data_train = pdb_na_dataset_base.PDB_NA_MDDataset(
                                self.hparams.data_cfg,
                                is_training=True,
                            )
            self.data_val = pdb_na_dataset_base.PDB_NA_MDDataset(
                                self.hparams.data_cfg,
                                is_training=False,
                            )
        elif dataset == "PDBNABaseDatasetMD":
            self.data_train = pdb_na_dataset_base.PDBNABaseDatasetMD(
                                self.hparams.data_cfg,
                                is_training=True,
                            )
            self.data_val = pdb_na_dataset_base.PDBNABaseDatasetMD(
                                self.hparams.data_cfg,
                                is_training=False,
                            )
        
    def train_dataloader(self, rank=None, num_replicas=None):
        num_workers = self.data_cfg.num_workers
        lb = RNALengthBatcher(
                sampler_cfg=self.data_cfg, 
                metadata_csv=self.data_train.csv,
                rank=rank,
                num_replicas=num_replicas
            )
        return DataLoader(
            self.data_train,
            batch_sampler=lb,
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else self.data_cfg.prefetch_factor,
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False,
        )
    
    def val_dataloader(self):
        if dist.is_initialized():
            val_samp = DistributedSampler(self.data_val, shuffle=False)
        else:
            val_samp = None  # 분산 학습이 아닌 경우 샘플러 없음

        return DataLoader(
            self.data_val,
            sampler=val_samp,
            batch_size=self.data_cfg.eval_batch_size,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
        )

    def test_dataloader(self):
        if dist.is_initialized():
            test_samp = DistributedSampler(self.data_test, shuffle=False)
        else:
            test_samp = None  # 분산 학습이 아닌 경우 샘플러 없음

        return DataLoader(
            self.data_test,
            sampler=test_samp,
            batch_size=self.data_cfg.eval_batch_size,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
        )
    
    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

"""
Taken from
https://github.com/microsoft/protein-frame-flow/blob/main/data/pdb_dataloader.py#L162
"""
class RNALengthBatcher:
    def __init__(
            self,
            sampler_cfg,
            metadata_csv,
            seed=123,
            shuffle=True,
            num_replicas=None,
            rank=None,
        ):
        super().__init__()
        # 분산 학습 초기화 여부 확인
        if dist.is_available() and dist.is_initialized():
            self.num_replicas = num_replicas if num_replicas is not None else dist.get_world_size()
            self.rank = rank if rank is not None else dist.get_rank()
        else:
            self.num_replicas = num_replicas if num_replicas is not None else 1
            self.rank = rank if rank is not None else 0
        

        self._sampler_cfg = sampler_cfg
        self._data_csv = metadata_csv
        # Each replica needs the same number of batches. We set the number
        # of batches to arbitrarily be the number of examples per replica.
        self._num_batches = math.ceil(len(self._data_csv) / self.num_replicas)
        self._data_csv['index'] = list(range(len(self._data_csv)))
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.max_batch_size =  self._sampler_cfg.max_batch_size
        
    def _replica_epoch_batches(self):
        # Make sure all replicas share the same seed on each epoch.
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self._data_csv), generator=rng).tolist()
        else:
            indices = list(range(len(self._data_csv)))

        if len(self._data_csv) > self.num_replicas:
            replica_csv = self._data_csv.iloc[
                indices[self.rank::self.num_replicas]
            ]
        else:
            replica_csv = self._data_csv
        
        # Each batch contains multiple RNA of the same length.
        sample_order = []
        for seq_len, len_df in replica_csv.groupby('modeled_na_seq_len'):
            max_batch_size = min(
                self.max_batch_size,
                self._sampler_cfg.max_num_res_squared // seq_len**2 + 1,
            )
            num_batches = math.ceil(len(len_df) / max_batch_size)
            for i in range(num_batches):
                batch_df = len_df.iloc[i*max_batch_size:(i+1)*max_batch_size]
                batch_indices = batch_df['index'].tolist()
                sample_order.append(batch_indices)
        
        # Remove any length bias.
        new_order = torch.randperm(len(sample_order), generator=rng).numpy().tolist()
        return [sample_order[i] for i in new_order]

    def _create_batches(self):
        # Make sure all replicas have the same number of batches Otherwise leads to bugs.
        # See bugs with shuffling https://github.com/Lightning-AI/lightning/issues/10947

        all_batches = []
        num_augments = -1
        while len(all_batches) < self._num_batches:
            all_batches.extend(self._replica_epoch_batches())
            num_augments += 1
            if num_augments > 1000:
                raise ValueError('Exceeded number of augmentations.')
        if len(all_batches) >= self._num_batches:
            all_batches = all_batches[:self._num_batches]
        self.sample_order = all_batches

    def __iter__(self):
        self._create_batches()
        self.epoch += 1
        return iter(self.sample_order)

    def __len__(self):
        if hasattr(self, "sample_order"):  
            return len(self.sample_order)
        else:
            return self._num_batches