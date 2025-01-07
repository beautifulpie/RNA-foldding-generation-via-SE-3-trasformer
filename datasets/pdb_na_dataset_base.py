# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for MMDiff (https://github.com/Profluent-Internships/MMDiff):
# -------------------------------------------------------------------------------------------------------------------------------------

import functools as fn
import numpy as np
import pandas as pd
import os
import re

import torch, tree
from beartype.typing import Optional
from omegaconf import DictConfig
from torch.utils.data import Dataset

from datasets import data_transforms
from datasets import utils as du
from datasets import rigid_utils
from collections import defaultdict

NUM_NA_RESIDUE_ATOMS = 23
max_length_for_residue = 300

class PDBNABaseDataset(Dataset):
    def __init__(
        self,
        data_conf,
        is_training: bool,
        filter_eval_split: bool = False,
        inference_cfg: Optional[DictConfig] = None,
    ):
        self._data_conf = data_conf
        self._is_training = is_training
        self._filter_eval_split = filter_eval_split
        self._inference_cfg = inference_cfg
        self._init_metadata_and_splits()

    @property
    def data_conf(self):
        return self._data_conf

    @property
    def ddpm(self):
        return self._ddpm

    @property
    def is_training(self):
        return self._is_training

    @property
    def filter_eval_split(self):
        return self._filter_eval_split

    @property
    def inference_cfg(self):
        return self._inference_cfg

    @property
    def sequence_ddpm(self):
        return self._sequence_ddpm

    def _init_metadata_and_splits(self): # new
        pdb_csv = pd.read_csv(self.data_conf.csv_path)
        self.raw_csv = pdb_csv # original CSV before filtering and transformations

        pdb_csv.fillna(
            {"helix_percent": 0, "coil_percent": 0, "strand_percent": 0, "radius_gyration": 0},
            inplace=True,
        )

        filter_conf = self.data_conf.filtering

        # length-based filtering
        pdb_csv = pdb_csv[pdb_csv.modeled_na_seq_len <= filter_conf.max_len]
        pdb_csv = pdb_csv[pdb_csv.modeled_na_seq_len >= filter_conf.min_len]

        pdb_csv = pdb_csv[pdb_csv.quaternary_category == "homomer"] # ignore multimers
        pdb_csv = pdb_csv[pdb_csv.num_protein_chains == 0] # remove proteins
        pdb_csv = pdb_csv.sort_values("modeled_na_seq_len", ascending=False)
        # pdb_csv.reset_index(inplace=True) # reset the index to ensure samples are taken within proper bounds

        if self.is_training:
            self.csv = pdb_csv
            print (f"Training: {len(self.csv)} filtered examples") # FINAL post-filtered set for RNASolo from metadata CSV
        else:
            # validation data
            eval_csv = pdb_csv
            all_lengths = pdb_csv['modeled_na_seq_len'].unique()
            length_indices = (len(all_lengths) - 1) * np.linspace(0.0, 1.0, self._data_conf.num_eval_lengths)
            length_indices = length_indices.astype(int)
            eval_lengths = all_lengths[length_indices]
            eval_csv = eval_csv[eval_csv.modeled_na_seq_len.isin(eval_lengths)]

            eval_csv = eval_csv.groupby("modeled_na_seq_len").sample(self._data_conf.samples_per_eval_length, replace=True, random_state=123)
            eval_csv = eval_csv.sort_values(["modeled_na_seq_len"], ascending=False)
            self.csv = eval_csv
            print (f"Validation: {len(self.csv)} examples with lengths {eval_lengths}.")

    # cache make the same sample in same batch
    @fn.lru_cache(maxsize=100)
    def _process_csv_row(self, processed_file_path):
        processed_feats = du.read_pkl(processed_file_path)
        processed_feats = du.parse_complex_feats(processed_feats)

        # Designate which residues to diffuse and which to fix. By default, diffuse all residues
        diffused_mask = np.ones_like(processed_feats["bb_mask"])
        if np.sum(diffused_mask) < 1:
            raise ValueError("Must be diffused")
        fixed_mask = 1 - diffused_mask
        processed_feats["fixed_mask"] = fixed_mask

        # Distinguish between protein residues and nucleic acid residues using corresponding masks
        processed_feats["is_protein_residue_mask"] = (
            processed_feats["molecule_type_encoding"][:, 0] == 1
        )
        processed_feats["is_na_residue_mask"] = (
            processed_feats["molecule_type_encoding"][:, 1] == 1
        ) | (processed_feats["molecule_type_encoding"][:, 2] == 1)
        na_inputs_present = processed_feats["is_na_residue_mask"].any().item()

        # Find interfaces
        inter_chain_interacting_residue_mask = torch.zeros(len(diffused_mask), dtype=torch.bool)
        inter_chain_interacting_residue_mask[processed_feats["inter_chain_interacting_idx"]] = True

        # Only take modeled residues
        modeled_idx = processed_feats["modeled_idx"]
        min_idx = np.min(modeled_idx)
        max_idx = np.max(modeled_idx)
        
        del processed_feats["modeled_idx"]
       
        if processed_feats["protein_modeled_idx"] is None:
            del processed_feats["protein_modeled_idx"]
       
        if processed_feats["na_modeled_idx"] is None:
            del processed_feats["na_modeled_idx"]
       
        processed_feats = tree.map_structure(lambda x: x[min_idx : (max_idx + 1)], processed_feats)
        inter_chain_interacting_residue_mask = inter_chain_interacting_residue_mask[
            min_idx : (max_idx + 1)
        ]

        # Run through OpenFold data transforms.
        chain_feats, na_chain_feats = (
            {
                "aatype": torch.tensor(processed_feats["aatype"]).long(),
                "all_atom_positions": torch.tensor(processed_feats["atom_positions"]).double(),
                "all_atom_mask": torch.tensor(processed_feats["atom_mask"]).double(),
                "atom_deoxy": torch.tensor(processed_feats["atom_deoxy"]).bool(),
            },
            {},
        )

        if na_inputs_present:
            na_chain_feats = {
                "aatype": chain_feats["aatype"][processed_feats["is_na_residue_mask"]],
                "all_atom_positions": chain_feats["all_atom_positions"][
                    processed_feats["is_na_residue_mask"]
                ][:, :NUM_NA_RESIDUE_ATOMS],
                "all_atom_mask": chain_feats["all_atom_mask"][
                    processed_feats["is_na_residue_mask"]
                ][:, :NUM_NA_RESIDUE_ATOMS],
                "atom_deoxy": chain_feats["atom_deoxy"][processed_feats["is_na_residue_mask"]],
            }
            na_chain_feats["atom23_gt_positions"] = na_chain_feats[
                "all_atom_positions"
            ]  # cache `atom23` positions
        
        if na_inputs_present:
            na_chain_feats = data_transforms.make_atom23_masks(na_chain_feats)
            data_transforms.atom23_list_to_atom27_list(
                na_chain_feats, ["all_atom_positions", "all_atom_mask"], inplace=True
            )
            na_chain_feats = data_transforms.atom27_to_frames(na_chain_feats)
            na_chain_feats = data_transforms.atom27_to_torsion_angles()(na_chain_feats)

        # Merge available protein and nucleic acid features using padding where necessary
        chain_feats = du.concat_complex_torch_features(
            chain_feats,
            {}, # empty protein features
            na_chain_feats,
            feature_concat_map=du.COMPLEX_FEATURE_CONCAT_MAP,
            add_batch_dim=False,
        )

        # cleaner version
        final_feats = {
            "torsion_angles_sin_cos": chain_feats["torsion_angles_sin_cos"],
            "is_na_residue_mask": processed_feats["is_na_residue_mask"]
        }

        rigids_1 = rigid_utils.Rigid.from_tensor_4x4(
                                        chain_feats["rigidgroups_gt_frames"]
                                    )[:, 0]
        rotmats_1 = rigids_1.get_rots().get_rot_mats()
        trans_1 = rigids_1.get_trans()

        final_feats["rotmats_1"] = rotmats_1
        final_feats["trans_1"] = trans_1
        final_feats['res_mask'] = torch.tensor(processed_feats['bb_mask']).int()

        """
        Final sample dict keys:
            - torsion_angles_sin_cos
            - rotmats_1
            - trans_1
            - res_mask
            - is_na_residue_mask
        """

        return final_feats

    def convert_dict_float64_items_to_float32(self, dictionary):
        converted_dict = {}
        for key, value in dictionary.items():
            if isinstance(value, np.ndarray) and value.dtype == np.float64:
                converted_dict[key] = value.astype(np.float32)
            elif isinstance(value, torch.Tensor) and value.dtype == torch.float64:
                converted_dict[key] = value.float()
            else:
                converted_dict[key] = value  # For non-NumPy array and non-PyTorch tensor types
        return converted_dict
    
    def __getitem__(self, idx):
        example_idx = idx
        csv_row = self.csv.iloc[example_idx]
        processed_file_path = csv_row["processed_path"]
        final_feats = self._process_csv_row(processed_file_path) # get the features for this instance

        # Convert all features to tensors.
        final_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), final_feats
        )
        final_feats = du.pad_feats(final_feats, csv_row["modeled_na_seq_len"])
        final_feats = self.convert_dict_float64_items_to_float32(final_feats)
        
        return final_feats
    
    def __len__(self):
        return len(self.csv)

class LengthDataset(torch.utils.data.Dataset):
    def __init__(self, samples_cfg):
        self._samples_cfg = samples_cfg
        
        all_sample_lengths = range(
            self._samples_cfg.min_length,
            self._samples_cfg.max_length+1,
            self._samples_cfg.length_step
        )

        # ignore the above variable if subset is given
        if samples_cfg.length_subset is not None:
            all_sample_lengths = [int(x) for x in samples_cfg.length_subset]
        
        print (f"#### Generating sequences with the following lengths: {list(all_sample_lengths)}")

        all_sample_ids = []
        for length in all_sample_lengths:
            for sample_id in range(self._samples_cfg.samples_per_length):
                all_sample_ids.append((length, sample_id))
        
        self._all_sample_ids = all_sample_ids

    def __len__(self):
        return len(self._all_sample_ids)

    def __getitem__(self, idx):
        num_res, sample_id = self._all_sample_ids[idx]
        batch = {
            'num_res': num_res,
            'sample_id': sample_id,
        }
        return batch
    
class PDBNABaseDatasetMD(Dataset):
    def __init__(self, data_conf, is_training: bool, filter_eval_split: bool = False, inference_cfg: Optional[DictConfig] = None):
        self._data_conf = data_conf
        self._is_training = is_training
        self._filter_eval_split = filter_eval_split
        self._inference_cfg = inference_cfg
        self._init_metadata_and_splits()

    def _init_metadata_and_splits(self):
        pdb_csv = pd.read_csv(self.data_conf.csv_path)
        
        # 결측값 채우기
        pdb_csv.fillna(
            {"helix_percent": 0, "coil_percent": 0, "strand_percent": 0, "radius_gyration": 0},
            inplace=True,
        )

        # 유효한 경로만 필터링
        pdb_csv = pdb_csv[pdb_csv["processed_path"].apply(lambda x: os.path.exists(x))]
        # pdb_csv = pdb_csv[pdb_csv["processed_path"].apply(self._is_valid_file)]  # 여기서 다 걸림
        
        filter_conf = self.data_conf.filtering
        pdb_csv = pdb_csv[pdb_csv.modeled_na_seq_len <= filter_conf["max_len"]]
        pdb_csv = pdb_csv[pdb_csv.modeled_na_seq_len >= filter_conf["min_len"]]
        pdb_csv = pdb_csv[pdb_csv.quaternary_category == "homomer"]
        pdb_csv = pdb_csv[pdb_csv.num_protein_chains == 0]
        pdb_csv = pdb_csv.sort_values("modeled_na_seq_len", ascending=False)

        self.csv = pdb_csv

        self.rna_name = list(set(pdb_csv.rna_name))
        # print(f"Initialized rna_name: {self.rna_name}")
        # print(f"rna_name length: {len(self.rna_name)}")
        if not self.rna_name:
            raise ValueError("rna_name is empty! Check your data preprocessing.")

    @fn.lru_cache(maxsize=100)
    def _process_csv_row(self, pdb_file_paths, rna_name):
        """
        입력 : 같은 RNA들 입력 받기
        출력 : 프로세싱 해서 텐서로 출력
        """
        processed_frames = []

        for pdb_path in pdb_file_paths:
            try:
                # .pkl 파일 로드
                processed_feats = du.read_pkl(pdb_path)
                processed_feats = du.parse_complex_feats(processed_feats)
            except Exception as e:
                print(f"Error loading file {pdb_path}: {e}")
                continue
            
            # Designate which residues to diffuse and which to fix. By default, diffuse all residues
            diffused_mask = np.ones_like(processed_feats["bb_mask"])
            if np.sum(diffused_mask) < 1:
                raise ValueError("Must be diffused")
            
            fixed_mask = 1 - diffused_mask
            processed_feats["fixed_mask"] = fixed_mask
            
            processed_feats["is_protein_residue_mask"] = (
                processed_feats["molecule_type_encoding"][:, 0] == 1
            )

            # Distinguish between protein residues and nucleic acid residues using corresponding masks
            processed_feats["is_na_residue_mask"] = (
                processed_feats["molecule_type_encoding"][:, 1] == 1
            ) | (processed_feats["molecule_type_encoding"][:, 2] == 1)
            na_inputs_present = processed_feats["is_na_residue_mask"].any().item()

            # Find interface
            inter_chain_interacting_residue_mask = torch.zeros(len(diffused_mask), dtype=torch.bool)
            inter_chain_interacting_residue_mask[processed_feats["inter_chain_interacting_idx"]] = True

            # Only take modeled residues
            modeled_idx = processed_feats["modeled_idx"]
            min_idx = np.min(modeled_idx)
            max_idx = np.max(modeled_idx)

            del processed_feats["modeled_idx"]
        
            if processed_feats["protein_modeled_idx"] is None:
                del processed_feats["protein_modeled_idx"]
        
            if processed_feats["na_modeled_idx"] is None:
                del processed_feats["na_modeled_idx"]

            processed_feats = tree.map_structure(lambda x: x[min_idx : (max_idx + 1)], processed_feats)
            inter_chain_interacting_residue_mask = inter_chain_interacting_residue_mask[
                min_idx : (max_idx + 1)
            ]

            # Run through OpenFold data transforms.
            chain_feats, na_chain_feats = (
                {
                    "aatype": torch.tensor(processed_feats["aatype"]).long(),
                    "all_atom_positions": torch.tensor(processed_feats["atom_positions"]).double(),
                    "all_atom_mask": torch.tensor(processed_feats["atom_mask"]).double(),
                    "atom_deoxy": torch.tensor(processed_feats["atom_deoxy"]).bool(),
                },
                {},
            )

            if na_inputs_present:
                na_chain_feats = {
                    "aatype": chain_feats["aatype"][processed_feats["is_na_residue_mask"]],
                    "all_atom_positions": chain_feats["all_atom_positions"][
                        processed_feats["is_na_residue_mask"]
                    ][:, :NUM_NA_RESIDUE_ATOMS],
                    "all_atom_mask": chain_feats["all_atom_mask"][
                        processed_feats["is_na_residue_mask"]
                    ][:, :NUM_NA_RESIDUE_ATOMS],
                    "atom_deoxy": chain_feats["atom_deoxy"][processed_feats["is_na_residue_mask"]],
                }
                na_chain_feats["atom23_gt_positions"] = na_chain_feats[
                    "all_atom_positions"
                ]  # cache `atom23` positions
            
            if na_inputs_present:
                na_chain_feats = data_transforms.make_atom23_masks(na_chain_feats)
                data_transforms.atom23_list_to_atom27_list(
                    na_chain_feats, ["all_atom_positions", "all_atom_mask"], inplace=True
                )
                na_chain_feats = data_transforms.atom27_to_frames(na_chain_feats)
                na_chain_feats = data_transforms.atom27_to_torsion_angles()(na_chain_feats)

            # Merge available protein and nucleic acid features using padding where necessary
            chain_feats = du.concat_complex_torch_features(
                chain_feats,
                {}, # empty protein features
                na_chain_feats,
                feature_concat_map=du.COMPLEX_FEATURE_CONCAT_MAP,
                add_batch_dim=False,
            )

            # cleaner version
            final_feats = {
                "torsion_angles_sin_cos": chain_feats["torsion_angles_sin_cos"],
                "is_na_residue_mask": processed_feats["is_na_residue_mask"]
            }

            rigids_1 = rigid_utils.Rigid.from_tensor_4x4(
                                            chain_feats["rigidgroups_gt_frames"]
                                        )[:, 0]
            rotmats_1 = rigids_1.get_rots().get_rot_mats()
            trans_1 = rigids_1.get_trans()
            
            final_feats["rotmats_1"] = rotmats_1
            final_feats["trans_1"] = trans_1
            final_feats['res_mask'] = torch.tensor(processed_feats['bb_mask']).int()

            """
            Final sample dict keys:
                - torsion_angles_sin_cos
                - rotmats_1
                - trans_1
                - res_mask
                - is_na_residue_mask
            """
            final_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), final_feats
            )

            final_feats = du.pad_feats(final_feats, max_length_for_residue)
            final_feats = self.convert_dict_float64_items_to_float32(final_feats)

            processed_frames.append(final_feats)

        combined_tensor = {
            # 'torsion_angles_sin_cos': [],
            # 'rotmats_1': [],
            # 'trans_1': [],
            # 'res_mask': [],
            # 'is_na_residue_mask': []
        }

        time_dim = len(processed_frames)
        for key in processed_frames[0].keys():
            time_tensors = [frame[key] for frame in processed_frames]
            
            # 모든 요소를 torch.Tensor로 변환
            # 이미 torch.Tensor인 경우에는 그대로 유지됨
            time_tensors = [torch.tensor(t) if not torch.is_tensor(t) else t for t in time_tensors]
            
            stacked_tensor = torch.stack(time_tensors, dim=0)
            combined_tensor[key] = stacked_tensor
            
        # print(f"{key} stacked shape: {stacked_tensor.shape}")

        # for key, value in combined_tensor.items():
        #     if isinstance(value, torch.Tensor):  # Tensor인 경우 shape 출력
        #         print(f"{key}: {value.shape}")
        #     elif isinstance(value, list) and all(isinstance(v, tuple) for v in value):  # List[Tuple]인 경우 길이 출력
        #         print(f"{key}: List of {len(value)} tuples")
        #     else:  # 다른 타입인 경우 타입 출력
        #         print(f"{key}: {type(value)}")

        return combined_tensor

    def __getitem__(self, idx):
        if idx >= len(self.rna_name):
            rna = self.rna_name[idx % len(self.rna_name)]     # 수정할 것 
            # raise IndexError(f"Index {idx} is out of range for rna_name with length {len(self.rna_name)}")
        else :
           rna = self.rna_name[idx]    
        filtered_df = self.csv[self.csv['rna_name'] == rna]
        pdb_file_paths = filtered_df['processed_path'].tolist()

        if not pdb_file_paths:
            raise KeyError(f"RNA {rna} does not contain 'processed_path'.")

        combined_feats = self._process_csv_row(pdb_file_paths = tuple(pdb_file_paths), rna_name = rna)
        
        torsion_angles = combined_feats['torsion_angles_sin_cos']

        Frame = combined_feats["is_na_residue_mask"].size(0)

        input_feat = {
            'res_mask': combined_feats['res_mask'],       # [T, N]
            'trans_1': combined_feats['trans_1'],       # [T, N, 3]
            'rotmats_1': combined_feats['rotmats_1'],# [T, N, 3, 3]
            'trans_sc': torch.zeros_like(combined_feats['trans_1']),    # [T, N, 3]
            'gt_torsions': torsion_angles.view(torsion_angles.shape[0], -1), # [T, N, 8]
            'gt_backbone_trajectory': [(   # 이부분 없애고 시간에 따라 구현 (모델 수정)
                    torch.rand(3),       # 랜덤 translation vector [3]
                    torch.rand(3, 3)    # 랜덤 rotation matrix [3, 3]
                )                        
                for _ in range(max_length_for_residue)
            ],
            "coord_4d" : torch.rand(Frame, max_length_for_residue, 3), # Shape = [T, N, 3]
            "is_na_residue_mask" : combined_feats['res_mask'],
            'torsion_angles_sin_cos': torsion_angles
        }

        # for key, value in input_feat.items():
        #     if isinstance(value, torch.Tensor):  # Tensor인 경우 shape 출력
        #         print(f"{key}: {value.shape}")
        #     elif isinstance(value, list) and all(isinstance(v, tuple) for v in value):  # List[Tuple]인 경우 길이 출력
        #         print(f"{key}: List of {len(value)} tuples")
        #     else:  # 다른 타입인 경우 타입 출력
        #         print(f"{key}: {type(value)}")


        return input_feat

    def __len__(self):
        print(f"Length : {len(self.rna_name)}")
        return len(self.rna_name)

    @property
    def data_conf(self):
        return self._data_conf

    @property
    def ddpm(self):
        return self._ddpm

    @property
    def is_training(self):
        return self._is_training

    @property
    def filter_eval_split(self):
        return self._filter_eval_split

    @property
    def inference_cfg(self):
        return self._inference_cfg

    @property
    def sequence_ddpm(self):
        return self._sequence_ddpm

    def _is_valid_file(self, path):
        try:
            processed_feats = torch.load(path)
            if "torsion_angles_sin_cos" not in processed_feats or "bb_mask" not in processed_feats:
                return False
            return True
        except:
            return False
        
    def convert_dict_float64_items_to_float32(self, dictionary):
        converted_dict = {}
        for key, value in dictionary.items():
            if isinstance(value, np.ndarray) and value.dtype == np.float64:
                converted_dict[key] = value.astype(np.float32)
            elif isinstance(value, torch.Tensor) and value.dtype == torch.float64:
                converted_dict[key] = value.float()
            else:
                converted_dict[key] = value  # For non-NumPy array and non-PyTorch tensor types
        return converted_dict