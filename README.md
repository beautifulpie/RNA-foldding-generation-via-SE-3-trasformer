# Work in progress
- [x] Implement the loss function
- [x] Implement the embedding
- [x] Set up for training 

# How to use
- git clone으로 하면 안되고 RNA-4DDiff-Folding.tar 파일만 다운 받아서 압축해제 하면 됩니다. tar 파일, 51000의 컨테이너, 60100번의 코드가 최신의 코드입니다. (깃허브에 있는 내용이 최신 코드가 아닙니다, tar 파일 제외) 
- 학습 파일은 process_rna_pdb_files.py를 이용해 변환합니다
- configs/config.yaml 파일을 통해 하이퍼 파라미터 및 데이터 경로 등을 조정합니다
- 학습은 train_se3_transformer를 통해 진행됩니다. (wandb로 로그인을 해야 결과를 확인할 수 있습니다. wandb relogin)
- Model의 folding_module이 직접적으로 학습과정을 담당하는 코드이고, 모댈은 diff_model에 있습니다.
- 임베딩 및 기타 여러 잡다한 유틸등은 data 폴더에 있습니다. (데이터셋 및 로더는 pdb_na_dataset_base.py, pdb_na_datasmodule_base.py에 있습니다), 노이징 관련은 Interpolant.py에 있습니다

# Issue
- 아직 generation 된 trajectory에서 에너지가 가장 낮은 frame을 구하는 것은 구현되지 않았습니다.
- valdiation을 데이터 셋 사이즈 부족 떄문인지 결과가 이상하게 나옵니다.

# Dependencies
## basic ML
### change pytorch-cuda version according to your system's specifications
conda install pytorch=2.1.2 torchvision torchaudio pytorch-cuda==12.1 -c pytorch -c nvidia -y
pip install lightning==2.0.7
pip install hydra-core==1.3.2

### pyg (optional to include pytorch-sparse and pytorch-spline-conv)
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.1.2+cu121.html

## molecular preprocessing
conda install mdanalysis MDAnalysisTests -c conda-forge -y
conda install biopandas biopython -c conda-forge -y
conda install openbabel -c conda-forge -y
pip install rdkit
pip install mdtraj
pip install graphein

## misc
pip install wandb hydra-colorlog rootutils rich matplotlib networkx gputil omegaconf beartype jaxtyping dm-tree tmtools POT iminuit tmscoring biotite einops ml_collections
