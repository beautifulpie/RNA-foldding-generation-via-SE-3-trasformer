# RNA-foldding-generation-via-SE-3-trasformer

RNA-foldding-generation-via-SE-3-trasformer는 RNA 3차원/4차원 구조 궤적을 학습하기 위한 PyTorch Lightning 기반 확산/플로우 모델 연구 코드베이스입니다. RNA PDB 파일을 전처리해 학습 가능한 피처로 변환하고, SE(3) 기하 정보를 다루는 Invariant Point Attention(IPA) 계열 모델로 backbone frame, translation, rotation, torsion angle을 예측하는 것을 목표로 합니다.

## 주요 기능

- **RNA PDB 전처리**: `process_rna_pdb_files.py`를 통해 RNA PDB 파일을 파싱하고 pickle 기반 학습 데이터와 metadata CSV를 생성합니다.
- **4D/SE(3) 확산 학습**: `data/interpolant.py`의 노이징 및 보간 로직과 `Model/folding_module.py`의 LightningModule을 이용해 시간에 따른 구조 복원 과정을 학습합니다.
- **IPA 기반 구조 모델**: `Model/diff_model.py`의 `FlowModel`은 node/edge embedding, spatial/motion alignment module, Invariant Point Attention block, torsion angle head를 결합합니다.
- **Hydra 설정 관리**: `configs/config.yaml`에서 데이터 경로, 모델 크기, optimizer, trainer, checkpoint, wandb 설정을 관리합니다.
- **PyTorch Lightning 학습 루프**: `train_se3_transformer.py`가 datamodule, model, checkpoint, wandb logger, GPU device selection을 구성해 학습을 실행합니다.

## 레포지토리 구조

```text
.
├── Model/                         # 확산/플로우 모델, folding module, loss, analysis utilities
├── data/                          # RNA/NA 파싱, dataset/datamodule, embedding, geometry utilities
├── invariant_point_attention/     # IPA 구현
├── configs/config.yaml            # 기본 학습 설정
├── process_rna_pdb_files.py       # RNA PDB 전처리 스크립트
├── train_se3_transformer.py       # 메인 학습 엔트리포인트
├── metadata/                      # 예시 metadata
└── preprocessed_pdbs/             # 전처리된 데이터 및 metadata 출력 위치
```

## 설치

> CUDA 및 PyTorch 버전은 사용 중인 GPU/드라이버 환경에 맞게 조정하세요.

```bash
# 기본 ML 의존성
conda install pytorch=2.1.2 torchvision torchaudio pytorch-cuda==12.1 -c pytorch -c nvidia -y
pip install lightning==2.0.7 hydra-core==1.3.2

# PyTorch Geometric
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.1.2+cu121.html

# 분자 구조 전처리 의존성
conda install mdanalysis MDAnalysisTests -c conda-forge -y
conda install biopandas biopython -c conda-forge -y
conda install openbabel -c conda-forge -y
pip install rdkit mdtraj graphein

# 기타 유틸리티
pip install wandb hydra-colorlog rootutils rich matplotlib networkx gputil omegaconf \
  beartype jaxtyping dm-tree tmtools POT iminuit tmscoring biotite einops ml_collections
```

## 데이터 준비

RNA PDB 파일이 들어 있는 디렉터리를 지정해 전처리를 실행합니다.

```bash
python process_rna_pdb_files.py \
  --pdb_dir /path/to/rna_pdb_files \
  --write_dir ./preprocessed_pdbs \
  --num_processes 16 \
  --skip_existing
```

전처리 결과는 기본적으로 `preprocessed_pdbs/` 아래에 저장되며, 학습에는 metadata CSV의 `processed_path`가 사용됩니다. 이후 `configs/config.yaml`의 `data_cfg.csv_path`를 생성된 metadata CSV 경로로 수정합니다.

## 설정

핵심 설정 파일은 `configs/config.yaml`입니다.

- `data_cfg`: metadata CSV 경로, batch size, worker 수, filtering 조건
- `interpolant`: translation/rotation schedule, sampling timestep, self-conditioning
- `experiment`: wandb, trainer, checkpoint, optimizer, loss weight 설정
- `model`: node/edge embedding 크기, IPA block 수, pretrained embedding parameter 경로

환경마다 절대 경로가 다르므로 다음 항목은 반드시 확인하세요.

- `data_cfg.csv_path`
- `model.node_features.Node_param`
- `model.edge_features.Edge_param`
- `experiment.checkpointer.dirpath`

## 학습 실행

기본 설정으로 학습을 시작하려면 다음을 실행합니다.

```bash
python train_se3_transformer.py
```

Hydra override를 사용해 설정을 명령행에서 바꿀 수 있습니다.

```bash
python train_se3_transformer.py \
  data_cfg.csv_path=/path/to/rna_metadata.csv \
  experiment.debug=False \
  experiment.wandb.name=my_run \
  experiment.num_devices=1
```

`experiment.debug=False`일 때는 wandb logger와 checkpoint callback이 활성화됩니다. wandb를 사용하는 경우 사전에 로그인하세요.

```bash
wandb login
```

## 주요 학습 흐름

1. `process_rna_pdb_files.py`가 PDB 구조를 파싱해 residue/atom feature와 metadata를 생성합니다.
2. `PDBNABaseDataModule`이 metadata CSV를 읽어 train/validation dataloader를 구성합니다.
3. `Interpolant`가 translation/rotation/torsion 관련 noisy batch를 생성합니다.
4. `FlowModel`이 noisy frame과 time embedding을 입력받아 denoised translation, rotation, torsion을 예측합니다.
5. `FlowModule`이 backbone atom loss, translation loss, rotation loss, torsion loss 등을 계산하고 Lightning trainer가 최적화합니다.

## 현재 개발 상태 및 참고 사항

- 모델과 데이터 파이프라인은 연구/실험용 코드에 가깝습니다.
- generation trajectory에서 에너지가 가장 낮은 frame을 선택하는 후처리는 아직 구현되어 있지 않습니다.
- validation 결과는 데이터셋 크기와 split 구성에 민감할 수 있습니다.
- 현재 설정 파일 일부는 로컬 절대 경로를 포함하므로, 다른 환경에서 실행할 때 경로를 수정해야 합니다.
- 레포지토리에는 실험 로그와 checkpoint 예시가 포함되어 있을 수 있습니다. 대규모 실험 결과는 별도 artifact storage 또는 wandb 사용을 권장합니다.

## 라이선스 및 출처

일부 구성 요소는 MMDiff 및 protein-frame-flow 계열 구현을 참고/수정한 코드 주석을 포함합니다. 재사용 또는 공개 배포 전 각 원본 프로젝트의 라이선스를 확인하세요.
