<div style="text-align: center;">
    <h1>DynamicRTL: RTL Representation Learning for Dynamic Circuit Behavior</h1>
</div>

## 1. Introduction
This is the source code of the paper submitted to AAAI 2026: *DynamicRTL: RTL Representation Learning for Dynamic Circuit Behavior* ([arXiv:2511.09593](https://arxiv.org/abs/2511.09593)).

The repository contains the source code for the training and evaluation of the DR-GNN model, and the baseline models for comparison and ablation experiments. The circuit dynamic dataset introduced in the paper is preprocessed and provided in .npz format. 

<!-- The overall circuit files, CDFGs, and simulation data will be open-source soon.  -->
<!-- Moreover, we will also open-source the flow of processing CDFGs and simulation results, which is removed from our [CDFG_utils](src/utils/CDFG_utils.py "CDFG_urils"), and the whole project of constructing CDFGs from Verilog design files to contribute to the community of Programming Language especially RTL representation learning, hardware design and Electronic Design Automation communities. -->

![Overview](_picture/overview.png "Overview")

## 2. Dataset
- **What we provide**: a pre-processed circuit dynamics dataset in NumPy `.npz` format under `dataset_npz/`.
- **Subsets in repo**: `graphs-200.npz`, `labels-200.npz`, `graphs-2000.npz`, `labels-2000.npz` (for quick smoke tests).
- **Full set download**: Google Drive folder: https://drive.google.com/drive/folders/18-jEWtzbFD719yQPGcUyb8fB72HiEias?usp=drive_link

### File structure
- `graphs*.npz`: dictionary `designs` keyed by design name. Each entry contains:
  - `x`: node features (type one-hot, width, etc.)
  - `edge_index`, `edge_type`: graph topology
  - `sim_res`: simulation traces; each trace becomes an independent sample
  - `has_sim_res`: trace availability flag
  - `power`, `slack`, `area`: physical attributes per trace
- `labels*.npz`: dictionary `labels` aligned with `designs`, carrying supervision target `y` for pre-training.

### Splits and loading
- Default split is design-disjoint train/val (90/10) in `NpzParser` (`--trainval_split` adjustable in code).
- To use a specific split or custom files, pass `--data_dir`, `--graph_npz_name`, and `--label_npz_name` to the training scripts.


## 3. Requirements and Usage
```bash
conda create --name test_req python=3.11.9
conda activate test_req
pip install -r requirements_torch.txt
pip install -r requirements_PyG.txt
```
To train and evaluate the model on pre-training tasks.
```bash
bash train.sh
```
To train and evaluate the model on downstream tasks after pre-training.
```bash
bash train_downstream.sh
```


### Project layout
- `src/train.py`, `src/trainer.py`: DR-GNN pre-training (branch-hit classification and transition-rate regression).
- `src/train_downstream.py`, `src/trainer_downstream.py`: downstream fine-tuning for power/area/slack and assertion prediction.
- `src/model_arch.py`, `src/gdrtl_arch/`: encoders/decoders used for dynamic representation learning.
- `src/model_downstream.py`: task-specific heads (assertion MLPs, power GNN readout).
- `src/npz_parser.py`: converts `.npz` pairs into torch-geometric `Data` objects with per-trace samples.
- `src/utils/`: helpers for logging, metrics, and CDFG utilities.

## 4. Citation
```
@inproceedings{ma2026dynamicrtl,
  title     = {{DynamicRTL}: {RTL} Representation Learning for Dynamic Circuit Behavior},
  author    = {Ma, Ruiyang and Zhou, Yunhao and Wang, Yipeng and Liu, Yi and Shi, Zhengyuan and Zheng, Ziyang and Chen, Kexin and He, Zhiqiang and Yan, Lingwei and Chen, Gang and Xu, Qiang and Luo, Guojie},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2026}
}
```