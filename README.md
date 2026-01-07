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
We provide pre-processed circuit dynamics in NumPy `.npz` format (stored under `dataset_npz`). The parser builds torch-geometric graphs from two files:

- `graphs.npz`: a dict `designs` where each key is a design name and each value holds node features `x`, edges `edge_index`, `edge_type`, simulation traces `sim_res`, and physical attributes (`power`, `slack`, `area`). Each simulation trace is treated as an independent sample.
- `labels.npz`: a dict `labels` aligned with `designs`, containing the supervision target `y` for pre-training tasks.

The default split keeps designs disjoint between train/val (90/10). To point the runner to a custom dataset, set `--data_dir`, `--graph_npz_name`, and `--label_npz_name` when invoking the training scripts.

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