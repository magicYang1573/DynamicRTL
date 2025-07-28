<div style="text-align: center;">
    <h1>DynamicRTL: RTL Representation Learning for Dynamic Circuit Behavior</h1>
</div>

<div style="text-align: right;">
    <div style="display: inline-block; text-align: left;">
        <span>Author: Anonymous Authors</span></br>
        <span>Date: 2025-07-28</span>
    </div>
</div>

## 1. Introduction
This is the source code of the paper submitted to AAAI'2026: *DynamicRTL: RTL Representation Learning for Dynamic Circuit Behavior*. 

The repository contains the source code for training and evaluation the DR-GNN model, and the baseline models for comparison and ablation experiments. To avoid an oversized repo, a subset of the dataset (2000 in 6300) is preprocessed and provided in .npz format. The overall circuit files, CDFGs, and simulation data will be open-source after paper accepted. 
<!-- Moreover, we will also open-source the flow of processing CDFGs and simulation results, which is removed from our [CDFG_utils](src/utils/CDFG_utils.py "CDFG_urils"), and the whole project of constructing CDFGs from Verilog design files to contribute to the community of Programming Language especially RTL representation learning, hardware design and Electronic Design Automation communities. -->
![Overview](_picture/overview.png "Overview")

## 2. Requirements
```bash
conda create --name test_req python=3.11.9
conda activate test_req
pip install -r requirements_torch.txt
pip install -r requirements_PyG.txt
```

## 3. Usage
To train and evaluate the model on pre-training tasks.
```bash
bash train.sh
```
To train and evaluate the model on downstream tasks after pre-training.
```bash
bash train_downstream.sh
```