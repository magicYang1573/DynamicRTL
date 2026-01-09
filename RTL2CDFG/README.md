# CDFG generator
A Python program that reads Verilog code to generate a control data flow graph.

## configure

### Yosys Installation
Refer to the [yosy installation guide](https://github.com/YosysHQ/yosys?tab=readme-ov-file).

### Python Package
```
pip install -r requirements.txt
```

### Graphviz Download(Optional)
If you want to visualize the results,please download graphviz.
Refer to the [installation guide](https://www.graphviz.org/download/).

## Usage

### Single Run

#### First Way
First, please save the Verilog code in the ***./verilog_data*** directory.
This will generate CDFG for the specified files in the verilog_data directory.
```
cd ./v2cdfg && ./cdfg_generator.sh -nodivide *.v
```
The generated results are saved in ***./v2cdfg/cdfg***.

#### Second Way
This way you can generate cdfg for rtl code in any path.
```
cd ./v2cdfg && python3 cdfg_generator.py rtl_path.
```
The generated results are saved in ***./v2cdfg/cdfg***.

### Batch Run
This will generate CDFG for all Verilog files in the ***verilog_data*** directory.
```
cd ./v2cdfg && python3 gencdfg-multi-thread.py
```
The generated results are saved in ***./log***.
