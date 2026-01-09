read_verilog -sv dataset_9_14/1/miter.sv
hierarchy -top miter
proc
flatten
sat -tempinduct -prove-asserts -verify