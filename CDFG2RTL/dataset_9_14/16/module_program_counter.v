module program_counter(next_pc, cur_pc, rst, clk);
  wire [31:0] _0_;
  wire [31:0] _1_;
  input clk;
  input [0:31] cur_pc;
  output [0:31] next_pc;
  reg [0:31] next_pc;
  input rst;
  assign _1_ = cur_pc +  32'd4;
  always @(posedge clk)
      next_pc <= _0_;
  assign _0_ = rst ?  32'd0 : _1_;
endmodule