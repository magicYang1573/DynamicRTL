module design1(next_pc, cur_pc, rst, clk);
  wire [31:0] _0_;
  wire [31:0] _1_;
  input clk;
  input [0:31] cur_pc;
  output [0:31] next_pc;
  reg [0:31] next_pc = 32'd0;
  input rst;
  assign _1_ = cur_pc +  32'd4;
  always @(posedge clk)
      next_pc <= _0_;
  assign _0_ = rst ?  32'd0 : _1_;
endmodule

module design2(cur_pc, rst, clk, next_pc);
input  cur_pc;
input  rst;
input  clk;
output  next_pc;
reg  next_pc = 1'd0;
wire [31:0] _1_;
wire [31:0] _0_;
always @(posedge clk)
    next_pc <= _0_;
assign _1_ = (cur_pc) + (32'd4);
assign _0_ = (rst) ? (32'd0) : (_1_);
endmodule

module miter();
wire [31:0] next_pc1, next_pc2;
design1 inst1 (.next_pc(next_pc1),.cur_pc(cur_pc),.rst(rst),.clk(clk));
design2 inst2 (.next_pc(next_pc2),.cur_pc(cur_pc),.rst(rst),.clk(clk));
always @(posedge clk) begin
assert(next_pc1 == next_pc2);
end
endmodule