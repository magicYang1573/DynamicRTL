module module_16(cur_pc, rst, clk, next_pc);
input  cur_pc;
input  rst;
input  clk;
output  next_pc;
reg  next_pc;
wire [31:0] _1_;
wire [31:0] _0_;
always @(posedge clk)
    next_pc <= _0_;
assign _1_ = (cur_pc) + (32'd4);
assign _0_ = (rst) ? (32'd0) : (_1_);
endmodule