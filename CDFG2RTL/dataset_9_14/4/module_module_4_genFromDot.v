module module_4(reset, count, clock, data_o);
input  reset;
input  count;
input  clock;
output [7:0] data_o;
reg [7:0] data_o;
wire [31:0] _1_;
wire [7:0] _2_;
wire [7:0] _0_;
always @(posedge clock)
    data_o <= _0_;
assign _1_ = (data_o) + (32'd1);
assign _2_ = (count) ? (_1_[7:0]) : (data_o);
assign _0_ = (reset) ? (8'h0) : (_2_);
endmodule