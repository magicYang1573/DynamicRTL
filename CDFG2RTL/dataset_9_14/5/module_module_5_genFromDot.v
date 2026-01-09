module module_5(clock_recovery, reset_n, data_rec, clock_50, data_stand);
input  clock_recovery;
input  reset_n;
input [13:0] data_rec;
input  clock_50;
output [13:0] data_stand;
reg [13:0] data_stand;
wire [13:0] _0_;
wire  _1_;
wire [13:0] _2_;
always @(posedge clock_50)
    data_stand <= _0_;
assign _0_ = (_1_) ? (14'h0) : (_2_);
assign _1_ = !(reset_n);
assign _2_ = (clock_recovery) ? (data_rec) : (data_stand);
endmodule