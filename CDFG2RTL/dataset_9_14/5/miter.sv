module design1(clock_recovery, clock_50, reset_n, data_rec, data_stand);
  wire [13:0] _0_;
  wire _1_;
  wire [13:0] _2_;
  input clock_50;
  input clock_recovery;
  input [13:0] data_rec;
  output [13:0] data_stand;
  reg [13:0] data_stand = 14'd0;
  input reset_n;
  assign _1_ = !  reset_n;
  always @(posedge clock_50)
      data_stand <= _0_;
  assign _2_ = clock_recovery ?  data_rec : data_stand;
  assign _0_ = _1_ ?  14'h0000 : _2_;
endmodule

module design2(clock_recovery, reset_n, data_rec, clock_50, data_stand);
input  clock_recovery;
input  reset_n;
input [13:0] data_rec;
input  clock_50;
output [13:0] data_stand;
reg [13:0] data_stand = 14'd0;
wire [13:0] _0_;
wire  _1_;
wire [13:0] _2_;
always @(posedge clock_50)
    data_stand <= _0_;
assign _0_ = (_1_) ? (14'h0) : (_2_);
assign _1_ = !(reset_n);
assign _2_ = (clock_recovery) ? (data_rec) : (data_stand);
endmodule

module miter();
wire [13:0] data_stand1, data_stand2;
design1 inst1 (.clock_recovery(clock_recovery),.clock_50(clock_50),.reset_n(reset_n),.data_rec(data_rec),.data_stand(data_stand1));
design2 inst2 (.clock_recovery(clock_recovery),.clock_50(clock_50),.reset_n(reset_n),.data_rec(data_rec),.data_stand(data_stand2));
always @(posedge clk) begin
assert(data_stand1 == data_stand2);
end
endmodule