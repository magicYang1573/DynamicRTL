module write_axi(clock_recovery, clock_50, reset_n, data_rec, data_stand);
  wire [13:0] _0_;
  wire _1_;
  wire [13:0] _2_;
  input clock_50;
  input clock_recovery;
  input [13:0] data_rec;
  output [13:0] data_stand;
  reg [13:0] data_stand;
  input reset_n;
  assign _1_ = !  reset_n;
  always @(posedge clock_50)
      data_stand <= _0_;
  assign _2_ = clock_recovery ?  data_rec : data_stand;
  assign _0_ = _1_ ?  14'h0000 : _2_;
endmodule