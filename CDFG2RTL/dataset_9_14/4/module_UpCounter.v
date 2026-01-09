module UpCounter(clock, reset, count, data_o);
  wire [7:0] _0_;
  wire [31:0] _1_;
  wire [7:0] _2_;
  input clock;
  input count;
  output [7:0] data_o;
  reg [7:0] data_o;
  input reset;
  assign _1_ = data_o +  32'd1;
  always @(posedge clock)
      data_o <= _0_;
  assign _2_ = count ?  _1_[7:0] : data_o;
  assign _0_ = reset ?  8'h00 : _2_;
endmodule