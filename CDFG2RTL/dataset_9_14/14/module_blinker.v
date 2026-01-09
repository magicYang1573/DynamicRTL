module blinker(clk, rst, blink);
  wire [24:0] _00_;
  wire _01_;
  wire [24:0] _02_;
  wire [24:0] _03_;
  output blink;
  input clk;
  wire [24:0] counter_d;
  reg [24:0] counter_q;
  reg dir;
  input rst;
  assign _02_ = counter_q +  1'h1;
  always @(posedge clk)
      dir <= _01_;
  always @(posedge clk)
      counter_q <= _00_;
  assign counter_d = dir ?  _03_ : _02_;
  assign _00_ = rst ?  25'h0000000 : counter_d;
  assign _01_ = rst ?  1'h0 : dir;
  assign _03_ = counter_q -  1'h1;
  assign blink = counter_d[24];
endmodule