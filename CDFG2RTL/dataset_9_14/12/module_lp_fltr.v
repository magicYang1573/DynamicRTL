module lp_fltr(clk, din, dout, ce);
  wire [9:0] _00_;
  wire [9:0] _01_;
  wire [7:0] _02_;
  wire [7:0] _03_;
  wire [7:0] _04_;
  wire [7:0] _05_;
  wire _06_;
  reg [9:0] add_tmp_1;
  reg [9:0] add_tmp_2;
  input ce;
  input clk;
  input [7:0] din;
  reg [7:0] din_tmp_1;
  reg [7:0] din_tmp_2;
  reg [7:0] din_tmp_3;
  output [7:0] dout;
  reg [7:0] dout;
  reg [9:0] sum_tmp_1;
  reg [9:0] sum_tmp_2;
  reg [9:0] sum_tmp_3;
  assign _00_ = sum_tmp_1 +  sum_tmp_2;
  assign _01_ = add_tmp_1 +  sum_tmp_3;
  assign _06_ = ce ==  1'h1;
  always @(posedge clk)
      din_tmp_3 <= _04_;
  always @(posedge clk)
      sum_tmp_1 <= { din_tmp_1[7], din_tmp_1[7], din_tmp_1 };
  always @(posedge clk)
      sum_tmp_2 <= { din_tmp_2[7], din_tmp_2, 1'h0 };
  always @(posedge clk)
      din_tmp_1 <= _02_;
  always @(posedge clk)
      din_tmp_2 <= _03_;
  always @(posedge clk)
      dout <= _05_;
  always @(posedge clk)
      sum_tmp_3 <= { din_tmp_3[7], din_tmp_3[7], din_tmp_3 };
  always @(posedge clk)
      add_tmp_1 <= _00_;
  always @(posedge clk)
      add_tmp_2 <= _01_;
  assign _04_ = _06_ ?  din_tmp_2 : din_tmp_3;
  assign _05_ = _06_ ?  add_tmp_2[9:2] : dout;
  assign _03_ = _06_ ?  din_tmp_1 : din_tmp_2;
  assign _02_ = _06_ ?  din : din_tmp_1;
endmodule