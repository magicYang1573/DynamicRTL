module design1(clk, din, dout, ce);
  wire [9:0] _00_;
  wire [9:0] _01_;
  wire [7:0] _02_;
  wire [7:0] _03_;
  wire [7:0] _04_;
  wire [7:0] _05_;
  wire _06_;
  reg [9:0] add_tmp_1 = 10'd0;
  reg [9:0] add_tmp_2 = 10'd0;
  input ce;
  input clk;
  input [7:0] din;
  reg [7:0] din_tmp_1 = 8'd0;
  reg [7:0] din_tmp_2 = 8'd0;
  reg [7:0] din_tmp_3 = 8'd0;
  output [7:0] dout;
  reg [7:0] dout = 8'd0;
  reg [9:0] sum_tmp_1 = 10'd0;
  reg [9:0] sum_tmp_2 = 10'd0;
  reg [9:0] sum_tmp_3 = 10'd0;
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

module design2(ce, din, clk, dout);
input  ce;
input [7:0] din;
input  clk;
output [7:0] dout;
reg [9:0] sum_tmp_2 = 10'd0;
reg [9:0] add_tmp_2 = 10'd0;
reg [9:0] add_tmp_1 = 10'd0;
reg [9:0] sum_tmp_1 = 10'd0;
reg [9:0] sum_tmp_3 = 10'd0;
reg [7:0] din_tmp_3 = 8'd0;
reg [7:0] din_tmp_1 = 8'd0;
reg [7:0] dout = 8'd0;
reg [7:0] din_tmp_2 = 8'd0;
wire  _06_;
wire [7:0] _04_;
wire [9:0] _00_;
wire [9:0] _01_;
wire [7:0] _05_;
wire [7:0] _02_;
wire [7:0] _03_;
always @(posedge clk)
    sum_tmp_2 <= {din_tmp_2[7:7], din_tmp_2, 1'h0};
always @(posedge clk)
    add_tmp_2 <= _01_;
always @(posedge clk)
    add_tmp_1 <= _00_;
always @(posedge clk)
    sum_tmp_1 <= {din_tmp_1[7:7], din_tmp_1[7:7], din_tmp_1};
always @(posedge clk)
    sum_tmp_3 <= {din_tmp_3[7:7], din_tmp_3[7:7], din_tmp_3};
always @(posedge clk)
    din_tmp_3 <= _04_;
always @(posedge clk)
    din_tmp_1 <= _02_;
always @(posedge clk)
    dout <= _05_;
always @(posedge clk)
    din_tmp_2 <= _03_;
assign _06_ = (ce) == (1'h1);
assign _04_ = (_06_) ? (din_tmp_2) : (din_tmp_3);
assign _00_ = (sum_tmp_1) + (sum_tmp_2);
assign _01_ = (add_tmp_1) + (sum_tmp_3);
assign _05_ = (_06_) ? (add_tmp_2[9:2]) : (dout);
assign _02_ = (_06_) ? (din) : (din_tmp_1);
assign _03_ = (_06_) ? (din_tmp_1) : (din_tmp_2);
endmodule

module miter();
wire [7:0] dout1, dout2;
design1 inst1 (.clk(clk),.din(din),.dout(dout1),.ce(ce));
design2 inst2 (.clk(clk),.din(din),.dout(dout2),.ce(ce));
always @(posedge clk) begin
assert(dout1 == dout2);
end
endmodule