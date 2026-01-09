module design1(CLK_I, RST_I, ADR_I, CYC_I, STB_I, WE_I, RTY_O, interrupt_o);
  wire _00_;
  wire [27:0] _01_;
  wire _02_;
  wire [27:0] _03_;
  wire _04_;
  wire _05_;
  wire _06_;
  wire _07_;
  wire _08_;
  wire _09_;
  wire _10_;
  wire _11_;
  wire _12_;
  wire _13_;
  wire _14_;
  wire [27:0] _15_;
  wire [27:0] _16_;
  wire _17_;
  wire _18_;
  wire _19_;
  wire _20_;
  input [31:2] ADR_I;
  input CLK_I;
  input CYC_I;
  input RST_I;
  output RTY_O;
  reg RTY_O = 1'd0;
  input STB_I;
  input WE_I;
  reg [27:0] counter = 28'd0;
  output interrupt_o;
  reg interrupt_o = 1'd0;
  assign _03_ = counter +  28'h0000001;
  assign _04_ = counter ==  28'h00fffff;
  assign _05_ = interrupt_o ==  1'h1;
  assign _06_ = ADR_I ==  30'h3ffffff9;
  assign _07_ = CYC_I ==  1'h1;
  assign _08_ = STB_I ==  1'h1;
  assign _09_ = WE_I ==  1'h0;
  assign _10_ = RST_I ==  1'h1;
  assign _11_ = _14_ &&  _09_;
  assign _12_ = _11_ &&  _05_;
  assign _13_ = _06_ &&  _07_;
  assign _14_ = _13_ &&  _08_;
  always @(posedge CLK_I)
      RTY_O <= _00_;
  always @(posedge CLK_I)
      interrupt_o <= _02_;
  always @(posedge CLK_I)
      counter <= _01_;
  assign _15_ = _12_ ?  28'h0000000 : counter;
  assign _16_ = _04_ ?  _15_ : _03_;
  assign _01_ = _10_ ?  28'h0000000 : _16_;
  assign _17_ = _12_ ?  1'h0 : 1'h1;
  assign _18_ = _04_ ?  _17_ : interrupt_o;
  assign _02_ = _10_ ?  1'h0 : _18_;
  assign _19_ = _12_ ?  1'h1 : RTY_O;
  assign _20_ = _04_ ?  _19_ : 1'h0;
  assign _00_ = _10_ ?  1'h0 : _20_;
endmodule

module design2(ADR_I, WE_I, STB_I, RST_I, CYC_I, CLK_I, interrupt_o, RTY_O);
input [29:0] ADR_I;
input  WE_I;
input  STB_I;
input  RST_I;
input  CYC_I;
input  CLK_I;
output  interrupt_o;
output  RTY_O;
reg  interrupt_o = 1'd0;
reg  RTY_O = 1'd0;
reg [27:0] counter = 28'd0;
wire  _11_;
wire  _10_;
wire  _17_;
wire [27:0] _01_;
wire [27:0] _15_;
wire  _00_;
wire  _04_;
wire [27:0] _16_;
wire  _13_;
wire  _06_;
wire  _02_;
wire  _19_;
wire  _07_;
wire  _05_;
wire  _12_;
wire  _20_;
wire  _09_;
wire  _08_;
wire  _14_;
wire [27:0] _03_;
wire  _18_;
always @(posedge CLK_I)
    interrupt_o <= _02_;
always @(posedge CLK_I)
    RTY_O <= _00_;
always @(posedge CLK_I)
    counter <= _01_;
assign _11_ = (_14_) && (_09_);
assign _10_ = (RST_I) == (1'h1);
assign _17_ = (_12_) ? (1'h0) : (1'h1);
assign _01_ = (_10_) ? (28'h0) : (_16_);
assign _15_ = (_12_) ? (28'h0) : (counter);
assign _00_ = (_10_) ? (1'h0) : (_20_);
assign _04_ = (counter) == (28'h00fffff);
assign _16_ = (_04_) ? (_15_) : (_03_);
assign _13_ = (_06_) && (_07_);
assign _06_ = (ADR_I) == (30'h3ffffff9);
assign _02_ = (_10_) ? (1'h0) : (_18_);
assign _19_ = (_12_) ? (1'h1) : (RTY_O);
assign _07_ = (CYC_I) == (1'h1);
assign _05_ = (interrupt_o) == (1'h1);
assign _12_ = (_11_) && (_05_);
assign _20_ = (_04_) ? (_19_) : (1'h0);
assign _09_ = (WE_I) == (1'h0);
assign _08_ = (STB_I) == (1'h1);
assign _14_ = (_13_) && (_08_);
assign _03_ = (counter) + (28'h1);
assign _18_ = (_04_) ? (_17_) : (interrupt_o);
endmodule

module miter();
wire  RTY_O1, RTY_O2;
wire  interrupt_o1, interrupt_o2;
design1 inst1 (.CLK_I(CLK_I),.RST_I(RST_I),.ADR_I(ADR_I),.CYC_I(CYC_I),.STB_I(STB_I),.WE_I(WE_I),.RTY_O(RTY_O1),.interrupt_o(interrupt_o1));
design2 inst2 (.CLK_I(CLK_I),.RST_I(RST_I),.ADR_I(ADR_I),.CYC_I(CYC_I),.STB_I(STB_I),.WE_I(WE_I),.RTY_O(RTY_O2),.interrupt_o(interrupt_o2));
always @(posedge clk) begin
assert(RTY_O1 == RTY_O2);
assert(interrupt_o1 == interrupt_o2);
end
endmodule