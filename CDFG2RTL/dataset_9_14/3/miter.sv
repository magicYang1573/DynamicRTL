module design1(address, chipselect, clk, reset_n, write_n, writedata, out_port, readdata);
  wire [10:0] _00_;
  wire _01_;
  wire _02_;
  wire _03_;
  wire _04_;
  wire _05_;
  wire _06_;
  wire [10:0] _07_;
  input [1:0] address;
  input chipselect;
  input clk;
  wire clk_en;
  reg [10:0] data_out = 11'd0;
  output [10:0] out_port;
  wire [10:0] read_mux_out;
  output [31:0] readdata;
  input reset_n;
  input write_n;
  input [31:0] writedata;
  assign read_mux_out = { _01_, _01_, _01_, _01_, _01_, _01_, _01_, _01_, _01_, _01_, _01_ } &  data_out;
  assign _01_ = address ==  32'd0;
  assign _02_ = reset_n ==  32'd0;
  assign _03_ = address ==  32'd0;
  assign _04_ = chipselect &&  _06_;
  assign _05_ = _04_ &&  _03_;
  assign _06_ = ~  write_n;
  assign readdata = 32'd0 |  read_mux_out;
  always @(posedge clk)
      data_out <= _00_;
  assign _07_ = _05_ ?  writedata[10:0] : data_out;
  assign _00_ = _02_ ?  11'h000 : _07_;
  assign clk_en = 1'h1;
  assign out_port = data_out;
endmodule

module design2(writedata, chipselect, write_n, reset_n, address, clk, out_port, readdata);
input [31:0] writedata;
input  chipselect;
input  write_n;
input  reset_n;
input [1:0] address;
input  clk;
output [10:0] out_port;
output [31:0] readdata;
reg [10:0] data_out = 11'd0;
wire  _01_;
wire  _04_;
wire [10:0] read_mux_out;
wire  clk_en;
wire  _06_;
wire  _02_;
wire [10:0] _07_;
wire  _05_;
wire [10:0] _00_;
wire  _03_;
always @(posedge clk)
    data_out <= _00_;
assign _01_ = (address) == (32'd0);
assign _04_ = (chipselect) && (_06_);
assign read_mux_out = ({_01_, _01_, _01_, _01_, _01_, _01_, _01_, _01_, _01_, _01_, _01_}) & (data_out);
assign clk_en = 1'h1;
assign _06_ = !(write_n);
assign _02_ = (reset_n) == (32'd0);
assign _07_ = (_05_) ? (writedata[10:0]) : (data_out);
assign _05_ = (_04_) && (_03_);
assign _00_ = (_02_) ? (11'h0) : (_07_);
assign _03_ = (address) == (32'd0);
assign out_port = data_out;
assign readdata = (32'd0) | (read_mux_out);
endmodule

module miter();
wire [10:0] out_port1, out_port2;
wire [31:0] readdata1, readdata2;
design1 inst1 (.address(address),.chipselect(chipselect),.clk(clk),.reset_n(reset_n),.write_n(write_n),.writedata(writedata),.out_port(out_port1),.readdata(readdata1));
design2 inst2 (.address(address),.chipselect(chipselect),.clk(clk),.reset_n(reset_n),.write_n(write_n),.writedata(writedata),.out_port(out_port2),.readdata(readdata2));
always @(posedge clk) begin
assert(out_port1 == out_port2);
assert(readdata1 == readdata2);
end
endmodule