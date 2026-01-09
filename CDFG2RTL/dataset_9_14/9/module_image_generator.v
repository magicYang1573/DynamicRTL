module image_generator(clk, reset, address, out, load);
  wire [15:0] _00_;
  wire [32:0] _01_;
  wire [15:0] _02_;
  wire _03_;
  wire [32:0] _04_;
  wire [32:0] _05_;
  wire _06_;
  wire [31:0] _07_;
  wire [15:0] _08_;
  wire _09_;
  wire [32:0] _10_;
  wire [15:0] _11_;
  wire [15:0] _12_;
  wire _13_;
  output [15:0] address;
  reg [15:0] buffer_addr;
  input clk;
  reg [32:0] counter;
  reg [15:0] data;
  output load;
  output [15:0] out;
  input reset;
  reg wren;
  assign _04_ = counter +  32'd1;
  assign _05_ = counter /  32'd16;
  assign _06_ = counter ==  32'd384000;
  assign _07_ = buffer_addr %  32'd50;
  always @(posedge clk)
      buffer_addr <= _00_;
  always @(posedge clk)
      counter <= _01_;
  always @(posedge clk)
      wren <= _03_;
  always @(posedge clk)
      data <= _02_;
  assign _08_ = _06_ ?  data : _12_;
  assign _02_ = reset ?  16'h0000 : _08_;
  assign _09_ = _06_ ?  1'h0 : wren;
  assign _03_ = reset ?  1'h1 : _09_;
  assign _10_ = _06_ ?  33'h000000000 : _04_;
  assign _01_ = reset ?  33'h000000000 : _10_;
  assign _11_ = _06_ ?  buffer_addr : _05_[15:0];
  assign _00_ = reset ?  buffer_addr : _11_;
  assign _12_ = _13_ ?  16'h0000 : 16'hffff;
  assign _13_ = |  _07_;
  assign address = buffer_addr;
  assign load = wren;
  assign out = data;
endmodule