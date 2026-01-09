module module_9(reset, clk, load, out, address);
input  reset;
input  clk;
output  load;
output [15:0] out;
output [15:0] address;
reg [32:0] counter;
reg [15:0] data;
reg [15:0] buffer_addr;
reg  wren;
wire [31:0] _07_;
wire  _13_;
wire [32:0] _01_;
wire [15:0] _02_;
wire  _06_;
wire [32:0] _04_;
wire [15:0] _11_;
wire [32:0] _10_;
wire  _09_;
wire [15:0] _00_;
wire [15:0] _08_;
wire  _03_;
wire [32:0] _05_;
wire [15:0] _12_;
always @(posedge clk)
    counter <= _01_;
always @(posedge clk)
    data <= _02_;
always @(posedge clk)
    buffer_addr <= _00_;
always @(posedge clk)
    wren <= _03_;
assign _07_ = (buffer_addr) % (32'd50);
assign _13_ = |(_07_);
assign _01_ = (reset) ? (33'h0) : (_10_);
assign _02_ = (reset) ? (16'h0) : (_08_);
assign _06_ = (counter) == (32'd384000);
assign _04_ = (counter) + (32'd1);
assign _11_ = (_06_) ? (buffer_addr) : (_05_[15:0]);
assign _10_ = (_06_) ? (33'h0) : (_04_);
assign _09_ = (_06_) ? (1'h0) : (wren);
assign _00_ = (reset) ? (buffer_addr) : (_11_);
assign _08_ = (_06_) ? (data) : (_12_);
assign _03_ = (reset) ? (1'h1) : (_09_);
assign _05_ = (counter) / (32'd16);
assign _12_ = (_13_) ? (16'h0) : (16'hffff);
assign load = wren;
assign out = data;
assign address = buffer_addr;
endmodule