module design1(clock, reset, count, data_o);
  wire [7:0] _0_;
  wire [31:0] _1_;
  wire [7:0] _2_;
  input clock;
  input count;
  output [7:0] data_o;
  reg [7:0] data_o = 8'd0;
  input reset;
  assign _1_ = data_o +  32'd1;
  always @(posedge clock)
      data_o <= _0_;
  assign _2_ = count ?  _1_[7:0] : data_o;
  assign _0_ = reset ?  8'h00 : _2_;
endmodule

module design2(reset, count, clock, data_o);
input  reset;
input  count;
input  clock;
output [7:0] data_o;
reg [7:0] data_o = 8'd0;
wire [31:0] _1_;
wire [7:0] _2_;
wire [7:0] _0_;
always @(posedge clock)
    data_o <= _0_;
assign _1_ = (data_o) + (32'd1);
assign _2_ = (count) ? (_1_[7:0]) : (data_o);
assign _0_ = (reset) ? (8'h0) : (_2_);
endmodule

module miter();
wire [7:0] data_o1, data_o2;
design1 inst1 (.clock(clock),.reset(reset),.count(count),.data_o(data_o1));
design2 inst2 (.clock(clock),.reset(reset),.count(count),.data_o(data_o2));
always @(posedge clk) begin
assert(data_o1 == data_o2);
end
endmodule