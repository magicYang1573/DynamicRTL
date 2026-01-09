module module_11(enable_clk, reset, clk, middle_of_high_level, middle_of_low_level, new_clk, rising_edge, falling_edge);
input  enable_clk;
input  reset;
input  clk;
output  middle_of_high_level;
output  middle_of_low_level;
output  new_clk;
output  rising_edge;
output  falling_edge;
reg [9:0] clk_counter;
reg  middle_of_high_level;
reg  middle_of_low_level;
reg  new_clk;
reg  rising_edge;
reg  falling_edge;
wire  _11_;
wire  _10_;
wire  _01_;
wire  _04_;
wire  _16_;
wire  _21_;
wire  _13_;
wire [9:0] _17_;
wire  _02_;
wire  _19_;
wire  _07_;
wire [31:0] _06_;
wire  _15_;
wire  _05_;
wire [9:0] _00_;
wire  _12_;
wire  _20_;
wire  _09_;
wire  _08_;
wire  _03_;
wire  _14_;
wire  _18_;
always @(posedge clk)
    clk_counter <= _00_;
always @(posedge clk)
    middle_of_high_level <= _02_;
always @(posedge clk)
    middle_of_low_level <= _03_;
always @(posedge clk)
    new_clk <= _04_;
always @(posedge clk)
    rising_edge <= _05_;
always @(posedge clk)
    falling_edge <= _01_;
assign _11_ = (_15_) & (_16_);
assign _10_ = (_09_) & (_18_);
assign _01_ = (reset) ? (1'h0) : (_08_);
assign _04_ = (reset) ? (1'h0) : (clk_counter[10:10]);
assign _16_ = !(clk_counter[9:9]);
assign _21_ = (clk_counter[10:10]) ^ (new_clk);
assign _13_ = !(new_clk);
assign _17_ = (enable_clk) ? (_06_[9:0]) : (clk_counter);
assign _02_ = (reset) ? (1'h0) : (_10_);
assign _19_ = &(clk_counter[8:1]);
assign _07_ = (_20_) & (_13_);
assign _06_ = (clk_counter) + (32'd1);
assign _15_ = !(clk_counter[10:10]);
assign _05_ = (reset) ? (1'h0) : (_07_);
assign _00_ = (reset) ? (10'h0) : (_17_);
assign _12_ = (_11_) & (_19_);
assign _20_ = (clk_counter[10:10]) ^ (new_clk);
assign _09_ = (clk_counter[10:10]) & (_14_);
assign _08_ = (_21_) & (new_clk);
assign _03_ = (reset) ? (1'h0) : (_12_);
assign _14_ = !(clk_counter[9:9]);
assign _18_ = &(clk_counter[8:1]);
endmodule