module altera_up_slow_clock_generator(clk, reset, enable_clk, new_clk, rising_edge, falling_edge, middle_of_high_level, middle_of_low_level);
  wire [9:0] _00_;
  wire _01_;
  wire _02_;
  wire _03_;
  wire _04_;
  wire _05_;
  wire [31:0] _06_;
  wire _07_;
  wire _08_;
  wire _09_;
  wire _10_;
  wire _11_;
  wire _12_;
  wire _13_;
  wire _14_;
  wire _15_;
  wire _16_;
  wire [9:0] _17_;
  wire _18_;
  wire _19_;
  wire _20_;
  wire _21_;
  input clk;
  reg [10:1] clk_counter;
  input enable_clk;
  output falling_edge;
  reg falling_edge;
  output middle_of_high_level;
  reg middle_of_high_level;
  output middle_of_low_level;
  reg middle_of_low_level;
  output new_clk;
  reg new_clk;
  input reset;
  output rising_edge;
  reg rising_edge;
  assign _06_ = clk_counter +  32'd1;
  assign _07_ = _20_ &  _13_;
  assign _08_ = _21_ &  new_clk;
  assign _09_ = clk_counter[10] &  _14_;
  assign _10_ = _09_ &  _18_;
  assign _11_ = _15_ &  _16_;
  assign _12_ = _11_ &  _19_;
  assign _13_ = ~  new_clk;
  assign _14_ = ~  clk_counter[9];
  assign _15_ = ~  clk_counter[10];
  assign _16_ = ~  clk_counter[9];
  always @(posedge clk)
      middle_of_low_level <= _03_;
  always @(posedge clk)
      middle_of_high_level <= _02_;
  always @(posedge clk)
      falling_edge <= _01_;
  always @(posedge clk)
      rising_edge <= _05_;
  always @(posedge clk)
      new_clk <= _04_;
  always @(posedge clk)
      clk_counter <= _00_;
  assign _03_ = reset ?  1'h0 : _12_;
  assign _02_ = reset ?  1'h0 : _10_;
  assign _01_ = reset ?  1'h0 : _08_;
  assign _05_ = reset ?  1'h0 : _07_;
  assign _04_ = reset ?  1'h0 : clk_counter[10];
  assign _17_ = enable_clk ?  _06_[9:0] : clk_counter;
  assign _00_ = reset ?  10'h000 : _17_;
  assign _18_ = &  clk_counter[8:1];
  assign _19_ = &  clk_counter[8:1];
  assign _20_ = clk_counter[10] ^  new_clk;
  assign _21_ = clk_counter[10] ^  new_clk;
endmodule