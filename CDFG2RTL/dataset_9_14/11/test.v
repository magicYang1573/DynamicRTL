module altera_up_slow_clock_generator (clk,reset,enable_clk,new_clk,rising_edge,falling_edge,middle_of_high_level,middle_of_low_level);
 
 parameter CB = 10; 
 input clk; 
 input reset; 
 input enable_clk; 
 output reg new_clk; 
 output reg rising_edge; 
 output reg falling_edge; 
 output reg middle_of_high_level; 
 output reg middle_of_low_level; 
 reg [CB:1] clk_counter; 
 always @(posedge clk) 
 begin 
  if (reset) 
  clk_counter <= 'h0; 
  else if (enable_clk) 
  clk_counter <= clk_counter + 1; 
 end 
 always @(posedge clk) 
 begin 
  if (reset) 
  new_clk <= 1'b0; 
  else 
  new_clk <= clk_counter[CB]; 
 end 
 always @(posedge clk) 
 begin 
  if (reset) 
  rising_edge <= 1'b0; 
  else 
  rising_edge <= (clk_counter[CB] ^ new_clk) & ~new_clk; 
 end 
 always @(posedge clk) 
 begin 
  if (reset) 
  falling_edge <= 1'b0; 
  else 
  falling_edge <= (clk_counter[CB] ^ new_clk) & new_clk; 
 end 
 always @(posedge clk) 
 begin 
  if (reset) 
  middle_of_high_level <= 1'b0; 
  else 
  middle_of_high_level <= 
  clk_counter[CB] & 
  ~clk_counter[(CB - 1)] & 
  (&(clk_counter[(CB - 2):1])); 
 end 
 always @(posedge clk) 
 begin 
  if (reset) 
  middle_of_low_level <= 1'b0; 
  else 
  middle_of_low_level <= 
  ~clk_counter[CB] & 
  ~clk_counter[(CB - 1)] & 
  (&(clk_counter[(CB - 2):1])); 
 end 
 endmodule