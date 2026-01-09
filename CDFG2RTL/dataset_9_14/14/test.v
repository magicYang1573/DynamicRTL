module blinker(input clk,input rst,output blink);
 
 reg [24:0] counter_q, counter_d; 
 reg dir; 
 assign blink = counter_d[24]; 
 always@(counter_q) begin 
  if (dir) begin 
  counter_d = counter_q - 1'b1; 
  end else begin 
  counter_d = counter_q + 1'b1; 
  end 
 end 
 always@(posedge clk) begin 
  if (rst) begin 
  counter_q <= 25'b0; 
  dir <= 1'b0; 
  end else begin 
  counter_q <= counter_d; 
  end 
 end 
 endmodule