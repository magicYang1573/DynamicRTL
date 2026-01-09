module UpCounter(clock,reset,count,data_o);
 
  parameter Size = 8; 
  input wire [('b1) - ('b1):0] clock; 
  input wire [('b1) - ('b1):0] reset; 
  input wire [('b1) - ('b1):0] count; 
  output reg [(Size) - ('b1):0] data_o; 
  always @ (posedge clock) begin 
  if (reset) begin 
  data_o <= {Size{1'b0}}; 
  end 
  else begin 
  if (count) begin 
  data_o <= data_o + 1; 
  end 
  end 
  end 
 endmodule 