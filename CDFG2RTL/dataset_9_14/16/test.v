module program_counter (next_pc,cur_pc,rst,clk);
 
  output [0:31] next_pc; 
  input [0:31] cur_pc; 
  input clk; 
  input rst; 
  reg [0:31] next_pc; 
  always @(posedge clk) 
  begin 
  if(rst) 
  begin 
  next_pc<=32'd0; 
  end 
  else 
  begin 
  next_pc<=cur_pc+32'd4; 
  end 
  end 
 endmodule