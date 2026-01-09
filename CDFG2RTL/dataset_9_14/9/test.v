module image_generator(clk,reset,address,out,load);
 
  input clk; 
  input reset; 
  output load; 
  output [15:0] address; 
  output [15:0] out; 
  reg [15:0] data; 
  reg [15:0] buffer_addr; 
  reg [32:0] counter; 
  reg wren; 
  always @(posedge clk) begin 
  if (reset) begin 
  data <= 0; 
  counter <= 0; 
  wren <= 1; 
  end else begin 
  if (counter == 384000) begin 
  counter <= 0; 
  wren <= 0; 
  end 
  else begin 
  counter <= counter + 1; 
  buffer_addr <= counter / 16; 
  if (buffer_addr % 50) begin 
  data <= 16'h00_00; 
  end 
  else begin 
  data <= 16'hFF_FF; 
  end 
  end 
  end 
  end 
  assign load = wren; 
  assign address = buffer_addr; 
  assign out = data; 
 endmodule