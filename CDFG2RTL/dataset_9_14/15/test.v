module timer(input CLK_I,input RST_I,input [31:2] ADR_I,input CYC_I,input STB_I,input WE_I,output reg RTY_O,output reg interrupt_o);
 
 reg [27:0] counter; 
 always @(posedge CLK_I) begin 
  if(RST_I == 1'b1) begin 
  RTY_O <= 1'b0; 
  interrupt_o <= 1'b0; 
  counter <= 28'd0; 
  end 
  else if(counter == 28'h00FFFFF) begin 
  if(ADR_I == { 27'b1111_1111_1111_1111_1111_1111_111, 3'b001 } && CYC_I == 1'b1 && STB_I == 1'b1 && WE_I == 1'b0 && interrupt_o == 1'b1) begin 
  RTY_O <= 1'b1; 
  interrupt_o <= 1'b0; 
  counter <= 28'd0; 
  end 
  else begin 
  interrupt_o <= 1'b1; 
  end 
  end 
  else begin 
  RTY_O <= 1'b0; 
  counter <= counter + 28'd1; 
  end 
 end 
 endmodule