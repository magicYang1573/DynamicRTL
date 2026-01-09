module write_axi(input clock_recovery,input clock_50,input reset_n,input [13:0] data_rec,output reg [13:0] data_stand);
 
 always@(posedge clock_50) 
 begin 
  if(!reset_n) 
  begin 
  data_stand <= 14'd0; 
  end 
  else 
  begin 
  if(clock_recovery) 
  data_stand <= data_rec; 
  else 
  data_stand <= data_stand; 
  end 
 end 
 endmodule