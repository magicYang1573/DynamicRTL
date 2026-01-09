module soc_design_JTAG_sim_scfifo_r (
  clk,          
  fifo_rd,      
  rst_n,        
  fifo_EF,      
  fifo_rdata,   
  rfifo_full,   
  rfifo_used    
);
output           fifo_EF;
output  [  7: 0] fifo_rdata;
output           rfifo_full;
output  [  5: 0] rfifo_used;
input            clk;
input            fifo_rd;
input            rst_n;
reg     [ 31: 0] bytes_left;
wire             fifo_EF;
reg              fifo_rd_d;      
wire    [  7: 0] fifo_rdata;
wire             new_rom;        
wire    [ 31: 0] num_bytes;      
wire    [  6: 0] rfifo_entries;  
wire             rfifo_full;
wire    [  5: 0] rfifo_used;
always @(posedge clk)
begin
  if (rst_n == 0)
    begin
      bytes_left <= 32'h0;
      fifo_rd_d <= 1'b0;
    end
  else
    begin
      fifo_rd_d <= fifo_rd;
      if (fifo_rd_d)
          bytes_left <= bytes_left - 1'b1;
      if (new_rom)
          bytes_left <= num_bytes;
    end
end
assign fifo_EF = bytes_left == 32'b0;
assign rfifo_full = bytes_left > 7'h40;
assign rfifo_entries = (rfifo_full) ? 7'h40 : bytes_left;
assign rfifo_used = rfifo_entries[5 : 0];
assign new_rom = 1'b0;      
assign num_bytes = 32'b0;   
assign fifo_rdata = 8'b0;   
endmodule