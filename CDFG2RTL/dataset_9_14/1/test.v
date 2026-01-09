module vga_text(clk25mhz,hindex,vindex,standard,emphasized,background,char_data,font_data,char_address,font_address,color);
 
 input wire clk25mhz; 
 input wire[9:0] hindex; 
 input wire[9:0] vindex; 
 input wire[13:0] font_data; 
 input wire[7:0] char_data; 
 input wire[3:0] standard; 
 input wire[3:0] emphasized; 
 input wire[3:0] background; 
 output wire[7:0] color; 
 output reg[9:0] font_address; 
 output reg[11:0] char_address = 12'b000000000000; 
 reg[2:0] char_col = 3'b000; 
 reg[3:0] char_row = 4'b0000; 
 reg[2:0] dark = 3'b000; 
 reg[2:0] bright = 3'b000; 
 reg[3:0] foreground; 
 wire[3:0] pixel_color; 
 assign color[7] = dark[2]; 
 assign color[6] = bright[2]; 
 assign color[5] = 0; 
 assign color[4] = dark[1]; 
 assign color[3] = bright[1]; 
 assign color[2] = 0; 
 assign color[1] = dark[0]; 
 assign color[0] = bright[0]; 
 assign pixel_color = (font_data[char_row] == 1) ? foreground : background; 
 always @ (posedge clk25mhz) begin: char_addressing 
  if(vindex >= 2 && vindex < 478) begin 
  if(hindex < 640) begin 
  if(char_col == 7) begin 
  char_col <= 0; 
  char_address <= char_address + 1; 
  end else begin 
  char_col <= char_col + 1; 
  end 
  end else if(hindex == 640) begin 
  if(char_row == 13) begin 
  char_row <= 0; 
  if(char_address == 2720) begin 
  char_address <= 0; 
  end 
  end else begin 
  char_row <= char_row + 1; 
  char_address <= char_address - 80; 
  end 
  end 
  end 
 end 
 always @ (posedge clk25mhz) begin: font_addressing 
  font_address <= {char_data[6:0], char_col}; 
  foreground <= (char_data[7] == 1) ? emphasized : standard; 
 end 
 always @ (posedge clk25mhz) begin: indexes 
  if(hindex > 0 && hindex < 641 && vindex < 480) begin 
  dark <= pixel_color[2:0]; 
  if(pixel_color[3] == 1) begin 
  if(pixel_color[2:0] == 0) begin 
  bright <= 3'b111; 
  end else begin 
  bright <= pixel_color[2:0]; 
  end 
  end else begin 
  bright <= 3'b000; 
  end 
  end else begin 
  dark <= 3'b000; 
  bright <= 3'b000; 
  end 
 end 
 endmodule