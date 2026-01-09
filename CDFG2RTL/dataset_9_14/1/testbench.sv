

module vga_text_tb;

    reg [3:0] background;
    reg [7:0] char_data;
    reg [0:0] clk25mhz;
    reg [3:0] emphasized;
    reg [13:0] font_data;
    reg [9:0] hindex;
    reg [3:0] standard;
    reg [9:0] vindex;
    wire [11:0] char_address;
    wire [7:0] color;
    wire [9:0] font_address;

    `ifdef DUMP_TRACE 
        initial begin
            $dumpfile("dump.vcd");
            $dumpvars(0, dut);
        end
    `endif // DUMP_TRACE
    
        initial
            clk25mhz = 0;
        always begin
            #1 clk25mhz = ~clk25mhz;
        end
    
    vga_text dut (
        .background(background),
        .char_data(char_data),
        .clk25mhz(clk25mhz),
        .emphasized(emphasized),
        .font_data(font_data),
        .hindex(hindex),
        .standard(standard),
        .vindex(vindex),
        .char_address(char_address),
        .color(color),
        .font_address(font_address)

        );
    initial begin
    integer file;
    string dummy_line;
    file = $fopen("workload.in", "r");
    if (file == 0) begin
      $display("Error: could not open file workload.in");
      $finish;
    end
    $fscanf(file, "%s", dummy_line);  // Read and ignore the first line
    while (!$feof(file)) begin
      $fscanf(file, "%d,%d,%d,%d,%d,%d,%d", background, char_data, emphasized, font_data, hindex, standard, vindex);
      #2;
    end
    $fclose(file);
    $finish;
  end

endmodule
    