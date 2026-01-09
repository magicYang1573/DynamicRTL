

module nios_system_sram_addr_tb;

    reg [1:0] address;
    reg [0:0] chipselect;
    reg [0:0] clk;
    reg [0:0] reset_n;
    reg [0:0] write_n;
    reg [31:0] writedata;
    wire [10:0] out_port;
    wire [31:0] readdata;

    `ifdef DUMP_TRACE 
        initial begin
            $dumpfile("dump.vcd");
            $dumpvars(0, dut);
        end
    `endif // DUMP_TRACE
    
        initial
            clk = 0;
        always begin
            #1 clk = ~clk;
        end
    
    nios_system_sram_addr dut (
        .address(address),
        .chipselect(chipselect),
        .clk(clk),
        .reset_n(reset_n),
        .write_n(write_n),
        .writedata(writedata),
        .out_port(out_port),
        .readdata(readdata)

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
      $fscanf(file, "%d,%d,%d,%d,%d", address, chipselect, reset_n, write_n, writedata);
      #2;
    end
    $fclose(file);
    $finish;
  end

endmodule
    