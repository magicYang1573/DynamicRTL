

module ramcard_tb;

    reg [15:0] addr;
    reg [0:0] mclk28;
    reg [0:0] reset_in;
    reg [0:0] we;
    wire [0:0] bank1;
    wire [0:0] card_ram_rd;
    wire [0:0] card_ram_we;
    wire [17:0] ram_addr;

    `ifdef DUMP_TRACE 
        initial begin
            $dumpfile("dump.vcd");
            $dumpvars(0, dut);
        end
    `endif // DUMP_TRACE
    
        initial
            mclk28 = 0;
        always begin
            #1 mclk28 = ~mclk28;
        end
    
    ramcard dut (
        .addr(addr),
        .mclk28(mclk28),
        .reset_in(reset_in),
        .we(we),
        .bank1(bank1),
        .card_ram_rd(card_ram_rd),
        .card_ram_we(card_ram_we),
        .ram_addr(ram_addr)

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
      $fscanf(file, "%d,%d,%d", addr, reset_in, we);
      #2;
    end
    $fclose(file);
    $finish;
  end

endmodule
    