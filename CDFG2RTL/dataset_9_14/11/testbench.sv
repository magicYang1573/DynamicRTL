

module altera_up_slow_clock_generator_tb;

    reg [0:0] clk;
    reg [0:0] enable_clk;
    reg [0:0] reset;
    wire [0:0] falling_edge;
    wire [0:0] middle_of_high_level;
    wire [0:0] middle_of_low_level;
    wire [0:0] new_clk;
    wire [0:0] rising_edge;

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
    
    altera_up_slow_clock_generator dut (
        .clk(clk),
        .enable_clk(enable_clk),
        .reset(reset),
        .falling_edge(falling_edge),
        .middle_of_high_level(middle_of_high_level),
        .middle_of_low_level(middle_of_low_level),
        .new_clk(new_clk),
        .rising_edge(rising_edge)

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
      $fscanf(file, "%d,%d", enable_clk, reset);
      #2;
    end
    $fclose(file);
    $finish;
  end

endmodule
    