

module program_counter_tb;

    reg [0:0] clk;
    reg [31:0] cur_pc;
    reg [0:0] rst;
    wire [31:0] next_pc;

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
    
    program_counter dut (
        .clk(clk),
        .cur_pc(cur_pc),
        .rst(rst),
        .next_pc(next_pc)

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
      $fscanf(file, "%d,%d", cur_pc, rst);
      #2;
    end
    $fclose(file);
    $finish;
  end

endmodule
    