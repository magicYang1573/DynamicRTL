

module blinker_tb;

    reg [0:0] clk;
    reg [0:0] rst;
    wire [0:0] blink;

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
    
    blinker dut (
        .clk(clk),
        .rst(rst),
        .blink(blink)

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
      $fscanf(file, "%d", rst);
      #2;
    end
    $fclose(file);
    $finish;
  end

endmodule
    