

module lp_fltr_tb;

    reg [0:0] ce;
    reg [0:0] clk;
    reg [7:0] din;
    wire [7:0] dout;

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
    
    lp_fltr dut (
        .ce(ce),
        .clk(clk),
        .din(din),
        .dout(dout)

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
      $fscanf(file, "%d,%d", ce, din);
      #2;
    end
    $fclose(file);
    $finish;
  end

endmodule
    