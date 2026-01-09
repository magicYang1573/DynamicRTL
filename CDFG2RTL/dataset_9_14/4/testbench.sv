

module UpCounter_tb;

    reg [0:0] clock;
    reg [0:0] count;
    reg [0:0] reset;
    wire [7:0] data_o;

    `ifdef DUMP_TRACE 
        initial begin
            $dumpfile("dump.vcd");
            $dumpvars(0, dut);
        end
    `endif // DUMP_TRACE
    
        initial
            clock = 0;
        always begin
            #1 clock = ~clock;
        end
    
    UpCounter dut (
        .clock(clock),
        .count(count),
        .reset(reset),
        .data_o(data_o)

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
      $fscanf(file, "%d,%d", count, reset);
      #2;
    end
    $fclose(file);
    $finish;
  end

endmodule
    