

module write_axi_tb;

    reg [0:0] clock_50;
    reg [0:0] clock_recovery;
    reg [13:0] data_rec;
    reg [0:0] reset_n;
    wire [13:0] data_stand;

    `ifdef DUMP_TRACE 
        initial begin
            $dumpfile("dump.vcd");
            $dumpvars(0, dut);
        end
    `endif // DUMP_TRACE
    
        initial
            clock_50 = 0;
        always begin
            #1 clock_50 = ~clock_50;
        end
    
    write_axi dut (
        .clock_50(clock_50),
        .clock_recovery(clock_recovery),
        .data_rec(data_rec),
        .reset_n(reset_n),
        .data_stand(data_stand)

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
      $fscanf(file, "%d,%d,%d", clock_recovery, data_rec, reset_n);
      #2;
    end
    $fclose(file);
    $finish;
  end

endmodule
    