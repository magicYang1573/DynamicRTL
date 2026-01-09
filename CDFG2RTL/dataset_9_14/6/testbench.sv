

module intern_sync_tb;

    reg [0:0] clk;
    reg [0:0] rc_is_idle;
    reg [0:0] rc_reqn;
    reg [0:0] rstn;
    wire [0:0] rc_ackn;

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
    
    intern_sync dut (
        .clk(clk),
        .rc_is_idle(rc_is_idle),
        .rc_reqn(rc_reqn),
        .rstn(rstn),
        .rc_ackn(rc_ackn)

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
      $fscanf(file, "%d,%d,%d", rc_is_idle, rc_reqn, rstn);
      #2;
    end
    $fclose(file);
    $finish;
  end

endmodule
    