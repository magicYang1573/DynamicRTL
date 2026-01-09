

module image_generator_tb;

    reg [0:0] clk;
    reg [0:0] reset;
    wire [15:0] address;
    wire [0:0] load;
    wire [15:0] out;

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
    
    image_generator dut (
        .clk(clk),
        .reset(reset),
        .address(address),
        .load(load),
        .out(out)

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
      $fscanf(file, "%d", reset);
      #2;
    end
    $fclose(file);
    $finish;
  end

endmodule
    