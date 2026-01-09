

module timer_tb;

    reg [29:0] ADR_I;
    reg [0:0] CLK_I;
    reg [0:0] CYC_I;
    reg [0:0] RST_I;
    reg [0:0] STB_I;
    reg [0:0] WE_I;
    wire [0:0] RTY_O;
    wire [0:0] interrupt_o;

    `ifdef DUMP_TRACE 
        initial begin
            $dumpfile("dump.vcd");
            $dumpvars(0, dut);
        end
    `endif // DUMP_TRACE
    
        initial
            CLK_I = 0;
        always begin
            #1 CLK_I = ~CLK_I;
        end
    
    timer dut (
        .ADR_I(ADR_I),
        .CLK_I(CLK_I),
        .CYC_I(CYC_I),
        .RST_I(RST_I),
        .STB_I(STB_I),
        .WE_I(WE_I),
        .RTY_O(RTY_O),
        .interrupt_o(interrupt_o)

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
      $fscanf(file, "%d,%d,%d,%d,%d", ADR_I, CYC_I, RST_I, STB_I, WE_I);
      #2;
    end
    $fclose(file);
    $finish;
  end

endmodule
    