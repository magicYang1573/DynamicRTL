

module Rs232Tx_tb;

    reg [0:0] clk;
    reg [7:0] data;
    reg [0:0] send;
    wire [0:0] UART_TX;
    wire [0:0] sending;
    wire [0:0] uart_ovf;

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
    
    Rs232Tx dut (
        .clk(clk),
        .data(data),
        .send(send),
        .UART_TX(UART_TX),
        .sending(sending),
        .uart_ovf(uart_ovf)

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
      $fscanf(file, "%d,%d", data, send);
      #2;
    end
    $fclose(file);
    $finish;
  end

endmodule
    