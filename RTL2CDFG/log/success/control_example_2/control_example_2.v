module control_example(input [7:0] a, input [7:0] b, input clk, input RESET, output reg [7:0] result);

  reg [7:0] temp;
  reg [7:0] intermediate;  // another register for complex computation

  always @(posedge clk) begin
    if (RESET) begin : Basic_Block_1
      result <= 8'h00;
      temp <= 8'h01;       // temp is initialized on reset
      intermediate <= 8'h02;
    end else begin
      if (a > b) begin : Basic_Block_2
        temp <= (a ^ b) + 8'h10;     // Basic Block 2: complex computation
        intermediate <= temp + a;    // temp participates in computation
        result <= intermediate - b;  // dependency between intermediate and result
      end else if (a < b) begin : Basic_Block_3
        temp <= (b & a) | 8'h55;     // Basic Block 3: complex computation
        intermediate <= temp - b;    // dependency between intermediate and temp
        result <= intermediate + a;  // result uses intermediate computation result
      end else begin : Basic_Block_4
        temp <= ~(a | b);            // Basic Block 4: complex computation
        intermediate <= temp ^ 8'hAA;  // dependency between intermediate and temp
        result <= intermediate;       // result uses intermediate computation result
      end
    end
  end

endmodule
