module control_example(input [7:0] a, input [7:0] b, input clk, input RESET, output reg [7:0] result);
  
  reg [7:0] temp;

  always @(posedge clk) begin
    if (RESET) begin : Basic_Block_1
      result <= 8'h00;  // Basic Block 1: two statements
      temp <= 8'h00;
    end else begin
      if (a > b) begin : Basic_Block_2
        result <= a - b; // Basic Block 2: two statements
        temp <= a;
      end else if (a < b) begin : Basic_Block_3
        result <= b - a; // Basic Block 3: two statements
        temp <= b;
      end else begin : Basic_Block_4
        result <= 8'hFF; // Basic Block 4: two statements
        temp <= 8'hFF;
      end
    end
  end

endmodule
