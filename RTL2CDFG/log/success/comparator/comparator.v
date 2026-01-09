module comparator (
    p,
    a,
    b,
    clk,
);
  parameter integer N = 8;
  input [N-1:0] a, b;
  input clk;
  output p;

  // assign q = (a<b)?v2cdfg/yosys_data1/comparator.v1'b0:1'b1;
  always @(posedge clk) begin
    if (a < b) begin
      p = 1'b0;
    end else begin
      p = 1'b1;
    end
  end
endmodule
