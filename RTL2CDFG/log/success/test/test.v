module dataflow_optimization(input [31:0] a, b, c, d, output [31:0] result);
    wire [31:0] t1, t2, t3, t4;

    assign t1 = a + b;   // common subexpression
    assign t2 = a + b;   // CSE: t1 and t2 are the same
    assign t3 = t1 + c;  // dataflow merge: reuse t1 
    assign t4 = t2 + d;

    assign result = t3 + t4; 
endmodule