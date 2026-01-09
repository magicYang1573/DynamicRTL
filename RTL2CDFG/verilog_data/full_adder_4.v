// Top Level Verilog code for N-bit Ripple Carry Full Adder using Structural Modeling
module full_adder (
    a,
    b,
    cin,
    s,
    cout
);
  parameter integer N = 4;          // N-bit adder
  input [N-1:0] a, b;               // N-bit inputs a and b
  input cin;                        // input carry
  output [N-1:0] s;                 // N-bit sum
  output cout;                      // output final carry

  wire [N:0] carry;                 // carry chain
  genvar i;                         // generate variable
  generate
    assign carry[0] = cin;          // initial carry equals cin

    for (i = 0; i < N; i = i + 1) begin : generate_N_bit_Adder
      assign s[i] = (a[i] ^ b[i]) ^ carry[i];   // compute sum of each bit
      assign carry[i+1] = (b[i] & carry[i]) | (a[i] & b[i]) | (a[i] & carry[i]); // compute carry
    end

    assign cout = carry[N];         // final carry output
  endgenerate
endmodule
