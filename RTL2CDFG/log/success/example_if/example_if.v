module simple_if_example (
    input wire data_in,  // input data, width 4 bits
    output wire data_out       // output signal, width 1 bit
);

    // internal signal declarations for continuous assignments
    wire msb;  // most significant bit signal

    // continuous assignment, gets the MSB of data_in
    assign msb = data_in;

    // continuous assignment using if logic
    // Note: in practical Verilog, assign cannot directly include if-else statements,
    // so we use a ternary operator to achieve similar behavior.
    assign data_out = msb ? 1'b1 : 1'b0;


endmodule
