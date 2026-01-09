module ALU (
    input [31:0] A,          
    input [31:0] B,          
    input [1:0] ALUOp,  
    output reg [31:0] Result, 
);

always @(*) begin
    if (ALUOp == 2'b00) 
        Result <= A + B;       
    else if (ALUOp == 2'b01) 
        Result <= A - B;       
    else if (ALUOp == 2'b10) 
        Result <= A & B;       
    else 
        Result <= A % B;      
end

endmodule
