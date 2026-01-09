module led_chaser (
    input wire clk,        // clock signal
    input wire rst,        // reset signal
    output reg [7:0] led   // 8-bit LED output
);

always @(posedge clk) begin
    if (rst) begin
        led <= 8'b00000001;   // on reset, light the first LED
    end else begin
        led <= {led[6:0], led[7]}; // on each rising clock edge, shift LED to the right
    end
end

endmodule

