module three_state_fsm (
    input wire clk,         // clock signal
    input wire rst,         // reset signal
    input wire in,          // input signal
    output reg [1:0] state  // current state output
);

// state encoding
localparam STATE0 = 2'b00;
localparam STATE1 = 2'b01;
localparam STATE2 = 2'b10;

// state machine logic
always @(posedge clk) begin
    if (rst) begin
        state <= STATE0;  // on reset, state machine returns to IDLE state
    end else begin
        // state transition logic
        if (state == STATE0) begin
            if (in)
                state <= STATE1;  // if input is high, transition from STATE0 to STATE1
        end else if (state == STATE1) begin
            if (in)
                state <= STATE2;  // if input is high, transition from STATE1 to STATE2
        end else if (state == STATE2) begin
            if (in)
                state <= STATE0;  // if input is high, transition from STATE2 to STATE0
        end else begin
            state <= STATE0;
        end
    end
end

endmodule
