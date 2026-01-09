module large_system (
    input wire clk,            // clock signal
    input wire rst_n,          // reset signal (active low)
    input wire [7:0] a,        // 8-bit input a
    input wire [7:0] b,        // 8-bit input b
    input wire start,          // start signal
    input wire add_en,         // add enable signal
    input wire mul_en,         // multiply enable signal
    output reg [15:0] result,  // 16-bit output result
    output reg [7:0] count,    // 8-bit output counter
    output reg done            // done signal
);

// state machine definition
reg [1:0] state, next_state;

// state values
localparam IDLE    = 2'b00,  // idle state
           ADD     = 2'b01,  // add state
           MUL     = 2'b10,  // multiply state
           DONE    = 2'b11;  // done state

// internal signals
reg [15:0] add_result;      // add result
reg [15:0] mul_result;      // multiply result
reg [7:0] internal_count;   // internal counter
reg done_reg;               // done flag register

// synchronous reset counter
always @(posedge clk) begin
    if (~rst_n)
        internal_count <= 8'b0;  // clear on reset
    else
        internal_count <= internal_count + 1;  // otherwise count
end

// synchronous reset state machine
always @(posedge clk) begin
    if (~rst_n)
        state <= IDLE;  // return to idle state on reset
    else
        state <= next_state;  // otherwise update state
end

// synchronous reset state transition
always @(*) begin
    case (state)
        IDLE: begin
            if (start)
                next_state = ADD; // enter add state after start
            else
                next_state = IDLE;
        end
        ADD: begin
            if (add_en)
                next_state = MUL; // enter multiply state after add completes
            else
                next_state = ADD;
        end
        MUL: begin
            if (mul_en)
                next_state = DONE; // enter done state after multiply completes
            else
                next_state = MUL;
        end
        DONE: begin
            next_state = IDLE; // return to idle state after done
        end
        default: next_state = IDLE;
    endcase
end

// synchronous reset add operation
always @(posedge clk) begin
    if (~rst_n)
        add_result <= 16'b0;
    else if (add_en)
        add_result <= a + b;  // perform add operation
end

// synchronous reset multiply operation
always @(posedge clk) begin
    if (~rst_n)
        mul_result <= 16'b0;
    else if (mul_en)
        mul_result <= a * b;  // perform multiply operation
end

// synchronous reset output register
always @(posedge clk) begin
    if (~rst_n) begin
        result <= 16'b0;
        done_reg <= 0;
    end else begin
        case (state)
            ADD: result <= add_result;   // output add result during add
            MUL: result <= mul_result;   // output multiply result during multiply
            DONE: done_reg <= 1;         // set done flag on completion
            default: done_reg <= 0;
        endcase
    end
end

// synchronous reset done signal
always @(posedge clk) begin
    if (~rst_n)
        done <= 0;
    else
        done <= done_reg;  // assign done flag to output signal
end

// synchronous reset counter output
always @(posedge clk) begin
    if (~rst_n)
        count <= 8'b0;
    else
        count <= internal_count;  // assign counter value to output signal
end

endmodule
