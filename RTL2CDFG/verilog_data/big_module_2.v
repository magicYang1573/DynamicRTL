module large_system (
    input wire clk,                  // clock signal
    input wire rst_n,                // reset signal (active low)
    input wire start,                // start signal
    input wire [7:0] a,              // 8-bit input a
    input wire [7:0] b,              // 8-bit input b
    input wire add_en,               // add enable signal
    input wire mul_en,               // multiply enable signal
    input wire [15:0] fifo_in,       // FIFO input data
    output reg [15:0] result,        // 16-bit output result
    output reg [7:0] count,          // 8-bit counter output
    output reg done,                 // done signal
    output wire [15:0] fifo_out      // FIFO output data
);

// state machine definition
reg [3:0] state, next_state;

// state values
localparam IDLE    = 4'b0001,  // idle state
           ADD     = 4'b0010,  // add state
           MUL     = 4'b0100,  // multiply state
           DONE    = 4'b1000;  // done state

// internal signals
reg [15:0] add_result;            // add result
reg [15:0] mul_result;            // multiply result
reg [7:0] internal_count;         // internal counter
reg done_reg;                     // done flag register
reg [15:0] fifo_mem [0:15];       // 16-depth FIFO memory
reg [3:0] fifo_write_ptr;         // FIFO write pointer
reg [3:0] fifo_read_ptr;          // FIFO read pointer

// FIFO read/write logic
assign fifo_out = fifo_mem[fifo_read_ptr]; // output FIFO data

// FIFO write operation
always @(posedge clk) begin
    if (~rst_n)
        fifo_write_ptr <= 4'b0; // clear write pointer on reset
    else if (start)
        fifo_mem[fifo_write_ptr] <= fifo_in; // write input data to FIFO
end

// FIFO read operation
always @(posedge clk) begin
    if (~rst_n)
        fifo_read_ptr <= 4'b0;  // clear read pointer on reset
    else if (start)
        fifo_read_ptr <= fifo_read_ptr + 1; // increment read pointer
end

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
