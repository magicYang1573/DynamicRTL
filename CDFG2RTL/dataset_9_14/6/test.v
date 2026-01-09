module intern_sync (input clk ,input rstn ,input rc_is_idle,input rc_reqn,output reg rc_ackn);
 
  reg [1:0] state_c, state_n; 
  localparam [1:0] 
  RC_Idle = 4'd0, 
  RC_ReqAck = 4'd1; 
  always @(posedge clk) begin 
  if (~rstn) 
  state_c <= RC_Idle; 
  else 
  state_c <= state_n; 
  end 
  always @(*) begin 
  {rc_ackn} = {1'b1}; 
  case (state_c) 
  RC_Idle: begin state_n = (~rc_reqn)? RC_ReqAck:RC_Idle; end 
  RC_ReqAck:begin 
  if(rc_is_idle) begin 
  state_n = RC_Idle; 
  rc_ackn = 1'b0; 
  end else begin 
  state_n = RC_ReqAck; 
  end 
  end 
  default: begin state_n = RC_Idle; end 
  endcase 
  end 
 endmodule