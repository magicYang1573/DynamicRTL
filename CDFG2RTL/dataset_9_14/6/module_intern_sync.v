module intern_sync(clk, rstn, rc_is_idle, rc_reqn, rc_ackn);
  wire [1:0] _00_;
  wire _01_;
  wire [1:0] _02_;
  wire _03_;
  wire _04_;
  wire _05_;
  wire [1:0] _06_;
  wire _07_;
  wire _08_;
  wire _09_;
  wire _10_;
  wire _11_;
  wire [1:0] _12_;
  input clk;
  output rc_ackn;
  input rc_is_idle;
  input rc_reqn;
  input rstn;
  reg [1:0] state_c;
  wire [1:0] state_n;
  assign _03_ = ~  rc_reqn;
  assign _04_ = ~  rstn;
  always @(posedge clk)
      state_c <= _00_;
  assign _01_ = _05_ ?  _11_ : 1'hx;
  assign _05_ = state_c ==  2'h1;
  assign _06_ = rc_is_idle ?  2'h0 : 2'h1;
  assign _02_ = _07_ ?  _06_ : 2'hx;
  assign _07_ = state_c ==  2'h1;
  function [1:0] _21_;
    input [1:0] a;
    input [3:0] b;
    input [1:0] s;
    casez (s)
      2'b?1:
        _21_ = b[1:0];
      2'b1?:
        _21_ = b[3:2];
      default:
        _21_ = a;
    endcase
  endfunction
  assign state_n = _21_(2'h0, { _12_, _02_ }, { _09_, _08_ });
  assign _08_ = state_c ==  2'h1;
  assign _09_ = state_c ==  2'h0;
  assign rc_ackn = _10_ ?  _01_ : 1'h1;
  assign _10_ = state_c ==  2'h1;
  assign _00_ = _04_ ?  2'h0 : state_n;
  assign _11_ = rc_is_idle ?  1'h0 : 1'h1;
  assign _12_ = _03_ ?  2'h1 : 2'h0;
endmodule