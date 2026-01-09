module Rs232Tx(clk, UART_TX, data, send, uart_ovf, sending);
  wire [9:0] _00_;
  wire _01_;
  wire [13:0] _02_;
  wire _03_;
  wire _04_;
  wire _05_;
  wire _06_;
  wire _07_;
  wire _08_;
  wire _09_;
  wire _10_;
  wire _11_;
  wire [13:0] _12_;
  wire [9:0] _13_;
  wire [9:0] _14_;
  wire [31:0] _15_;
  output UART_TX;
  input clk;
  input [7:0] data;
  input send;
  reg [9:0] sendbuf = 10'h001;
  output sending;
  reg sending;
  reg [13:0] timeout;
  output uart_ovf;
  reg uart_ovf;
  assign _04_ = timeout ==  32'd0;
  assign _05_ = sendbuf[8:0] ==  9'h001;
  assign _06_ = sending &&  _04_;
  assign _07_ = send &&  sending;
  assign _08_ = send &&  _09_;
  assign _09_ = !  sending;
  always @(posedge clk)
      sendbuf <= _00_;
  always @(posedge clk)
      timeout <= _02_;
  always @(posedge clk)
      uart_ovf <= _03_;
  always @(posedge clk)
      sending <= _01_;
  assign _10_ = _08_ ?  1'h1 : sending;
  assign _11_ = _05_ ?  1'h0 : _10_;
  assign _01_ = _06_ ?  _11_ : _10_;
  assign _03_ = _07_ ?  1'h1 : uart_ovf;
  assign _12_ = _08_ ?  14'h0063 : _15_[13:0];
  assign _02_ = _06_ ?  14'h0063 : _12_;
  assign _13_ = _08_ ?  { 1'h1, data, 1'h0 } : sendbuf;
  assign _14_ = _05_ ?  _13_ : { 1'h0, sendbuf[9:1] };
  assign _00_ = _06_ ?  _14_ : _13_;
  assign _15_ = timeout -  32'd1;
  assign UART_TX = sendbuf[0];
endmodule