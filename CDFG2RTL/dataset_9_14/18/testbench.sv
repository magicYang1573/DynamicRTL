

module execute_flag_register_tb;

    reg [4:0] iADDER_FLAG;
    reg [0:0] iADDER_VALID;
    reg [0:0] iCLOCK;
    reg [0:0] iCTRL_HOLD;
    reg [4:0] iLOGIC_FLAG;
    reg [0:0] iLOGIC_VALID;
    reg [4:0] iMUL_FLAG;
    reg [0:0] iMUL_VALID;
    reg [4:0] iPFLAGR;
    reg [0:0] iPFLAGR_VALID;
    reg [0:0] iPREV_BUSY;
    reg [0:0] iPREV_FLAG_WRITE;
    reg [0:0] iPREV_INST_VALID;
    reg [0:0] iRESET_SYNC;
    reg [4:0] iSHIFT_FLAG;
    reg [0:0] iSHIFT_VALID;
    reg [0:0] inRESET;
    wire [4:0] oFLAG;

    `ifdef DUMP_TRACE 
        initial begin
            $dumpfile("dump.vcd");
            $dumpvars(0, dut);
        end
    `endif // DUMP_TRACE
    
        initial
            iCLOCK = 0;
        always begin
            #1 iCLOCK = ~iCLOCK;
        end
    
    execute_flag_register dut (
        .iADDER_FLAG(iADDER_FLAG),
        .iADDER_VALID(iADDER_VALID),
        .iCLOCK(iCLOCK),
        .iCTRL_HOLD(iCTRL_HOLD),
        .iLOGIC_FLAG(iLOGIC_FLAG),
        .iLOGIC_VALID(iLOGIC_VALID),
        .iMUL_FLAG(iMUL_FLAG),
        .iMUL_VALID(iMUL_VALID),
        .iPFLAGR(iPFLAGR),
        .iPFLAGR_VALID(iPFLAGR_VALID),
        .iPREV_BUSY(iPREV_BUSY),
        .iPREV_FLAG_WRITE(iPREV_FLAG_WRITE),
        .iPREV_INST_VALID(iPREV_INST_VALID),
        .iRESET_SYNC(iRESET_SYNC),
        .iSHIFT_FLAG(iSHIFT_FLAG),
        .iSHIFT_VALID(iSHIFT_VALID),
        .inRESET(inRESET),
        .oFLAG(oFLAG)

        );
    initial begin
    integer file;
    string dummy_line;
    file = $fopen("workload.in", "r");
    if (file == 0) begin
      $display("Error: could not open file workload.in");
      $finish;
    end
    $fscanf(file, "%s", dummy_line);  // Read and ignore the first line
    while (!$feof(file)) begin
      $fscanf(file, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d", iADDER_FLAG, iADDER_VALID, iCTRL_HOLD, iLOGIC_FLAG, iLOGIC_VALID, iMUL_FLAG, iMUL_VALID, iPFLAGR, iPFLAGR_VALID, iPREV_BUSY, iPREV_FLAG_WRITE, iPREV_INST_VALID, iRESET_SYNC, iSHIFT_FLAG, iSHIFT_VALID, inRESET);
      #2;
    end
    $fclose(file);
    $finish;
  end

endmodule
    