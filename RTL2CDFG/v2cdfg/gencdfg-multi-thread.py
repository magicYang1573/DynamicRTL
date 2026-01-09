import os, time, json
from multiprocessing import Pool
import concurrent.futures

#################
## global vars ##
#################
# verilog_dir = '/data/ruiyang/small_verilog_data'    # dir to store the original verilog codes
verilog_dir = '../verilog_data'

success_data_dir = '../log/success'   # dir to store success verilog and its generated cdfg
fail_data_dir = '../log/fail'
overtime_data_dir = '../log/overtime'

# for every verilog file in verilog_dir, process this verilog
# and generate a dir in data_dir to store the log
total_count = 0
success_count = 0
fail_count = 0
overtime_count = 0
yosys_timeout = 10
gencdfg_timeout = 10

# convert a verilog file to CDFG
# this will create a running dir to each verilog
# inorder to be applied concurrently
def process_file(filename):
    global verilog_dir, data_dir, success_data_dir, fail_data_dir
    global total_count, success_count, fail_count, overtime_count
    global yosys_timeout, gencdfg_timeout

    running_dir = ''
    run_ast_verilog = ''
    run_clean_ast_verilog = ''
    output_data_dir = ''

    yosys_success_flag = 1
    cdfg_success_flag = 1
    if filename.endswith(".v"):
        verilog_path = os.path.join(verilog_dir, filename)
        design_name = filename[:-2]  # remove .v extension
        
        # create running dir for a thread
        running_dir = os.path.join("../", 'run-' + design_name)
        log_dir = os.path.join(running_dir,"output.log")
        os.system('cp -dr ../v2cdfg {}'.format(running_dir))

        # run_verilog_path = os.path.join(running_dir,filename)
        # os.system('cp {} {}'.format(verilog_path, run_verilog_path))
        
        # yosys_success_flag = os.system('cd {} && timeout {}s yosys run_ast.ys >> yosys.log 2>&1'.format(running_dir, yosys_timeout))
        # if yosys_success_flag==0:
        exit_code = os.system('cd {} && ./cdfg_generator.sh -nodivide {}'.format(running_dir,filename))
            # os.system('cd {} && python3 auto_yosys.py  > /dev/null 2>&1'.format(running_dir))

            # name = "test"
            # cmd = "ast_clean"
            # run_ast_verilog = os.path.join(running_dir, "test_ast.v")
            # clean_ast_verilog = "../test_ast_clean.v"
            # run_clean_ast_verilog = os.path.join(running_dir, "test_ast_clean.v")

            # # Call the function to convert the design
            # output_dir = "../example"
            # # convert_one_design(clean_ast_verilog, name, cmd, output_dir)
            # cdfg_success_flag = os.system(f'cd {running_dir} && cd vlg2ir && timeout {gencdfg_timeout}s python3 analyze.py {clean_ast_verilog} -N {name} -C {cmd} -O {output_dir} >> ../gencdfg.log 2>&1')
    else:
        return

    # Generate a dir in data_dir to store the log
    # if yosys_success_flag==0 and cdfg_success_flag==0:
    #     success_count += 1
    #     output_data_dir = os.path.join(success_data_dir, design_name)
    # elif yosys_success_flag==31744 or cdfg_success_flag==31744:
    #     overtime_count += 1
    #     output_data_dir = os.path.join(overtime_data_dir, design_name)
    # else:
    #     fail_count += 1
    #     output_data_dir = os.path.join(fail_data_dir, design_name)
    if exit_code == 0:
        success_count += 1
        output_data_dir = os.path.join(success_data_dir, design_name)
    else:
        fail_count += 1
        output_data_dir = os.path.join(fail_data_dir, design_name)
    
    os.makedirs(output_data_dir, exist_ok=True)
    os.system('cp {} {}  > /dev/null 2>&1'.format(verilog_path, output_data_dir))
    os.system('cp {} {}  > /dev/null 2>&1'.format(log_dir, output_data_dir))
    # os.system('cp {} {}  > /dev/null 2>&1'.format(run_ast_verilog, output_data_dir))
    # os.system('cp {} {}  > /dev/null 2>&1'.format(run_clean_ast_verilog, output_data_dir))
    os.system('cp -r {}/tmp/yosys_output_data_clean {} > /dev/null 2>&1'.format(running_dir, output_data_dir))
    os.system('cp -r {}/tmp/ast {} > /dev/null 2>&1'.format(running_dir, output_data_dir))
    # os.system('cd {}/cdfg && cp *.png {} > /dev/null 2>&1'.format(running_dir, output_data_dir))
    # os.system('cd {}/S_data && cp *.png {} > /dev/null 2>&1'.format(running_dir, output_data_dir))
    # os.system('cd {} && cp *.log {}  > /dev/null 2>&1'.format(running_dir, output_data_dir))
    # os.system('cd {} && cp *.pkl {}  > /dev/null 2>&1'.format(running_dir, output_data_dir))
    os.system('cp -r {}/cdfg {}  > /dev/null 2>&1'.format(running_dir, output_data_dir))
    

    total_count += 1
    print('{} finished. Process total {}, Success {}, Failed {}, Overtime {}'.format(filename, total_count, success_count, fail_count, overtime_count))

    # delete the running dir
    os.system('rm -rf {}'.format(running_dir))


if __name__ == '__main__':

    # delete previous data
    os.system('cd ../log && rm -rf *')
    os.system('cd .. && rm -rf run-* && cd ./v2cdfg')
    os.system('cd cdfg && rm -rf * && cd ..')
    os.system('cd tmp/ast && rm -rf * && cd .. && cd ..')
    os.system('cd tmp/s_exp && rm -rf * && cd .. && cd ..')
    os.system('cd tmp/yosys_output_data && rm -rf * && cd .. && cd ..')
    os.system('cd tmp/yosys_output_data_clean && rm -rf * && cd .. && cd ..')
    if os.path.isfile("test.v"):
        ()# os.system('rm test.v && rm test_ast.v')
    else:
        print("no such file")
    
    ############
    ## Serial ##
    ############
    # for filename in os.listdir(verilog_dir):
    #     process_file(filename)
    
    ##############
    ## Parallel ##
    ##############
    thread_num = 128
    read_num = 500 #4096
    cnt = 0  
    cnt_total = 0
    # setup thread pool
    read_list = []
    for filename in os.listdir(verilog_dir):
        read_list.append(filename)
        cnt += 1
        cnt_total += 1
        if cnt == read_num:
            cnt = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
                futures = []
                for f in read_list:
                    futures.append(executor.submit(process_file, f))
                concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
            read_list = []
        # if cnt_total==32:
        #     break
    # Process remaining files.
    if read_list:
        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
            futures = [executor.submit(process_file, f) for f in read_list]
            concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
    
    
