import CDFG2RTL
import Miter_generator
import os
from glob import glob
import subprocess

# def run_yosys_sat(miter_path):
#     cmd = f'yosys -p "read_verilog -sv {miter_path}; hierarchy -top miter; proc; flatten; sat -tempinduct -prove-asserts -verify;"'
    
#     process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#     output = []
    
#     for line in process.stdout:
#         output.append(line)

#     process.wait()
#     stderr_output = process.stderr.read()
#     if stderr_output:
#         output.append(stderr_output)

#     return ''.join(output)


class SubprocessTimeoutException(Exception):
    """SubprocessTimeoutException"""
    pass

def run_yosys_sat(miter_path, timeout=15):
    cmd = f'yosys -p "read_verilog -sv {miter_path}; hierarchy -top miter; proc; flatten; sat -tempinduct -prove-asserts -verify;"'
    
    try:
        # start subprocess
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # communicate allows setting a timeout
        stdout, stderr = process.communicate(timeout=timeout)

        output = []
        if stdout:
            output.append(stdout)
        if stderr:
            output.append(stderr)
        
        return ''.join(output)
    
    except subprocess.TimeoutExpired:
        # if the subprocess exceeds the specified time, raise a timeout exception
        process.kill()  # terminate process
        raise SubprocessTimeoutException(f"SAT timed out after {timeout} seconds")


def syntax_check(verilog_path):
    cmd = f'iverilog {verilog_path}'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if 'error' in result.stderr.lower():
        return 'Syntax error'
    return 'Syntax correct'

def CDFG2RTL_equivalence_checking(design_path):
    origin_rtl_path = glob(os.path.join(design_path, 'module*.v'))
    origin_rtl_path = [path for path in origin_rtl_path if not 'genFromDot' in path and not '.info' in path][0]
    # gen rtl from cdfg
    dot_path = glob(os.path.join(design_path, 'cdfg', '*.dot'))[0]
    module_name = f'module_{os.path.basename(design_path)}'
    cdfg2rtler = CDFG2RTL.Cdfg2rtler(dot_path, module_name, original_design_path=origin_rtl_path)
    regen_rtl_path = os.path.join(design_path, f'module_{module_name}_genFromDot.v')
    cdfg2rtler.write_rtl(regen_rtl_path)
    
    if 'error' in syntax_check(origin_rtl_path) or 'error' in syntax_check(regen_rtl_path):
        raise Exception("Syntax error in RTL")
    
    # gen miter
    # miter_generator = Miter_generator.Miter_generator(origin_rtl_path, regen_rtl_path)
    miter_generator = Miter_generator.Miter_generator(origin_rtl_path, regen_rtl_path, force_same_IO=True)
    miter_path = os.path.join(design_path, 'miter.sv')
    miter_generator.write_miter(miter_path)
    
    output = run_yosys_sat(miter_path)
    
    return output


if __name__ == '__main__':
    design_path = 'dataset_9_14/494'
    print(CDFG2RTL_equivalence_checking(design_path))
    verilog_path = '/home/user/projects/CDFG2RTL/dataset_9_14/1/module_module_1_genFromDot.v'
    print(syntax_check(verilog_path))
