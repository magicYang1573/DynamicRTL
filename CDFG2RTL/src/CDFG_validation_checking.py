import RTL_CDFG2RTL_equivalence_checking as Equiv_checker
import argparse
from glob import glob
import os
import time
from utils.logger import Logger
from tqdm import tqdm

def get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='dataset_9_14')

    args = parser.parse_args()
    return args

def get_logger():
    
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    if not os.path.exists('log/'):
        os.makedirs('log/')
    logger = Logger(f'log/log_{timestamp}.txt')
    return logger


if __name__ == '__main__':
    args = get_parse_args()
    logger = get_logger()
    design_dir_list = glob(os.path.join(args.dataset_dir, '*'))
    tot_design_num = len(design_dir_list)
    equivalent_design_num = 0
    design_checked_num = 0
    
    for design_path in design_dir_list:
        try:
            output = Equiv_checker.CDFG2RTL_equivalence_checking(design_path)
            if 'SUCCESS' in output:
                equiv_state = 'Equivalent'
                equivalent_design_num += 1
            else:
                equiv_state = 'NotEquivalent'
            result = f'[{design_checked_num:<4}/{tot_design_num - 1:<4}] design_name: {os.path.basename(design_path):<4}  {equiv_state}'
            print(result)
            logger.write(result + '\n')
            
        except Exception as e:
            error = f'[{design_checked_num:<4}/{tot_design_num - 1:<4}] design_name: {os.path.basename(design_path):<4}  error occur: {e}'
            print(error)
            logger.write(error + '\n')
        design_checked_num += 1
        
    print(f'{equivalent_design_num} designs are equivalent')

    