import argparse
import os
from clean_ast import *


def arguments():
    parser = argparse.ArgumentParser(description="Example program for CLI arguments.")
    parser.add_argument('filename', type=str, help="File name to process.", nargs='?')
    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    file_name = args.filename
    print(args)
    name, ext = os.path.splitext(file_name)

    input_file = name + ".v"
    output_file = name + "_ast.v"
    ast_clean_file = name + "_ast_clean.v"
    input_directory = "../verilog_data"  # "./yosys_data_1000"
    output_directory = "./tmp/yosys_output_data"

    input_path = os.path.join(input_directory, input_file)

    out_path = os.path.join(output_directory, output_file)
    # Open source and target files.
    with open(input_path, 'r') as source_file, open("test.v", 'w') as target_file:
        # Read source file content.
        content = source_file.read()

        # Write content to target file.
        target_file.write(content)

    print("Content written to target file successfully.")

    # yosys_directory = "/home/wangyipeng/oss-cad-suite/bin/"
    # os.system("yosys run_ast.ys".format(yosys_directory))
    os.system("yosys run_ast.ys")

    # Open source and target files.
    with open("test_ast.v", 'r') as source_file, open(out_path, 'w') as target_file:
        # Read source file content.
        content = source_file.read()

        # Write content to target file.
        target_file.write(content)

    print("Content written to target file successfully.")

    ast_clean_directory = "./tmp/yosys_output_data_clean"
    ast_clean_path = os.path.join(ast_clean_directory, ast_clean_file)
    vlg_clean(out_path, ast_clean_path)

    # test
    # vlg_clean("./test.v", ast_clean_path)
