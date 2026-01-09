import argparse
import os
import subprocess
import sys
from anytree.exporter import JsonExporter, UniqueDotExporter
from anytree.importer import JsonImporter

original_stdout = sys.stdout

sys.path.append(os.path.abspath('src'))
import tree
import ast2cdfg
import clean_ast
import ast_from_json
import ast_pre


LOG_FILE = "../log/log.txt"


def arguments():
    parser = argparse.ArgumentParser(description="Example program for CLI arguments.")
    parser.add_argument('filename', type=str, help="File name to process.", nargs='?')
    return parser.parse_args()


def yosys(origin_filename):
    name = os.path.splitext(os.path.basename(origin_filename))[0]
    output_file = name + "_ast.v"
    ast_clean_file = name + "_ast_clean.v"
    output_directory = "./tmp/yosys_output_data"

    out_path = os.path.join(output_directory, output_file)
    # Open source and target files.
    with open(origin_filename, 'r') as source_file, open("test.v", 'w') as target_file:
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
    clean_ast.vlg_clean(out_path, ast_clean_path)


def parser(origin_filename, input_filename, output_filename):
    print(input_filename)
    print(output_filename)
    # Use absolute paths to ensure correct file reading.
    # input_file = os.path.abspath(f"./tmp/yosys_output_data_clean/{input_filename}")
    # output_file = os.path.abspath(f"./tmp/s_exp/{output_filename}")
    command = [
        "../Stagira/stagira.exe",
        "-toegg",
        f"./tmp/yosys_output_data_clean/{input_filename}",
    ]
    with open(f"./tmp/s_exp/{output_filename}", "w") as output_file:
        try:
            # Use subprocess.run with a 120-second timeout.
            result = subprocess.run(command, timeout=120, check=True, stdout=output_file)
            print("S-expression generated successfully.")

        except subprocess.TimeoutExpired:
            print("Command timed out: exceeded 120 seconds.")
            with open(LOG_FILE, "a") as log_file:
                log_file.write(f"{origin_filename} : Parser failed (timeout)\n")

        except subprocess.CalledProcessError as e:
            print(f"{origin_filename} : Parser failed (non-zero exit code {e.returncode})")
            with open(LOG_FILE, "a") as log_file:
                log_file.write(f"{origin_filename} : Parser failed\n")


def ast(origin_filename, input_filename, output_filename):
    input_path = os.path.join("./tmp/s_exp", input_filename)
    ast = tree.read_file(input_path, True)
    with open("operatorlist.txt", "w+") as file:
        operatorlist = list(set(tree.operatorlist))
        for operator in operatorlist:
            file.write(str(operator) + ',')
    output_path = os.path.join("./tmp/ast", output_filename)
    with open(output_path, 'w') as f:
        sys.stdout = f
        exporter = JsonExporter(indent=2, sort_keys=True)
        print(exporter.export(ast))
    sys.stdout = original_stdout
    print("AST JSON file generated.")
    dot_filename = f"{origin_filename}_ast_clean.dot"
    dot_path = os.path.join("./tmp/ast", dot_filename)
    importer = JsonImporter()
    with open(output_path, 'r') as file:
        root = importer.read(file)
        # graphviz needs to be installed for the next line!
        # UniqueDotExporter(root).to_picture(output_path2)
        # UniqueDotExporter(root).to_dotfile(output_path4)
        # subprocess.check_call(['dot', output_path4, '-T', 'png', '-o', output_path2])
        UniqueDotExporter(root).to_dotfile(dot_path)
        # subprocess.check_call(['dot', output_path4, '-T', 'svg', '-o', output_path2])
    sys.stdout = original_stdout
    print("AST dot file generated.")


def cdfg(origin_filename, input_filename):
    input_path = os.path.join("./tmp/ast", input_filename)
    out_path = os.path.join("./cdfg", origin_filename + "_cdfg.dot")
    data = ast_from_json.read_json_file(input_path)
    ast = ast_from_json.build_tree_from_json(data)
    # Preprocess.
    ast_pre.pre(ast)
    # Extract CDFG; dfg is the extracted graph.
    dfg_extractor = ast2cdfg.DFGExtractor()
    dfg_extractor.visit(ast)
    dfg_extractor.Preprocessing()
    dfg_extractor.show_graph(filename=out_path)


def generator(filename):
    # Extract file name and extension with os.path.splitext.
    name_without_extension = os.path.splitext(os.path.basename(filename))[0]

    # Generate new file names based on the extracted name.
    filename_s = f"{name_without_extension}.s"
    new_filename_v = f"{name_without_extension}_ast_clean.v"
    new_filename_s = f"{name_without_extension}_ast_clean.s"
    new_filename_j = f"{name_without_extension}_ast_clean.json"

    # yosys
    yosys(args.filename)
    # Parser syntax analysis.
    parser(name_without_extension, new_filename_v, new_filename_s)
    # AST generation.
    ast(name_without_extension, new_filename_s, new_filename_j)
    # CDFG generation.
    cdfg(name_without_extension, new_filename_j)

if __name__ == "__main__":
    args = arguments()

    generator(args.filename)
