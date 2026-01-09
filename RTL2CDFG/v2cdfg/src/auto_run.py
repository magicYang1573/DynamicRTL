import argparse
import subprocess
import os
from ast2cdfg import *


def arguments():
    parser = argparse.ArgumentParser(description="Example program for CLI arguments.")
    parser.add_argument('filename', type=str, help="File name to process.", nargs='?')
    parser.add_argument('-d', '--divide', action='store_true', help="Split basic blocks.")
    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    file_name = args.filename
    name, ext = os.path.splitext(file_name)

    # Read input file.
    input_directory = "./tmp/ast"
    output_directory = "./cdfg/"
    # Create directory.
    # output_directory = "../data/"+name
    os.makedirs(output_directory, exist_ok=True)

    input_path = os.path.join(input_directory, file_name)
    out_path1 = os.path.join(output_directory, name + "_cdfg.dot")
    out_path2 = os.path.join(output_directory, name + "_cdfg.png")
    out_path3 = os.path.join(output_directory, name + "_basic_block_combined.dot")
    data = read_json_file(input_path)
    ast = build_tree_from_json(data)

    # Preprocess: add suffix to identical keywords.
    pre(ast)

    # Extract CDFG; dfg is the extracted graph.
    dfg_extractor = DFGExtractor()
    dfg_extractor.visit(ast)

    dfg_extractor.Preprocessing()
    dfg_extractor.show_graph(filename=out_path1)
    dfg = dfg_extractor.dfg
    # print(dfg_extractor.resolve_var_exp())
    # print(dfg_extractor.node_dict.keys())
    # print(dfg.nodes)
    # print(dfg.edges)
    if args.divide:
        Subgraphs = dfg_extractor.basic_block(outpath=output_directory)
        dfg_extractor.show_basic_block_combine_graph(Subgraphs, out_path3)

    # CDFG visualization.
    # dot_path = "/mnt/c/ProgramFiles(x86)/Graphviz2.38/bin"

    # os.system("dot -Tpng {} -o {}".format(out_path1,out_path2))
