import json
import os

if not os.getcwd() == os.path.abspath(__file__):
    print('[INFO] Change working directory to the directory of this file')
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
operator_file = 'operators.txt'
output_file = 'op_to_index.json'
with open(operator_file, 'r') as f:
    content = f.readlines()
content = [item.strip() for item in content]
op_dict = {item: id for id, item in enumerate(content)}

with open(output_file, 'w') as f:
    json.dump(op_dict, f, indent=4)