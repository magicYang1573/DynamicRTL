from anytree import NodeMixin,AnyNode,RenderTree
from anytree.importer import JsonImporter
import json

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
        
def build_tree_from_json(data, level = 1, parent = None):
    operator = data['name']
    node = AnyNode(name = operator,level = level, parent = parent)
    for child_data in data.get('children', []):
        child_node = build_tree_from_json(child_data,level = level+1, parent = node)
    return node

