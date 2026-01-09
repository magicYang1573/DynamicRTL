import re
import os


class Miter_generator:
    def __init__(self, design_path1, design_path2, force_same_IO=False):
        with open(design_path1) as f:
            self.design1 = f.read()
        with open(design_path2) as f:
            self.design2 = f.read()
        self.design1_module_header = self.get_module_header(self.design1)
        self.design2_module_header = self.get_module_header(self.design2)
        self.design1_module_name = self.get_module_name_from_header(self.design1_module_header)
        self.design2_module_name = self.get_module_name_from_header(self.design2_module_header)
        self.header_var_list1 = self.get_header_var_list(self.design1_module_header)
        self.header_var_list2 = self.get_header_var_list(self.design2_module_header)
        
        self.input_width_dict = self.get_input_width_dict(1)
        self.output_width_dict = self.get_output_width_dict(1)
        
        if not set(self.header_var_list1) == set(self.header_var_list2):
            if force_same_IO:
                print('[WARNING] The two design modules have different input output lists, the program will force them to have the same IO')
                self.force_same_IO()
            else:
                raise Exception(f"The two design modules have different input output lists: {sorted(self.header_var_list1)} {sorted(self.header_var_list2)}")
                
        
    def force_same_IO(self):
        def regen_IO_def(text):
            # find insert id
            text_lines = text.split(';')
            for id in range(len(text_lines)):
                text_lines[id] = text_lines[id].strip()
            id_module = 0
            for id in range(len(text_lines)):
                if text_lines[id].startswith('module '):
                    id_module = id
                    break
            insert_id = id_module + 1
            
            # remove old IO
            id = 0
            while(id < len(text_lines)):
                if text_lines[id].startswith('input') or text_lines[id].startswith('output'):
                    text_lines.pop(id)
                    id -= 1
                id += 1
            
            # insert input
            for input in self.input_width_dict:
                text_lines.insert(insert_id, f'input [{self.input_width_dict[input] - 1}:0] {input}')
                
            # insert output
            for output in self.output_width_dict:
                text_lines.insert(insert_id, f'output [{self.output_width_dict[output] - 1}:0] {output}')
            
            text = ';\n'.join(text_lines)
            
            return text
            
        
        self.input_width_dict.update(self.get_input_width_dict(2))
        self.output_width_dict.update(self.get_output_width_dict(2))
        new_module_IO_list = ','.join(list(self.input_width_dict.keys()) + list(self.output_width_dict.keys()))
        new_design1_module_header = f'module {self.design1_module_name}({new_module_IO_list});'
        new_design2_module_header = f'module {self.design2_module_name}({new_module_IO_list});'
        
        self.design1 = self.design1.replace(self.design1_module_header, new_design1_module_header)
        self.design2 = self.design2.replace(self.design2_module_header, new_design2_module_header)
        
        self.design1 = regen_IO_def(self.design1)
        self.design2 = regen_IO_def(self.design2)
        
        self.design1_module_header = self.get_module_header(self.design1)
        self.design2_module_header = self.get_module_header(self.design2)
        self.design1_module_name = self.get_module_name_from_header(self.design1_module_header)
        self.design2_module_name = self.get_module_name_from_header(self.design2_module_header)
        self.header_var_list1 = self.get_header_var_list(self.design1_module_header)
        self.header_var_list2 = self.get_header_var_list(self.design2_module_header)
        self.input_width_dict = self.get_input_width_dict(1)
        self.output_width_dict = self.get_output_width_dict(1)
        
    
        
    def get_module_header(self, text):
        pattern = r'module\s+[^\(\s]+\s*(?:#\s*\(.*\)){0,1}\s*\(.*?\);'
        verilog_code = re.findall(pattern, text, re.DOTALL)[0]
        return verilog_code
    
    def get_module_name_from_header(self, header):
        pattern = r'module\s+([^\(\s]+)'
        module_name = re.findall(pattern, header, re.DOTALL)[0]
        return module_name
        
    def get_header_var_list(self, header):
        pattern = r'[^#]\s*\((.*)\)'
        input_list = re.findall(pattern, header, re.DOTALL)[0]
        input_list = input_list.split(',')
        input_list = [x.strip() for x in input_list]
        return input_list
    
    def parse_input(self, line):
        def get_width(text):
            text = text.replace(']','').replace('[','').replace(' ','')
            num_list = text.split(':')
            return abs(int(num_list[0]) - int(num_list[1])) + 1
        if '[' in line:
            pattern = r'input\s+([^\s]+)\s+([^\s]+)'
            match = re.search(pattern, line)
            width_text = match.group(1)
            input_name = match.group(2)
            input_width = get_width(width_text)
        else:
            pattern = r'input\s+([^\s]+)'
            match = re.search(pattern, line)
            input_name = match.group(1)
            input_width = 1
        return input_name, input_width
    
    def parse_output(self, line):
        def get_width(text):
            text = text.replace(']','').replace('[','').replace(' ','')
            num_list = text.split(':')
            return abs(int(num_list[0]) - int(num_list[1])) + 1
        if '[' in line:
            pattern = r'output\s+([^\s]+)\s+([^\s]+)'
            match = re.search(pattern, line)
            width_text = match.group(1)
            output_name = match.group(2)
            output_width = get_width(width_text)
        else:
            pattern = r'output\s+([^\s]+)'
            match = re.search(pattern, line)
            output_name = match.group(1)
            output_width = 1
        return output_name, output_width
        
    def get_input_width_dict(self, design_num):
        design_content = self.design1 if design_num == 1 else self.design2
        design_list = design_content.split(';')
        design_list = [x.strip() for x in design_list]
        input_list = [x for x in design_list if x.startswith('input')]
        input_width_dict = {}    
        for input in input_list:
            input_name, input_width = self.parse_input(input)
            input_width_dict[input_name] = input_width
        return input_width_dict
            
    def get_output_width_dict(self, design_num):
        design_content = self.design1 if design_num == 1 else self.design2
        design_list = design_content.split(';')
        design_list = [x.strip() for x in design_list]
        output_list = [x for x in design_list if x.startswith('output')]
        output_width_dict = {}    
        for output in output_list:
            output_name, output_width = self.parse_output(output)
            output_width_dict[output_name] = output_width
        return output_width_dict
            
    def gen_miter_header(self):
        return 'module miter();\n'
    
    def gen_define_list(self):
        content = ''
        for output in self.output_width_dict:
            var_type = 'wire'
            bits = f'[{self.output_width_dict[output]-1}:0]' if self.output_width_dict[output] > 1 else ''
            name = f'{output}1, {output}2'
            content += f'{var_type} {bits} {name};\n'
        return content
    
    def gen_inst(self):
        content = ''
        for num in [1, 2]:
            module_type = f'design{num}'
            inst_name = f'inst{num}'
            var_state = ','.join([f'.{x}({x})' if x not in self.output_width_dict else f'.{x}({x}{num})' for x in self.header_var_list1])
            inst = f'{module_type} {inst_name} ({var_state});\n'
            content += inst
        return content
    
    def gen_compare(self):
        content = 'always @(posedge clk) begin\n'
        
        asserts = ''
        for output in self.output_width_dict:
            asserts += f'assert({output}1 == {output}2);\n'
        content += asserts
        
        content += 'end\n'
        
        return content
    
    def gen_endmodule(self):
        return 'endmodule'
            
            
    def generate_miter_module(self):
        content = ''
        content += self.gen_miter_header()
        content += self.gen_define_list()
        content += self.gen_inst()
        content += self.gen_compare()
        content += self.gen_endmodule()
        return content
        
    def rename_module(self, new_name):
        if new_name == 'design1':
            pattern = rf'{self.design1_module_name}'
            new_text = re.sub(pattern, new_name, self.design1, count=1)
            return new_text
        else:
            pattern = rf'{self.design2_module_name}'
            new_text = re.sub(pattern, new_name, self.design2, count=1)
            return new_text
                

    def set_reg_init(self, text):
        def get_width(line):
            if not '[' in line:
                return 1
            width_state = line.split('[')[1].split(']')[0]
            param1 = int(width_state.split(':')[0].strip())
            param2 = int(width_state.split(':')[1].strip())
            return abs(param1 - param2) + 1
        
        text_lines = text.split('\n')
        for i, line in enumerate(text_lines):
            if line.strip().startswith('reg'):
                if not '=' in line:
                    text_lines[i] = line.replace(';', f" = {get_width(line)}\'d0;")
                elif '=' in line:
                    text1 = line.split('=')
                    text_lines[i] = text1[0] + f" = {get_width(text1[1])}'d0;"
        text = '\n'.join(text_lines)
        return text
                
    def generate_miter_file_content(self):
        content = ''
        content += self.set_reg_init(self.rename_module('design1')) + '\n\n'
        content += self.set_reg_init(self.rename_module('design2')) + '\n\n'
        content += self.generate_miter_module()
        
        # ############### tackle plus1 bug
        # content = content.replace(':plus', '+:1')
        # ###############
        
        return content
    
    def write_miter(self, output_path):
        with open(output_path, 'w') as f:
            f.write(self.generate_miter_file_content())
    
        
if __name__ == "__main__":
    
    design1 = 'dataset_9_14/1/module_vga_text.v'
    design2 = 'dataset_9_14/1/module_module_1_genFromDot.v'
    miter_generator = Miter_generator(design1, design2)
    # print(miter_generator.design1_module_header)
    # print(miter_generator.design1_module_name)
    print(miter_generator.generate_miter_file_content())
    miter_generator.write_miter(os.path.join(os.path.dirname(design1), 'miter.sv'))
    
    yosys_commend = 'yosys -p "read_verilog -sv miter.sv; hierarchy -top miter; proc; flatten; sat -tempinduct -prove-asserts -verify;"'
        
    