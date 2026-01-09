import json, os, re, time

def vlg_clean(file, file_clean):
    # file_tmp = file
    # os.system("cp {0} ./{1}".format(file, file_tmp))
    # os.remove(file)
    with open(file, "r") as f:
        lines = f.readlines()
        with open(file_clean, "w+") as f_tmp:
            for line in lines:
                line = re.sub(r'\(\*(.*)\*\)', '', line)
                if line.strip():
                    f_tmp.writelines(line)
    # os.remove(file_tmp)

    
if __name__ == '__main__':
    design_name = "test"
    cmd = 'ast'
    file_dir = f"./{design_name}_{cmd}.v"
    file_clean_dir = f"./{design_name}_{cmd}_clean.v"
    vlg_clean(file_dir, file_clean_dir)
    


