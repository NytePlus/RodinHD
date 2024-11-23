list_dir = '../data/portrait3d_data/fitting_obj_list.txt'
with open(list_dir, 'r') as file:
    lines = file.readlines()

def extract_number(line):
    return int(line.strip().split('_')[-1])

sorted_lines = sorted(lines, key=extract_number)

with open(list_dir, 'w') as file:
    for line in sorted_lines:
        file.write(line.replace('raw_data', 'portrait3d_data'))
