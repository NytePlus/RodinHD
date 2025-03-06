import json
import os
from tqdm import tqdm

def read_json(file_path):
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
            return data
        except:
            fails.append(file_path)
            return None;

def modify_transformation(data):
    try:
        for camera in data.get('cameras', []):
            if camera.get('transformation', []) and len(camera.shape) == 3 and len(camera['transformation'][0]) == 4:
                del camera['transformation'][0][-1]
                camera['transformation'] = camera['transformation'][0]
        return data
    except:
        return None;

def modify_focal(data):
    try:
        for camera in data.get('cameras', []):
            if camera.get('focal_length', []):
                camera['focal_length'] = 40
        return data
    except:
        return None;

def write_json(data, file_path):
    if data is not None:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

if __name__ == '__main__':
    directory_path = '../data/metahuman_data'
    fails = []
    for root, dirs, files in tqdm(os.walk(directory_path)):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                data = read_json(file_path)
                modified_data = modify_focal(data)
                write_json(modified_data, file_path)
    print(f'unable to load {fails}')