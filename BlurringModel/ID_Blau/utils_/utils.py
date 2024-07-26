import os

def list_directories(path):
    # output all of dir from input path
    output_dirs = []
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path):
            output_dirs.append(full_path)
    return output_dirs

def list_files_sorted(path):
    # output all of file from input path
    output_dirs = []
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        if not os.path.isdir(full_path):
            output_dirs.append(full_path)
    sorted_paths = sorted(output_dirs, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    return sorted_paths

def list_files_nosorted(path):
    # output all of file from input path
    output_dirs = []
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        if not os.path.isdir(full_path):
            output_dirs.append(full_path)
    return output_dirs