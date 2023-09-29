import os


def list_files(dir_path, depth=0, output_file=None):
    if depth > 3:
        return

    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_path):
            if output_file:
                output_file.write(file_path + "\n")
            else:
                print(file_path)
        elif os.path.isdir(file_path):
            if output_file:
                output_file.write(file_path + "/\n")
            else:
                print(file_path + "/")
            list_files(file_path, depth + 1, output_file)

output_file_path = "/cache/output/res.txt"
with open(output_file_path, "w") as output_file:
    list_files(".", depth=0, output_file=output_file)