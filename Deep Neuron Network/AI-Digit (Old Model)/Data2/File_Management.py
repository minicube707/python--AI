import os
import re

def file_management(dimensions, test_accu):
    str_dim = str(tuple(dimensions)).replace(" ", "")
    str_accu = f"{test_accu:.5f}".replace(".", ",")
    name_model = f"DM{str_dim}({str_accu}).pickle"

    return name_model

def select_model():

    module_dir = os.path.dirname(__file__)
    dir_list = os.listdir(module_dir + "/Model")
    res_file = [x for x in dir_list if any(y in x for y in "DM")]

    print("\nModel present in the file:")
    for i, file in enumerate(res_file):
        print(f"{i+1}: {file}")

    index = 0
    while(index < 1 or index > len(res_file)):
        index = int(input("\nWhich would you select ?\n"))

        if index < 1 or index > len(res_file):
            print(f"Please enter a number between 1 and {len(res_file)}")

    print(f"\nModel select is ", res_file[index-1])
    return res_file[index-1]

def get_hidden_layers(name):

    match = re.search(r"\(([^)]*)\)", name)
    numbers = re.findall(r"\d+", match.group(1))  # Extraire uniquement les nombres
    result = tuple(map(int, numbers))  # Convertir en tuple d'entiers

    print("\nHidden layers:\n",result)  
    return result 