import os
import pickle

def load_model(model_name):
    with open("Model/" + str(model_name), 'rb') as file:
        return pickle.load(file)

def save_model(filename, data):
    with open("Model/" + filename, 'wb') as file:
        pickle.dump(data, file)

def file_management(test_accu, test_conf, dimensions_CNN):
    str_nb_kernel = ','.join(str(v[3]) for v in dimensions_CNN.values() if v[4] == 'kernel')
    str_size = ','.join(str(v[0]) for v in dimensions_CNN.values() if v[4] == 'kernel')
    str_accu = f"{test_accu:.5f}".replace(".", ",")
    str_conf = f"{test_conf:.5f}".replace(".", ",")

    name_model = f"DM({str_accu})({str_conf})({str_size})({str_nb_kernel}).pickle"

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
