
def set_mode():

    while(1):
        print("1: Train a new model")
        print("2: Fine-tuning a train model")
        print("3: Test a model")
        str_answer = input("Qu'est ce que vous voulez faire ?\n")
        try:
            int_answer = int(str_answer)
        except:
            print("Veuilliez repondre que par 1, 2 ou 3")
            continue
        if (int_answer == 1):
            print("Vous avez choisi le mode Traning")
            return(1)
        
        elif (int_answer == 2):
            print("Vous avez choisi le mode fine-tunning")
            return(2)
        
        elif (int_answer == 3):
            print("Vous avez choisi le mode test")
            return(3)
        
        else:
            print("Veuilliez repondre que par 1, 2 ou 3")