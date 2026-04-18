
def set_mode():

    while(1):
        print("\n0: Exit")
        print("1: Train a new model")
        print("2: Fine-tuning a train model")
        print("3: Test a model")
        print("4: Examine a model")
        print("5: Delete a model")
        
        str_answer = input("What do you want to do ?\n").strip()
                
        if (str_answer == "0"):
            print("Exit")
            exit(0)

        if (str_answer == "1"):
            print("You chose the mode Traning")
            return(1)
        
        elif (str_answer == "2"):
            print("You chose the mode fine-tunning")
            return(2)
        
        elif (str_answer == "3"):
            print("You chose the mode test")
            return(3)
        
        elif (str_answer == "4"):
            print("You chose the mode test")
            return(4)
        
        elif (str_answer == "5"):
            print("You chose the mode delete")
            return(5)
        
        else:
            print("Please enter only 1, 2, 3, 4 or 5")