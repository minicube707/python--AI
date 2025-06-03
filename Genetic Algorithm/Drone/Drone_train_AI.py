
import numpy as np
import random
import time
import multiprocessing as mp
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import keyboard
import os

from Settings import settings, update_X, initialisation, contact_coin, make_list_coin, move_drone, out_of_picture
from Drone_AI import  WIDTH, HEIGHT

# Function that returns the best model of the session
def fitness_function(all_list_nb_coin, all_list_time, list_AI, nb_place):
    new_list_AI = []

    # Sort indices of all_list_nb_coin in descending order
    sort_list_nb_coin = np.lexsort((all_list_time, all_list_nb_coin))[::-1]

    # Index for coins sorted list
    index_coin = 0  

    #While there is still the place and the AI has collected more than  one peace
    while (0 < nb_place) and (all_list_nb_coin[sort_list_nb_coin[index_coin]] > 0):

        #Add the AI for the next generation
        new_list_AI.append(list_AI[sort_list_nb_coin[index_coin]])

        #Delete the AI for the seconde part, within reset their time
        all_list_time[sort_list_nb_coin[index_coin]] = -1

        #Pass to the next AI and reduce the place of one
        index_coin += 1
        nb_place -= 1


    # Sort remaining models by time in descending order
    sort_indices_time = np.argsort(all_list_time)[::-1]
    for index_time in sort_indices_time:
        if nb_place <= 0:
            break
        
        new_list_AI.append(list_AI[index_time])
        nb_place -= 1

    
    return new_list_AI

#Function that train the models, and return the best of them
def train_models(list_AI, shape_drone, max_thrust, max_velocity, delta_rot, delta_thrust, list_coin, nb_place,  c_pressed):

    all_list_nb_coin = np.array([], np.int8)
    all_list_time = np.array([])
    list_c_pressed = np.array([], np.bool_)

    # Determine the number of available CPU cores, let to core free to not overload the CPU
    num_cores = mp.cpu_count() - 1
    
    # Activation of the multiprocessing
    with mp.Pool(processes=num_cores) as pool:
        results = []

        # For all the models
        for i in range(0, len(list_AI), num_cores):
            batch = list_AI[i:i + num_cores]  # Create a batch for processing

            # Asynchronously train the models in the batch
            # Use a list comprehension to prepare the arguments for train_model
            results = pool.starmap(train_neuron_network, [(parametres, shape_drone, list_coin, max_thrust, max_velocity, delta_rot, delta_thrust) for parametres in batch])

            for result in results:
                all_list_nb_coin = np.append(all_list_nb_coin, result[0])
                all_list_time = np.append(all_list_time, result[1])
                list_c_pressed = np.append(list_c_pressed, result[2])

            #If the trainnig is over
            if keyboard.is_pressed("c") or any(list_c_pressed):
                print("\nLa session a été arrété")
                c_pressed = True
                break           
  
    #If the trainnig isn't over return the best models, and the maximun time in game
    if not c_pressed:
        new_list_AI = fitness_function(all_list_nb_coin, all_list_time, list_AI, nb_place)
        return new_list_AI, np.sort(all_list_time), np.sort(all_list_nb_coin), False

    else:
        return list_AI, np.sort(all_list_time), np.sort(all_list_nb_coin), True

#Function that train the AI 
def train_neuron_network(parametres, shape_drone, list_coin_original, max_thrust, max_velocity, delta_rot, delta_thrust):

    #Bool 
    c_pressed = False
    finish = False
    bad_AI = False

    #Initilisation of the varible for the drone
    theta =  0
    thrust = 0
    velocity = (0, 0)
    center_mass = (WIDTH//2, HEIGHT//2)
    list_coin = list_coin_original

    nb_coin = 0
    start = time.time()
    delay = 5

    last_time = time.time()
    while not finish and not c_pressed:

        dt = time.time() - last_time
        last_time = time.time()
    
        #The simulation   
        list_coin, finish, more_delay = contact_coin(center_mass, list_coin, shape_drone)
        X = update_X(WIDTH, HEIGHT, list_coin, center_mass, velocity, thrust, max_thrust, max_velocity, theta)
        center_mass, velocity, theta, thrust = move_drone (center_mass, velocity, theta, thrust, X, parametres, max_thrust, max_velocity, delta_rot, delta_thrust, dt)

        if not finish:
            finish = out_of_picture(WIDTH, HEIGHT, center_mass, shape_drone)
            nb_coin = 11 - len(list_coin)

        delay += more_delay
        if(time.time() > start + delay):
           finish = True 

        #If the drone doesn't move, it's finish
        elif (center_mass == (WIDTH//2, HEIGHT//2)  and start + 1 > time.time() > start + 0.5):
            finish = True
            bad_AI = True
        
        #If the trainnig is over
        if keyboard.is_pressed("c"):
                c_pressed = True
    
    if bad_AI:
        return 0, 0, c_pressed
    
    else:
        return nb_coin, time.time() - start, c_pressed


#Function that allowed of the models to evolve
def add_mutation(chances_of_mutations, degree_of_mutation, matrice):

    #Creat of arrays of the size of the matrice
    random_matrice_probs = np.random.rand(matrice.shape[0], matrice.shape[1])
    #Where there is a number below the chances of mutation the variable is change, to simule a mutation
    #The variation of the mutation go to -0.05, 0.05 
    random_matrice_probs = np.where(random_matrice_probs < chances_of_mutations, (np.float16(degree_of_mutation*(2*np.random.rand()-1))), 0)
    
    return matrice + random_matrice_probs


def mutation(nb_mutation, select_AI, chances_of_mutation, degree_of_mutation):

    new_list_AI = []
    c_pressed = False
    for _ in range(nb_mutation):
        
        #If the simulatoin is over
        if keyboard.is_pressed("c"):
            print("\nLa session a été arrété")
            c_pressed = True
            break

        #For all Ai selectioned
        for AI in select_AI:

            #Extract their keys and value
            new_AI = {}
            for key, val in AI.items():
            
                #Create the mutation
                new_AI.update({key: add_mutation(chances_of_mutation, degree_of_mutation, val)})

            new_list_AI.append(new_AI)

    return new_list_AI, c_pressed
    

def crossover(select_AI, dimensions):
    a = 0
    b = 0
    list_crossover = []
    lenght = len(select_AI)

    for _ in range(len(select_AI)//2):

        #Selection the parents, different AI
        a = random.randint(0, len(select_AI)-1)
        b = random.randint(0, len(select_AI)-1)

        while a == b:
            a = random.randint(0, len(select_AI)-1)
            b = random.randint(0, len(select_AI)-1)
        

        AI1 = select_AI.pop(a)
        AI2 = select_AI.pop(b-1)    #The lenght is reduce of one because of pop()
        
        #Selection the gene
        c = random.randint(0, len(AI1)-1)

        key =   list(AI1.keys())[c]
        gene1 = list(AI1.values())[c]
        gene2 = list(AI2.values())[c]

        #Transfer the gene
        AI1[key] = gene2
        AI2[key] = gene1
        

        #Add to the new list
        list_crossover.append(AI1)
        list_crossover.append(AI2)


    #To keep the same number of AI
    if lenght != len(list_crossover):
        parametres = initialisation(dimensions)
        list_crossover.append(parametres)

    return list_crossover


#Main function
def main():

    #Stop System
    c_pressed = False
    setting = settings(WIDTH, HEIGHT)
    #Drone
    #Shape
    width_drone  = setting["width_drone"]
    height_drone = setting["height_drone"]
    shape_drone = (width_drone, height_drone)

    #Data
    theta =  setting["theta"]
    thrust = setting["thrust"]
    max_thrust = setting["max_thrust"]
    velocity = setting["velocity"]
    max_velocity = setting["max_velocity"]
    center_mass = setting["center_mass"]
    delta_rot = setting["delta_rot"]
    delta_thrust = setting["delta_thrust"]

    #Objectif
    list_coin = make_list_coin(WIDTH, HEIGHT)

    #Parametre of train
    pourcent_ratio = 0.02
    ratio = int(1/pourcent_ratio)
    nb_iter = 1_000
    nb_AI = 1_000
    nb_select = nb_AI//ratio
    nb_duplicated = 2
    nb_mutation =  ratio - nb_duplicated
    chances_of_mutation = 0.2
    degree_of_mutation = 0.1

    #Neuron Network
    X = update_X(WIDTH, HEIGHT, list_coin, center_mass, velocity, thrust, max_thrust, max_velocity, theta)
    dimensions = list((10, 10))
    dimensions.insert(0, X.shape[0])
    dimensions.append(4)

    #Data
    list_data_time = np.array([])
    list_data_coin = np.array([])
    list_mean = np.array([])

    #Générate AI
    list_AI = []
    for _ in range(nb_AI):
        parametres = initialisation(dimensions)
        list_AI.append(parametres)
    
    #Selet and mute AI
    for i in tqdm(range(nb_iter)):
            
        #Train the models
        select_AI, data_time, data_coin, c_pressed = train_models(list_AI, shape_drone, max_thrust, max_velocity, delta_rot, delta_thrust, list_coin, nb_select, c_pressed)
        
        print(" ")
        print("MEAN: ", np.mean(data_time))
        print(data_time[-50:-1])
        print(data_coin[-50:-1])

        new_list_AI = []

        for _ in range(nb_duplicated):
            new_list_AI.extend(select_AI)

        #Data
        if i%10 == 0:
            list_data_time  = np.append(list_data_time, data_time[-1])
            list_data_coin  = np.append(list_data_coin, data_coin[-1])
            list_mean  = np.append(list_mean, np.mean(data_time))
            

        #Crossover
        #list_crossover = crossover(select_AI, dimensions)

        #Mututation of AI
        #If the session is over
        if c_pressed:
            break

        else:
            list_AI_mutated, c_pressed = mutation(nb_mutation, select_AI, chances_of_mutation, degree_of_mutation)
            new_list_AI.extend(list_AI_mutated)
            list_AI = new_list_AI.copy()

            #Clean the memory
            del new_list_AI


    #Save the best model    
    with open('model.pickle', 'wb') as file:
        pickle.dump(select_AI[0], file)

    #Test the best model
    res = train_neuron_network(select_AI[0], shape_drone, list_coin, max_thrust, max_velocity, delta_rot, delta_thrust)
    print("")
    print(res)

    for key, val in select_AI[0].items():
        print("")
        print(key)
        print(val)

    x = np.arange(0, list_data_time.shape[0]) * 10
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot for the best time of each generation
    axs[0].plot(x, list_data_time)
    axs[0].set_title("The best time of each generation")
    axs[0].set_xlabel("Generations")
    axs[0].set_ylabel("Best Time")
    
    # Plot for the best coin of each generation
    axs[1].plot(x, list_data_coin)
    axs[1].set_title("The best coin of each generation")
    axs[1].set_xlabel("Generations")
    axs[1].set_ylabel("Best Coin")
    
    # Plot for the mean of the AI of each generation
    axs[2].plot(x, list_mean)
    axs[2].set_title("The mean of the AI of each generation")
    axs[2].set_xlabel("Generations")
    axs[2].set_ylabel("Mean AI")
    
    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':  
    # Required for frozen executables on Windows
    mp.freeze_support()  
    os.chdir("Desktop\\Document\\Programmation\\Python\\AI\\Genetic Algorithm\\Drone")
    main()

