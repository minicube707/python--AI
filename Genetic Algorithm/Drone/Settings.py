import numpy as np
import random
import math

SCALE = 1/2000

def settings(WIDTH, HEIGHT):
    width_drone  = 55
    height_drone = 30

    width_arm = 35
    height_arm = 10

    width_thruster = 5
    height_thruster = 15

    theta =  0
    thrust = 0
    max_thrust = 700
    velocity = (0, 0)
    max_velocity = 15_000
    center_mass = (WIDTH//2, HEIGHT//2)
    delta_rot = 0.003
    delta_thrust = 2

    # Retourne les paramètres sous forme de dictionnaire
    return {
        "width_drone": width_drone,
        "height_drone": height_drone,
        "width_arm": width_arm,
        "height_arm": height_arm,
        "width_thruster": width_thruster,
        "height_thruster": height_thruster,
        "theta": theta,
        "thrust": thrust,
        "max_thrust": max_thrust,
        "velocity": velocity,
        "max_velocity": max_velocity,
        "center_mass": center_mass,
        "delta_rot" : delta_rot,
        "delta_thrust": delta_thrust
    }



#Function that verify if the drone touch the an objectif
def contact_coin(center_mass, list_coin, shape_drone):
    
    delay = 0

    if len(list_coin) == 0:
        return [], True, 0
        
    x_coin = list_coin[0, 0]
    y_coin = list_coin[0, 1]
    
    #If the drone touch the objectif
    if (center_mass[0] - shape_drone[0]//2 < x_coin < center_mass[0] + shape_drone[0]//2) and (center_mass[1] - shape_drone[1]//2 < y_coin < center_mass[1] + shape_drone[1]//2):
        #Delete the coordonner, and reshape the list
        list_coin = np.delete(list_coin, 0)
        list_coin = np.delete(list_coin, 0)
        list_coin = list_coin.reshape((-1, 2))
        delay = 5

    return list_coin, False, delay


def make_list_coin(WIDTH, HEIGHT):
    list_coin = np.array([], np.int8)
    list_num = []

    a = random.randint(0, 1)
    b = random.randint(0, 3)

    if a == 0:
        list_num.extend([1, 2, 3, 4, 0])
    else:
        list_num.extend([3, 4, 1, 2, 0])

    if b == 0:
        list_num.extend([1, 3, 2, 4, 1, 0])
    elif b == 1:
        list_num.extend([2, 4, 3, 1, 4, 0])
    elif  b ==2:
        list_num.extend([3, 1, 4, 2, 3, 0])
    else:
        list_num.extend([4, 2, 1, 3, 2, 0])
    
    for num in list_num:
        
        if num == 0:
            x_coin = WIDTH//2
            y_coin = HEIGHT//2
        elif num == 1:
            x_coin = 4*WIDTH//5
            y_coin = HEIGHT//5
        elif num == 2:
            x_coin = WIDTH//5
            y_coin = HEIGHT//5
        elif num == 3:
            x_coin = WIDTH//5
            y_coin = 4*HEIGHT//5
        else:
            x_coin = 4*WIDTH//5
            y_coin = 4*HEIGHT//5

        list_coin = np.append(list_coin, x_coin)
        list_coin = np.append(list_coin, y_coin)
    
    list_coin = list_coin.reshape((-1, 2))

    return list_coin

#Function that return if the drone is out of window
def out_of_picture(WIDTH, HEIGHT, center_mass, shape_drone):
      
    #I the drone get out the picture. It's over
    if (center_mass[0] + shape_drone[0]//2 < 0) or (center_mass[0] - shape_drone[0]//2 > WIDTH) or (center_mass[1] + shape_drone[1]//2 < 0) or (center_mass[1] - shape_drone[1]//2 > HEIGHT):
        return True

    #Else continue to work
    else:
        return False


#Function that return the input of the AI
def update_X(WIDTH, HEIGHT, list_coin, center_mass, velocity, thrust, max_thrust, max_velocity, theta):

    #Information on the target
    #Coordonnate of the target
    x_coin = list_coin[0,0]    
    y_coin = list_coin[0,1]

    #Distance to the target                      
    x_distance = (x_coin - center_mass[0]) / WIDTH      #Distance to the target on the x axis normalised
    y_distance = (y_coin - center_mass[1]) / HEIGHT     #Distance to the target on the y axis normalised

    #Information of the location of the target
    x0 = math.sqrt(x_distance**2 + y_distance**2)       #Norme to the target
    x1 = math.acos(x_distance/x0)/math.pi               #The direction of the target normalised
    x2 = math.acos(y_distance/x0)/math.pi               #The direction of the target normalised

    #Information on the drone
    x3 = center_mass[0]/WIDTH                           #Position of the drone on the x axis
    x4 = center_mass[0]/HEIGHT                          #Position of the drone on the y axis

    #Information on the velocity
    velocity_x = velocity[0]/max_velocity
    velocity_y = velocity[1]/max_velocity
    x5 = math.sqrt(velocity_x**2 + velocity_y**2)                   #The norme of the velocity
    x6 = math.acos(velocity_x/(x5 + 1e-15))/math.pi                #The direction on the velocity normalised
    x7 = math.acos(velocity_y/(x5 + 1e-15))/math.pi                #The direction on the velocity normalised
    
    #Thrust
    x8 = thrust/max_thrust                              #The power available normalised
    x9 = (math.cos(theta)+1)/2                          #The angle of the thruter normalised

    X = np.array([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9])
    return X.reshape((len(X), 1))

#Function that inithialise the model
def initialisation(dimension):

    parametres ={}
    C = len(dimension)

    #The weight and bias are initialise to bettween -1 and 1
    for c in range(1, C):
        parametres["W" + str(c)] = (np.float16(np.random.rand(dimension[c], dimension[c-1]))*2 -1)
        parametres["b" + str(c)] = (np.float16(np.random.rand(dimension[c], 1))*2 -1)

    return parametres


#Activation fonction
def softmax(numerateur, denominateur):
    return np.exp(numerateur)/ np.exp(denominateur).sum()

def tanh(x):
    return np.tanh(x)

def sigmoïde(x):
    return 1/(1 + np.exp(-x))


#Neuron network
def neron_network(X, parametres):

    activation = {"A0" : X}
    C = len(parametres) // 2

    for c in range(1, C+1):

        Z = parametres["W" + str(c)].dot(activation["A" + str(c-1)]) + parametres["b" + str(c)]
        activation["A" + str(c)] = np.float16(tanh(Z)) 
        
    return activation


#Function that move the robot with the neuron network
def move_drone (center_mass, velocity, theta, thrust, X, parametres, max_thrust, max_velocity, delta_rot, delta_thrust, dt):

    #Aceleration
    delta_x = 0
    delta_y = 0

    #Velocity
    velocity_x = velocity[0]
    velocity_y = velocity[1]

    #Neron_network
    activations = neron_network(X, parametres)
    C = len(parametres) // 2
    A = activations["A" + str(C)]

    #Use the softmax function to only one output activite
    #The angle of the thruster

    #Use the softmax function to only one output activite
    if A[0] >= 0.5:
        theta += delta_rot
    
    if A[1] >= 0.5:
        theta += -delta_rot
    
    #The power available
    if A[2] >= 0.5 and (thrust + 2 <= max_thrust):
        thrust += delta_thrust

    if A[3] >= 0.5 and (thrust - 2 >= 0):
        thrust += -delta_thrust


    delta_x = math.cos(theta + 3*math.pi/2)
    delta_y = math.sin(theta + 3*math.pi/2)

    #Scale and module the acceleration
    delta_x = delta_x * thrust * dt
    delta_y = delta_y * thrust * dt

    #Add the acceleration to the speed
    velocity_x += delta_x
    velocity_y += delta_y

    #Normalise the velocity
    if velocity_x > max_velocity:
        velocity_x = max_velocity
    elif velocity_x < -max_velocity:
        velocity_x = -max_velocity
    
    if velocity_y > max_velocity:
        velocity_y = max_velocity
    elif velocity_y < -max_velocity:
        velocity_y = -max_velocity

    #Update the velocity and the position
    new_velocity = (velocity_x,  velocity_y)
    new_center_mass= (center_mass[0] + velocity_x*SCALE, center_mass[1] + velocity_y*SCALE)

    return new_center_mass, new_velocity, theta,  thrust
