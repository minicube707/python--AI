import pygame
import numpy as np
import time
import math
from Settings import settings, contact_coin, make_list_coin

pygame.init()

# Crée une fenêtre de 800x600 pixels
WIDTH, HEIGHT = 1900, 900
WIN = pygame.display.set_mode((WIDTH, HEIGHT))

# Définit la couleur blanche
WHITE = [255, 255, 255]
DARK_GREY = [100, 100, 100]
BLACK = [0, 0, 0]
RED = [255, 0, 0]
GREEN = [0, 255, 0]
YELLOW = [255, 255, 0]

SCALE = 1/2000


# Remplit l'écran avec la couleur blanche
WIN.fill(BLACK)

# Définit le titre de la fenêtre
pygame.display.set_caption("Basic pygame")

def move_drone (center_mass, velocity, theta, thrust, max_thrust, max_velocity, delta_rot, delta_thrust, dt):

    #Aceleration
    delta_x = 0
    delta_y = 0

    #Velocity
    velocity_x = velocity[0]
    velocity_y = velocity[1]

    keys = pygame.key.get_pressed()

    #Left
    if keys[pygame.K_a] :
        theta += -delta_rot

    #Right
    if keys[pygame.K_e]:
        theta += delta_rot
                
    #Down
    if keys[pygame.K_s] and thrust - 1 >= 0:
        thrust += -delta_thrust

    #Up 
    if keys[pygame.K_z] and thrust + 1 <= max_thrust:
        thrust += delta_thrust
    
    delta_x = math.cos(theta + 3*math.pi/2)
    delta_y = math.sin(theta + 3*math.pi/2)

   
    delta_x = delta_x * thrust * dt
    delta_y = delta_y * thrust * dt

    velocity_x += delta_x
    velocity_y += delta_y

    if velocity_x > max_velocity:
        velocity_x = max_velocity
    elif velocity_x < -max_velocity:
        velocity_x = -max_velocity
    
    if velocity_y > max_velocity:
        velocity_y = max_velocity
    elif velocity_y < -max_velocity:
        velocity_y = -max_velocity

    new_velocity = (velocity_x,  velocity_y)

    new_center_mass= (center_mass[0] + velocity_x*SCALE, center_mass[1] + velocity_y*SCALE)

    return new_center_mass, new_velocity, theta,  thrust

def draw_coin(list_coin):
    coin = list_coin[0]
    x_coin = coin[0]
    y_coin = coin[1]

    pygame.draw.circle(WIN, YELLOW, (x_coin, y_coin), 10)


def draw_drone(center_mass, shape_drone, shape_arm):
    
    width_drone = shape_drone[0]
    height_drone = shape_drone[1]

    width_arm = shape_arm[0]
    height_arm = shape_arm[1]

    #Body
    pygame.draw.rect(WIN, DARK_GREY, (center_mass[0] - width_drone//2 , center_mass[1] - height_drone//2 , width_drone, height_drone))
    
    #Left arm
    pygame.draw.rect(WIN, DARK_GREY, (center_mass[0] - width_drone//2 - width_arm, center_mass[1] + height_arm//2, width_arm, height_arm))
    pygame.draw.line(WIN, DARK_GREY, (center_mass[0] - width_drone//2 - width_arm, center_mass[1] + height_arm), (center_mass[0] - width_drone//2, center_mass[1] - height_drone//2 + height_arm), 5)

    #Right arm
    pygame.draw.rect(WIN, DARK_GREY, (center_mass[0] + width_drone//2, center_mass[1] + height_arm //2, width_arm, height_arm))
    pygame.draw.line(WIN, DARK_GREY, (center_mass[0] + width_drone//2, center_mass[1] - height_drone//2  + height_arm), (center_mass[0] + width_drone//2 + width_arm, center_mass[1] + height_arm), 5)

    #Shader
    pygame.draw.rect(WIN, BLACK, (center_mass[0] - width_drone//2 , center_mass[1] - height_drone//2 , width_drone, height_drone), 1)
    pygame.draw.rect(WIN, BLACK, (center_mass[0] - width_drone//2 - width_arm, center_mass[1] + height_arm//2, width_arm, height_arm), 1)
    pygame.draw.rect(WIN, BLACK, (center_mass[0] + width_drone//2, center_mass[1] + height_arm //2, width_arm, height_arm), 1)


def draw_thruster(center_mass, theta, thrust, shape_drone, shape_arm, shape_thruster):

    width_thruster = shape_thruster[0]
    height_thruster = shape_thruster[1]

    width_arm = shape_arm[0]
    height_arm = shape_arm[1]

    width_drone = shape_drone[0]

    new_left_thruster = np.array([])
    new_right_thruster = np.array([])
    thruster = np.array([[-width_thruster//2, -height_thruster//2, 1],
                        [0, -height_thruster//2 - height_thruster//5, 1],
                        [width_thruster//2, -height_thruster//2, 1],
                        [width_thruster//2, height_thruster//2, 1],
                        [-width_thruster//2, height_thruster//2, 1]])
    
    new_left_fire = np.array([])
    new_right_fire = np.array([])
    fire= np.array([[width_thruster//2, height_thruster//2, 1],
                    [-width_thruster//2, height_thruster//2, 1],
                    [0, thrust/20 + height_thruster//2, 1]])
    
    #Matrice rotation
    left_matrice_rotation = np.array(   [[np.cos(theta), -np.sin(theta), center_mass[0] - width_drone//2 - width_arm], 
                                        [np.sin(theta), np.cos(theta), center_mass[1] + height_arm]])

    right_matrice_rotation = np.array(  [[np.cos(theta), -np.sin(theta), center_mass[0] + width_drone//2 + width_arm], 
                                        [np.sin(theta), np.cos(theta), center_mass[1] + height_arm]])
    
    #Thrust
    for point in thruster:
        new_left_thruster = np.append(new_left_thruster, left_matrice_rotation.dot(point.T))       
    new_left_thruster = new_left_thruster.reshape((-1, 2))

    for point in thruster:
        new_right_thruster = np.append(new_right_thruster, right_matrice_rotation.dot(point.T))       
    new_right_thruster = new_right_thruster.reshape((-1, 2))


    #Fire
    for point in fire:
        new_left_fire = np.append(new_left_fire, left_matrice_rotation.dot(point.T))
    new_left_fire = new_left_fire.reshape((-1, 2))

    for point in fire:
        new_right_fire = np.append(new_right_fire, right_matrice_rotation.dot(point.T))
    new_right_fire = new_right_fire.reshape((-1, 2))


    pygame.draw.polygon(WIN, RED, new_left_fire)
    pygame.draw.polygon(WIN, RED, new_right_fire)
    pygame.draw.polygon(WIN, DARK_GREY, new_left_thruster)
    pygame.draw.polygon(WIN, DARK_GREY, new_right_thruster)


def draw(center_mass, theta, thrust, max_thrust, shape_drone, shape_arm, shape_thruster, list_coin):

    WIN.fill(BLACK)

    #Coin
    if len(list_coin) > 0:
        draw_coin(list_coin)

    #Drone
    draw_drone(center_mass, shape_drone, shape_arm)
    draw_thruster(center_mass, theta, thrust, shape_drone, shape_arm, shape_thruster)
    
    #Thrust Level
    pygame.draw.rect(WIN, GREEN, (WIDTH//2 -20, HEIGHT - 80, 40, 80))
    pygame.draw.rect(WIN, WHITE, (WIDTH//2 -20, HEIGHT-80, 40, 80*(max_thrust-thrust)/max_thrust))
    
    pygame.display.flip()

def main():
    setting = settings(WIDTH, HEIGHT)

    width_drone  = setting["width_drone"]
    height_drone = setting["height_drone"]
    shape_drone = (width_drone, height_drone)

    width_arm = setting["width_arm"]
    height_arm = setting["height_arm"]
    shape_arm = (width_arm, height_arm)

    width_thruster = setting["width_thruster"]
    height_thruster = setting["height_thruster"]
    shape_thruster = (width_thruster, height_thruster)

    theta =  setting["theta"]
    thrust = setting["thrust"]
    max_thrust = setting["max_thrust"]
    velocity = setting["velocity"]
    max_velocity = setting["max_velocity"]
    center_mass = setting["center_mass"]
    delta_rot = setting["delta_rot"]
    delta_thrust = setting["delta_thrust"]

    list_coin = make_list_coin(WIDTH, HEIGHT)
    end = False
    finish = False

    # Boucle principale
    run = True
    last_time = time.time()
    while run:

        dt = time.time() - last_time
        last_time = time.time()

        #Pygame event
        for event in pygame.event.get():

            #Quit pygame
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    run = False

        if not end and not  finish:

            list_coin, finish, _ = contact_coin(center_mass, list_coin, shape_drone)
            center_mass, velocity, theta, thrust = move_drone (center_mass, velocity, theta, thrust, max_thrust, max_velocity, delta_rot, delta_thrust, dt)
            draw(center_mass, theta, thrust, max_thrust, shape_drone, shape_arm, shape_thruster, list_coin)

            if (center_mass[0] + shape_drone[0]//2 < 0) or (center_mass[0] - shape_drone[0]//2 > WIDTH) or (center_mass[1] + shape_drone[1]//2 < 0) or (center_mass[1] - shape_drone[1]//2 > HEIGHT):
                end = True

        else:
            WIN.fill(BLACK)
            FONT = pygame.font.SysFont("trebuchetms", 20)
            FINISH_TEXT = pygame.font.SysFont("trebuchetms", 50)
            if not finish:
                finish_text = FINISH_TEXT.render("Game over", 1, WHITE)
            else:
                finish_text = FINISH_TEXT.render("Congratulation", 1, WHITE)

            text_score = FONT.render(f"Score: {11 - len(list_coin):,}/11", 1, WHITE)

            WIN.blit(finish_text, (WIDTH//2 - finish_text.get_width() // 2, HEIGHT//2))
            WIN.blit(text_score, (WIDTH//2 - text_score.get_width()// 2, HEIGHT//2 + finish_text.get_height()))

            # Rafraîchit l'affichage
            pygame.display.flip()


    # Ferme Pygame
    pygame.quit()

main()
