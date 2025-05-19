import os
os.environ['SDL_AUDIODRIVER'] = 'directx'

import pygame

pygame.init()

#MAKE A SCREEN FOR THE PLAYER TO SEE
screen = pygame.display.set_mode((800, 400))

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    pygame.display.update()

pygame.quit()
quit()