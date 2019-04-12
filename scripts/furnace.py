# furnace object class:

import pygame
from main import init_graphics


# init of object with sprite - pygames requirement
class Furnace(pygame.sprite.Sprite):

    # procedure of printing object properties when called by matrix
    def __repr__(self):
        return "Furnace"

    # init of object with coordinates in simulation
    def __init__(self, x, y):

        # call init of parent class
        pygame.sprite.Sprite.__init__(self)

        # init graphics with object's sprite - do not touch!
        init_graphics(self, x, y, "furnace_active")

        # real coordinates of object
        self.x = x
        self.y = y

        # lists with data
        self.orderedDishes = {}
        self.readyDishes = {}

        # how long does this furnace cook?
        # for ai learning purpose - waiter has to minimize time in restaurant
        self.time = 5

    def next_round(self):
        # change the environment:
        pass

    def activated(self):
        # serve object:
        # init graphics with object's sprite - do not touch!
        init_graphics(self, self.x, self.y, "furnace")
