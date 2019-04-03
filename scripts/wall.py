# furnace object class:

from scripts.__init__ import *


# init of object with sprite - pygames requirement
class Wall(pygame.sprite.Sprite):

    # procedure of printing object properties when called by matrix
    def __repr__(self):
        return "Wall"

    # init of object with coordinates in simulation
    def __init__(self, x, y):

        # call init of parent class
        pygame.sprite.Sprite.__init__(self)

        # init graphics with object's sprite - do not touch!
        init_graphics(self, x, y, "wall")

        # real coordinates of object
        self.x = x
        self.y = y

    def next_round(self):
        pass