# agent object class:

import itertools
import sys
import time
import math
import heapq

from pygame.locals import *

from UberKelner import init_graphics, blocksize
from scripts.matrix import *
from scripts.wall import *

# set recursion limit:
sys.setrecursionlimit(1500)


# init of object with sprite - pygames requirement
# noinspection PyTypeChecker
class Waiter (pygame.sprite.Sprite):

    # procedure of printing object properties when called by matrix
    def __repr__(self):
        return "Waiter"

    # initialize agent with list of coordinates for tables and furnaces and their number
    def __init__(self, n, matrix_fields, num_tables, num_furnaces, num_walls, solving_method):

        # call init of parent class
        pygame.sprite.Sprite.__init__(self)

        # check if there is enough space for everyting in simulation
        if num_tables + num_furnaces + num_walls + 1 > n*n:
            print("Agent: Not enough space in restaurant for objects!")
            sys.exit("N-space overflow")

        self.n = n

        # init restaurant - matrix of objects
        self.restaurant = Matrix(n, n)

        # set random coordinates of agent
        self.x, self.y = matrix_fields[0][0], matrix_fields[0][1]

        # init graphics with object's sprite - do not touch!
        init_graphics(self, self.x, self.y, "waiter")

        # add objects to restaurant - creates tables and furnaces basing on random positions in the matrix
        # objects have coordinates like in matrix (0..n, 0..n):

        # add ghostwaiter to restaurant to mark waiters position
        self.restaurant.insert('Waiter', self.x, self.y)

        # counter counts number of used coordinates, so no object will occupy the same space in simulation
        counter = 1

        # add tables
        for i in range(num_tables):
            self.restaurant.simple_insert(DinningTable(matrix_fields[i + counter][0], matrix_fields[i + counter][1]))

        # increase counter with number of used coordinates
        counter += num_tables

        # add furnaces
        for i in range(num_furnaces):
            self.restaurant.simple_insert(Furnace(matrix_fields[i + counter][0], matrix_fields[i + counter][1]))

        # increase counter with number of used coordinates
        counter += num_furnaces

        # add furnaces
        for i in range(num_walls):
            self.restaurant.simple_insert(Wall(matrix_fields[i + counter][0], matrix_fields[i + counter][1]))

        # calculate graph
        self.graph = self.restaurant.to_graph()
        self.graph2 = self.restaurant.to_graph_visited_or_not()
        # set list of objects
        self.objects_coordinates = matrix_fields[1:counter]
        # set list of goals
        self.goals = self.objects_coordinates[:]

        # set permutations of goals
        self.goalsPer = list(map(list, list(itertools.permutations(self.goals[:]))))

        # set list of solutions
        self.solutions = []

        # set path
        self.path = []

        # set all available solving methods names
        self.available_methods = ['depthfs', 'breadthfs', 'bestfs']
        self.unsupervised_learning = ['rabbit', 'svm', 'dtree', 'lreg']

        # set neighbourhood
        self.neighbourhood = []

        # set solving method
        self.solving_method = solving_method

        # run solution seeking
        self.solve(self.solving_method)
        self.control = True

    # function returning list of coordinates of agent
    def get_coordinates(self):
        return [self.x, self.y]

    # movement procedure - change position of agent on defined difference of coordinates
    def move(self, delta_x, delta_y):
        # temporarily set new coordinates
        new_x = self.x + delta_x
        new_y = self.y + delta_y

        # if movement is allowed by matrix, within restaurant borders and the field is empty:
        if new_x in range(self.restaurant.size()) and new_y in range(self.restaurant.size()):
            if self.restaurant.move(self.x, self.y, new_x, new_y):

                # set new coordinates
                self.x = new_x
                self.y = new_y

                # update waiter sprite localization after changes
                self.rect.x = self.x * blocksize
                self.rect.y = self.y * blocksize

            else:
                # activate object you tried to move on
                self.restaurant.activate(new_x, new_y)
                # remove next coordinate from path (it tries to come back from object when no move was made)
                if self.path:
                    self.path.pop(0)
        else:
            print("Agent: movement outside of simulation is prohibited (%s, %s)" % (new_x, new_y))

        # remove used move
        if self.path:
            self.path.pop(0)

    # noinspection PyTypeChecker
    def next_round(self, key):

        # list of events on keys:
        if key == K_RIGHT:
            self.control = False
            self.move(1, 0)
        elif key == K_LEFT:
            self.control = False
            self.move(-1, 0)
        elif key == K_DOWN:
            self.control = False
            self.move(0, 1)
        elif key == K_UP:
            self.control = False
            self.move(0, -1)

        # activate AI agent on key SPACE:
        if key == K_SPACE:

            # check if agent left his path:
            if not self.control:

                # run solution seeking
                self.control = True
                self.solve(self.solving_method)

            # move agent on path
            if self.path:
                self.move(self.path[0][0], self.path[0][1])
            else:
                print("Agent: No moves left!")

        # DIAGRAM SEQUENCE HERE! - ADD IN NEXT VERSION!
        # if if if if

    # parser of list of lists of coordinates to list of lists of moves
    @staticmethod
    def calculate_vector_movement(list_):
        # calculate movement vectors basing on coordinates
        for i in range(len(list_) - 1):
            list_[i][0] = list_[i + 1][0] - list_[i][0]
            list_[i][1] = list_[i + 1][1] - list_[i][1]
            if list_[i] is [0, 0]:
                print("Agent: Error - zero movement detected!")
        # remove last move (it can't be executed)
        list_.pop(-1)
        return list_

    # //////////////////////////////////////////////////
    #           S O L U T I O N S
    # //////////////////////////////////////////////////

    # Serve multiple solutions choice
    def solve(self, method):
        if method in self.available_methods:
            # set solving method
            self.solving_method = method

            # reload lists
            self.goals = self.objects_coordinates[:]
            self.path = []
            self.solutions = []

            # measure time
            starttime = time.time()
            print("Agent: %s path calculation executed..." % self.solving_method)

            if self.solving_method == "depthfs":
                # get dfs path and add results to self.solutions
                self.get_dfs_path()
            elif self.solving_method == "breadthfs":
                # get bfs path and add results to self.solutions
                self.get_bfs_path()
            elif self.solving_method == "bestfs":
                # get bestfs path and add results to self.solutions
                self.get_bestfs_path()

            # print execution time
            print("Agent: %s path calculation execution complete "
                  "after {0:.2f} seconds.".format(time.time() - starttime) % self.solving_method)

            # choose the shortest solution of restaurant and parse it to movement vector
            self.path = list(min(self.solutions, key=len))
            if len(self.path) > 0:
                # parse list to get coordinates of next moves
                self.path = self.calculate_vector_movement(self.path)
            else:
                print("Agent: no %s path found!" % self.solving_method)

        elif method in self.unsupervised_learning:
            # set solving method
            self.solving_method = method
            if self.solving_method == "rabbit":
                self.get_rabbit_path(5)  # set desired neighbourhood size
            elif self.solving_method == "svm":
                self.get_svm_path()
            elif self.solving_method == "lreg":
                self.get_logistic_regression_path()
            elif self.solving_method == "dtree":
                self.get_decision_tree_path()
            # because these methods calculate only one step (not the whole path),
            # they should be called again for next move
            self.control = False

        elif method == "all":
            for method in self.available_methods:
                self.solve(method)
        else:
            print("Agent: Unknown method of solving (%s)" % method)

    # //////////////////////////////////////////////////
    #           S E A R C H E S

    # Depth-First Search
    @staticmethod
    def parse_dfs_list(list_):
        # parse list to get coordinates of next moves
        for e in list_:
            for i in range(len(e)):
                e[i] = list(map(int, e[i].split(',')))
        # make list from list of lists
        list_ = [item for sublist in list_ for item in sublist]
        return list_

    # recursive calculation of dfs path saved temporarly in self.path
    def calculate_dfs_path(self, graph, start, goal):
        stack = [(start, [start])]
        while stack:
            (vertex, path) = stack.pop()
            for next_ in graph[vertex] - set(path):
                if next_ == goal:
                    # add path
                    self.path.append(path)
                    # remove goal and calculate next path
                    temp = self.goals.pop(0)
                    if self.goals:
                        # call next goal
                        self.calculate_dfs_path(self.graph, next_, str(self.goals[0][0]) + "," + str(self.goals[0][1]))
                        # free memory
                        del temp
                    else:
                        # add last goal to path
                        self.path.append([str(temp[0]) + "," + str(temp[1])])
                else:
                    stack.append((next_, path + [next_]))

    def experimental_calculate_dfs_path(self, graph, start, goal):
        stack = [(start, [start])]
        stack_visited_or_not = [(start, [start])]
        while stack:
            (vertex, path) = stack.pop()
            for next_ in graph[vertex] - set(path):
                if next_ == goal and stack_visited_or_not is not "visited":
                    # add path
                    stack_visited_or_not = "visited"
                    self.path.append(path)
                    # remove goal and calculate next path
                    temp = self.goals.pop(0)
                    if self.goals:
                        # call next goal
                        self.calculate_dfs_path(self.graph, next_, str(self.goals[0][0]) + "," + str(self.goals[0][1]))
                        # free memory
                        del temp
                    else:
                        # add last goal to path
                        self.path.append([str(temp[0]) + "," + str(temp[1])])
                else:
                    stack.append((next_, path + [next_]))

    # procedure responsible of calculating all possible dfs paths
    def get_dfs_path(self):
        # for all permutations of goals list:
        for self.goals in copy.deepcopy(self.goalsPer):
            # clear dfs_path and run next permutation
            self.path = []
            # calculate dfs
            start = str(self.x) + "," + str(self.y)
            goal = str(self.goals[0][0]) + "," + str(self.goals[0][1])
            self.calculate_dfs_path(self.graph, start, goal)
            # add parsed dfs_path to solutions
            self.solutions.append(self.parse_dfs_list(self.path[:]))
        # now self.solutions contains all solutions of dfs
    # //////////////////////////////////////////////////

    # Breadth-First Search

    # recursive calculation of dfs path saved temporarly in self.path
    def calculate_bfs_path(self, graph, start, goal):
        queue = [(start, [start])]
        while queue:
            (vertex, path) = queue.pop()
            for next_ in graph[vertex] - set(path):
                if next_ == goal:
                    # add path
                    self.path.append(path)
                    # remove goal and calculate next path
                    temp = self.goals.pop(0)
                    if self.goals:
                        # call next goal
                        self.calculate_bfs_path(self.graph, next_,
                                                str(self.goals[0][0]) + "," + str(self.goals[0][1]))
                        # free memory
                        del temp
                    else:
                        # add last goal to path
                        self.path.append([str(temp[0]) + "," + str(temp[1])])
                else:
                    queue.append((next_, path + [next_]))

    # procedure responsible of calculating all possible dfs paths
    def get_bfs_path(self):
        # for all permutations of goals list:
        for self.goals in copy.deepcopy(self.goalsPer):
            # clear bfs and run next permutation
            self.path = []
            # calculate bfs
            start = str(self.x) + "," + str(self.y)
            goal = str(self.goals[0][0]) + "," + str(self.goals[0][1])
            self.calculate_bfs_path(self.graph, start, goal)
            # add parsed bfs_path to solutions
            self.solutions.append(self.parse_dfs_list(self.path))
        # now self.solutions contains all solutions of bfs
    # //////////////////////////////////////////////////

    # Best-First Search

    # procedure responsible for calculating distance heuristics for bestfs
    @staticmethod
    def calculate_bestfs_distance(field, goal):
        fieldCoord = field.split(",")
        goalCoord = goal.split(",")
        dist = math.sqrt(pow(int(fieldCoord[0]) - int(goalCoord[0]), 2)
                         + pow(int(fieldCoord[1]) - int(goalCoord[1]), 2))
        return int(dist)

    # recursive calculation of bestfs path saved temporarly in self.path
    def calculate_bestfs_path(self, graph, start, goal):
        queue = [(self.calculate_bestfs_distance(start, goal), start, [start])]
        heapq.heapify(queue)
        while queue:
            (cost, vertex, path) = heapq.heappop(queue)
            heapq.heapify(queue)
            for next_ in graph[vertex] - set(path):
                if next_ == goal:
                    # add path
                    self.path.append(path)
                    # remove goal and calculate next path
                    temp = self.goals.pop(0)
                    if self.goals:
                        # call next goal
                        self.calculate_bestfs_path(self.graph, next_,
                                                   str(self.goals[0][0]) + "," + str(self.goals[0][1]))
                        # free memory
                        del temp
                    else:
                        # add last goal to path
                        self.path.append([str(temp[0]) + "," + str(temp[1])])
                else:
                    heapq.heappush(queue, (self.calculate_bestfs_distance(next_, goal), next_, path + [next_]))
                    heapq.heapify(queue)

    # procedure responsible of calculating all possible bestfs paths
    def get_bestfs_path(self):
        # for all permutations of goals list:
        for self.goals in copy.deepcopy(self.goalsPer):
            # clear bestfs_path and run next permutation
            self.path = []
            # calculate bestfs
            start = str(self.x) + "," + str(self.y)
            goal = str(self.goals[0][0]) + "," + str(self.goals[0][1])
            self.calculate_bestfs_path(self.graph, start, goal)
            # add parsed bestfs_path to solutions
            self.solutions.append(self.parse_dfs_list(self.path))
        # now self.solutions contains all solutions of bestfs

    # //////////////////////////////////////////////////

    # U N S U P E R V I S E D   L E A R N I N G

    # //////////////////////////////////////////////////

    # datamodel parser

    @staticmethod
    def save(filename, log):
        with open(filename, "a") as myfile:
            myfile.write(log + '\n')

    # calculate neighbourhood from matrix and coordinates of agent
    def get_neighbourhood(self, n):
        # get neighbourhood from matrix and get_coordinates of waiter:
        shift = int((n - 1)/2)  # coefficient of shift

        # use self.get_coordinates() & self.restaurant.get_matrix() to get data required to find neighbourhood
        # matrix = self.restaurant.get_matrix()
        matrix = self.restaurant
        [agent_x, agent_y] = self.get_coordinates()

        # set matrix of neighbourhood - walls by default
        self.neighbourhood = [['Wall' for _ in range(n)] for _ in range(n)]
        self.neighbourhood[shift][shift] = 'Waiter'

        # fill neighbourhood
        for x in range(n):
            for y in range(n):
                # fill matrix of neighbourhood - NOT OPTIMAL, REPAIR: has to run through whole matrix
                # instead of only common part of neighbourhood range and matrix
                if agent_x + x - shift in range(0, self.n) and agent_y + y - shift in range(0, self.n):
                    self.neighbourhood[y][x] = matrix.matrix[agent_x + x - shift][agent_y + y - shift]

    # method used only in model generation, called in UberKelner.py ONLY
    def parse_neighbourhood(self, n):
        # get nieghbourhood of agent and save it to self.neighbourhood
        self.get_neighbourhood(n)
        # parse neighbourhood to data model standard:
        # rabbit:
        # rabbit_standard = ""

        # save neighbourhood AND movement solution to data model for rabbit
        # according to the standard set in documentation/unsupervised_learning.txt
        # self.save("data\datamodel_rabbit.txt", rabbit_standard)

        # save neighbourhood and solution to data model for scikit
        # self.save("data\datamodel_scikit.txt", "scikitoszki")

    # //////////////////////////////////////////////////

    # Rabbit Search - Adam Lewicki & Julia Maria May

    def get_rabbit_path(self, n):
        # get neighbourhood
        self.get_neighbourhood(n)
        # get proposed solution of current state from model


        # set response to path
        self.path = [0, 0]

    # //////////////////////////////////////////////////////

    # SciKit Support Vector Machines Search - Marcin Drzewiczak

    def get_svm_path(self):
        # get neighbourhood
        self.get_neighbourhood(n)
        # get proposed solution of current state

        # set response to path
        self.path = [0, 0]

    # //////////////////////////////////////////////////////

    # SciKit Logistic Regression Search - Michał Kubiak

    def get_logistic_regression_path(self):
        # get neighbourhood
        self.get_neighbourhood(n)
        # get proposed solution of current state

        # set response to path
        self.path = [0, 0]

    # //////////////////////////////////////////////////////

    # SciKit Decision-Tree Search - Przemysław Owczar XD

    def get_decision_tree_path(self):
        # get neighbourhood
        self.get_neighbourhood(n)
        # get proposed solution of current state

        # set response to path
        self.path = [0, 0]

    # //////////////////////////////////////////////////////
