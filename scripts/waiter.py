# -*- coding: utf-8 -*-
# agent object class:

import itertools
import sys
import time
import math
import heapq
from os import path
from numpy import ndarray
import numpy
import random
from sklearn import svm
from sklearn import tree
from pygame.locals import *
from operator import add

from UberKelner import init_graphics, blocksize
from scripts.matrix import *
from scripts.wall import *

from vowpalwabbit import pyvw
import os

# set recursion limit:
sys.setrecursionlimit(1500)


# init of object with sprite - pygames requirement
# noinspection PyTypeChecker
class Waiter (pygame.sprite.Sprite):

    # procedure of printing object properties when called by matrix
    def __repr__(self):
        return "W"

    # initialize agent with list of coordinates for tables and furnaces and their number
    def __init__(self, n, matrix_fields, num_tables, num_furnaces, num_walls, solving_method):
        print("Agent: initializing object...")

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
        self.restaurant.insert('W', self.x, self.y)

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

        # set unsupervised learning safety switch
        self.moves_queue = [[0, 0], [0, 0], [0, 0], [0, 0]]

        # set neighbourhood
        self.neighbourhood = []
        self.neighbourhood_size = 5

        # set solving method
        self.solving_method = solving_method

        # svm model variable
        self.svm_data = []
        self.svm_target = []
        if self.solving_method == 'svm':
            self.init_svm()
        elif self.solving_method == "dtree":
            self.init_dtree()

        # run solution seeking
        self.solve(self.solving_method)
        self.control = True

        # add steps counter
        self.steps_count = 0

        print("Agent: initialization completed.")

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

        # add step to counter
        self.steps_count = self.steps_count + 1

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
            self.get_svm_path()
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
                print("Agent: path contains %s steps. " % len(self.path))
            else:
                print("Agent: no %s path found!" % self.solving_method)

        elif method in self.unsupervised_learning:
            # set solving method
            self.solving_method = method
            if self.goals:
                if self.solving_method == "rabbit":
                    self.get_rabbit_path()
                elif self.solving_method == "svm":
                    self.get_svm_path()
                elif self.solving_method == "lreg":
                    self.get_logistic_regression_path()
                elif self.solving_method == "dtree":
                    self.get_decision_tree_path()
                # because these methods calculate only one step (not the whole path),
                # they should be called again for next move
                self.control = False

                real_coordinates = list(map(add, self.get_coordinates(), self.path[0]))
                # remove goal if reached
                if real_coordinates in self.goals:
                    self.goals.remove(real_coordinates)

                # check if agent is not moving in circles
                self.next_switch()
            else:
                print("Agent: No goals left!")

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
            (vertex, tpath) = stack.pop()
            for next_ in graph[vertex] - set(tpath):
                if next_ == goal:
                    # add path
                    self.path.append(tpath)
                    # remove goal and calculate next path
                    try:
                        temp = self.goals.pop(0)
                        if self.goals:
                            # call next goal
                            self.calculate_dfs_path(self.graph, next_,
                                                    str(self.goals[0][0]) + "," + str(self.goals[0][1]))
                            # free memory
                            del temp
                        else:
                            # add last goal to path
                            self.path.append([str(temp[0]) + "," + str(temp[1])])
                    except IndexError:
                        print("Agent: map processing error - loop detected")
                        self.path = [[]]
                else:
                    stack.append((next_, tpath + [next_]))

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
            (vertex, tpath) = queue.pop()
            for next_ in graph[vertex] - set(tpath):
                if next_ == goal:
                    # add path
                    self.path.append(tpath)
                    # remove goal and calculate next path
                    try:
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
                    except IndexError:
                        print("Agent: map processing error - loop detected")
                        self.path = [[]]
                else:
                    queue.append((next_, tpath + [next_]))

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
            (cost, vertex, tpath) = heapq.heappop(queue)
            heapq.heapify(queue)
            for next_ in graph[vertex] - set(tpath):
                if next_ == goal:
                    # add path
                    self.path.append(tpath)
                    # remove goal and calculate next path
                    try:
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
                    except IndexError:
                        print("Agent: map processing error - loop detected")
                        self.path = [[]]
                else:
                    heapq.heappush(queue, (self.calculate_bestfs_distance(next_, goal), next_, tpath + [next_]))
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

    # safety switch controller
    def next_switch(self):
        # append last move and remove first
        if self.path[0]:
            self.moves_queue = self.moves_queue[1:]
            self.moves_queue.append(self.path[0])
        # check safety
        if self.moves_queue in [[[-1, 0], [1, 0], [-1, 0], [1, 0]], [[1, 0], [-1, 0], [1, 0], [-1, 0]],
                                [[0, 1], [0, -1], [0, 1], [0, -1]], [[0, -1], [0, 1], [0, -1], [0, 1]]]:
            # if agent is stuck, generate random move
            self.path = [random.choice([[0, -1], [0, 1], [-1, 0], [1, 0]])]

    # datamodel parser

    @staticmethod
    def save(filename, log):
        with open(filename, "a") as myfile:
            myfile.write(log + '\n')

    # calculate neighbourhood from matrix and coordinates of agent
    def get_neighbourhood(self):
        # get neighbourhood from matrix and get_coordinates of waiter:
        shift = int((self.neighbourhood_size - 1)/2)  # coefficient of shift

        # use self.get_coordinates() & self.restaurant.get_matrix() to get data required to find neighbourhood
        # matrix = self.restaurant.get_matrix()
        matrix = self.restaurant
        [agent_x, agent_y] = self.get_coordinates()

        # set matrix of neighbourhood - walls by default
        self.neighbourhood = [["X" for _ in range(self.neighbourhood_size)] for _ in range(self.neighbourhood_size)]
        self.neighbourhood[shift][shift] = "W"

        # fill neighbourhood
        for x in range(self.neighbourhood_size):
            for y in range(self.neighbourhood_size):
                # fill matrix of neighbourhood - NOT OPTIMAL, REPAIR: has to run through whole matrix
                # instead of only common part of neighbourhood range and matrix
                if agent_x + x - shift in range(0, self.n) and agent_y + y - shift in range(0, self.n):
                    self.neighbourhood[y][x] = matrix.matrix[agent_x + x - shift][agent_y + y - shift]

    def parse_neighbourhood_to_rabbit(self):
        # get neighbourhood of agent and save it to self.neighbourhood
        self.get_neighbourhood()
        # parse neighbourhood to data model standard:
        # rabbit:
        convert = {
            "_": 0,
            "X": 1,
            "F": 20,
            "E": 21,
            "T": 30,
            "Y": 31,
            "W": 4
        }
        rabbit_standard = ""
        for x in range(self.neighbourhood_size):
            for y in range(self.neighbourhood_size):
                rabbit_standard += " {}:{}".format(str(x)+"x"+str(y), convert.get(str(self.neighbourhood[x][y])))
        return rabbit_standard

    def parse_neighbourhood_to_scikit(self):
        # get neighbourhood of agent and save it to self.neighbourhood
        self.get_neighbourhood()
        # parse neighbourhood to data model standard:
        # rabbit:
        convert = {
            "_": 0,
            "X": 1,
            "F": 20,
            "E": 21,
            "T": 30,
            "Y": 31,
            "W": 4
        }
        scikit_standard = ""
        for x in range(self.neighbourhood_size):
            for y in range(self.neighbourhood_size):
                scikit_standard += "{}, ".format(convert.get(str(self.neighbourhood[x][y])))
        return scikit_standard

    def parse_neighbourhood_to_scikit_SVM(self):
        # get neighbourhood of agent and save it to self.neighbourhood
        self.get_neighbourhood()
        # parse neighbourhood to data model standard:
        # rabbit:
        convert = {
            "_": 0,
            "X": 0.9,
            "F": 0.20,
            "E": 0.9,
            "T": 0.30,
            "Y": 0.9,
            "W": 0.4
        }

        svm_standard = ndarray(shape=(1, self.neighbourhood_size, self.neighbourhood_size), dtype=float)
        for index_x, x in enumerate(self.neighbourhood_size):
            for index_y, y in enumerate(self.neighbourhood_size):
                svm_standard[0][index_x][index_y] = convert.get(str(self.neighbourhood[index_x][index_y]))
        nsamples, nx, ny = svm_standard
        svm_standard = svm_standard.reshape((nsamples, nx * ny))
        return svm_standard

    # method used only in model generation, called in UberKelner.py ONLY
    def parse_neighbourhood_to_model(self):
        # parse neighbourhood to rabbit string
        moves = {
            "[0, -1]": "W",
            "[0, 1]": "S",
            "[-1, 0]": "A",
            "[1, 0]": "D",
        }
        # there has to be run self.solve("depthfs") before this part, otherwise self.path will be empty
        predicted_move = moves.get(str([self.path[0][0], self.path[0][1]]))  # returns value from "moves"

        rabbit_standard = "{} | ".format(predicted_move)
        rabbit_standard = rabbit_standard + self.parse_neighbourhood_to_rabbit()

        # save neighbourhood AND movement solution to data model for rabbit
        # according to the standard set in documentation/unsupervised_learning.txt
        self.save(path.join('data', 'datamodel_rabbit.txt'), rabbit_standard)

        # save neighbourhood to data model for scikit
        scikit_standard = "{}, ".format(predicted_move)
        scikit_standard = scikit_standard + self.parse_neighbourhood_to_scikit()
        self.save(path.join('data', 'datamodel_scikit.txt'), scikit_standard)

    # //////////////////////////////////////////////////

    # Rabbit Search - Adam Lewicki & Julia Maria May

    def get_rabbit_path(self):
        # get neighbourhood in rabbit
        rabbit_standard = self.parse_neighbourhood_to_rabbit()
        # get proposed solution of current state from model

        moves = {
            1: 'W',
            2: 'S',
            3: 'A',
            4: 'D'
        }

        model = './data/rabbit.model'
        input = './data/rabbit_input.txt'
        output = './data/rabbit_result.txt'

        rabbit_standard = "|{}".format(rabbit_standard)

        with open(input, 'w') as myfile:
            myfile.write(rabbit_standard)

        run_rabbit = 'vw -i {} -t {} -p {} --quiet'.format(model, input, output)
        os.system(run_rabbit)

        with open(output, 'r') as myfile:
            result = int(round(float(myfile.readline())))
        result = moves.get(result)
        print(result)

        # set response to path
        self.path = [[0, 0]]  # this has to be double list!

    # //////////////////////////////////////////////////////

    # SciKit Support Vector Machines Search - Marcin Drzewiczak

    # procedure running in init of agent, loading data to model once
    def init_svm(self):
        self.svm_target = numpy.load(path.join('data', 'svm_target.npy'))
        self.svm_data = numpy.load(path.join('data', 'svm_data.npy'))
        nsamples, nx, ny = self.svm_data.shape
        self.svm_data = self.svm_data.reshape((nsamples, nx * ny))
        # self.svm_data = list(self.svm_data)
        self.clf = svm.SVC(gamma='scale', C=100)
        self.clf.fit(self.svm_data, self.svm_target)

    def init_dtree(self):
        self.svm_target = numpy.load(path.join('data', 'svm_target.npy'))
        self.svm_data = numpy.load(path.join('data', 'svm_data.npy'))
        nsamples, nx, ny = self.svm_data.shape
        self.svm_data = self.svm_data.reshape((nsamples, nx * ny))
        # self.svm_data = list(self.svm_data)
        self.clf = tree.DecisionTreeClassifier()
        self.clf.fit(self.svm_data, self.svm_target)

    def scikit_standard_to_svm_standard(self, scikit_standard):
        try:
            scikit_standard = scikit_standard.split(', ')
            scikit_standard.pop()
        except:
            pass
        scikit_standard = list(map(int, scikit_standard))
        scikit_standard = list(map(lambda a: a/100, scikit_standard))

        iter = 0
        svm_standard = ndarray(shape=(self.neighbourhood_size, self.neighbourhood_size), dtype=float)
        for x in range(self.neighbourhood_size):
            for y in range(self.neighbourhood_size):
                svm_standard[x][y] = scikit_standard[iter]
                iter += 1
        to_return = ndarray(shape=(1, 25), dtype=ndarray)
        to_return[0] = svm_standard.flatten()
        return to_return

    def get_svm_path(self):
        # get neighbourhood in scikit
        scikit_standard = self.parse_neighbourhood_to_scikit()
        svm_standard = self.scikit_standard_to_svm_standard(scikit_standard)
        # get proposed solution of current state from model

        # print(self.svm_data.ndim)
        # print(self.svm_data.shape)
        # print(svm_standard.shape)

        prediction = self.clf.predict(svm_standard)
        moves = {
            'W': [0, -1],
            'S': [0, 1],
            'A': [-1, 0],
            'D': [1, 0],
        }
        # print(prediction)
        move_to_append = moves.get(prediction[0])
        # set response to path
        # this has to be double list!
        self.path.clear()
        self.path.append(move_to_append)

    # //////////////////////////////////////////////////////

    # SciKit Logistic Regression Search - Michał Kubiak

    def get_logistic_regression_path(self):
        # get neighbourhood in scikit
        scikit_standard = self.parse_neighbourhood_to_scikit()
        # get proposed solution of current state from model

        # set response to path
        self.path = [[0, 0]]  # this has to be double list!

    # //////////////////////////////////////////////////////

    # SciKit Decision-Tree Search - Przemysław Owczar XD

    def get_decision_tree_path(self):
        # get neighbourhood in scikit
        scikit_standard = self.parse_neighbourhood_to_scikit()
        svm_standard = self.scikit_standard_to_svm_standard(scikit_standard)
        # get proposed solution of current state from model

        # print(self.svm_data.ndim)
        # print(self.svm_data.shape)
        # print(svm_standard.shape)

        prediction = self.clf.predict(svm_standard)
        moves = {
            'W': [0, -1],
            'S': [0, 1],
            'A': [-1, 0],
            'D': [1, 0],
        }
        # print(prediction)
        move_to_append = moves.get(prediction[0])
        # set response to path
        # this has to be double list!
        self.path.clear()
        self.path.append(move_to_append)

    # //////////////////////////////////////////////////////
