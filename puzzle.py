
from __future__ import division
from __future__ import print_function

import sys
import math
import time
import queue as Q
import resource
from heapq import heappush, heappop

#### SKELETON CODE ####
## The Class that Represents the Puzzle
class PuzzleState(object):
    """
        The PuzzleState stores a board configuration and implements
        movement instructions to generate valid children.
    """
    def __init__(self, config, n, parent=None, action="Initial", cost=0):
        """
        :param config->List : Represents the n*n board, for e.g. [0,1,2,3,4,5,6,7,8] represents the goal state.
        :param n->int : Size of the board
        :param parent->PuzzleState
        :param action->string
        :param cost->int
        """
        if n*n != len(config) or n < 2:
            raise Exception("The length of config is not correct!")
        if set(config) != set(range(n*n)):
            raise Exception("Config contains invalid/duplicate entries : ", config)

        self.n        = n
        self.cost     = cost
        self.parent   = parent
        self.action   = action
        self.config   = config
        self.children = []

        # Get the index and (row, col) of empty block
        self.blank_index = self.config.index(0)

    def display(self):
        """ Display this Puzzle state as a n*n board """
        for i in range(self.n):
            print(self.config[3*i : 3*(i+1)])

    def move_up(self):
        """ 
        Moves the blank tile one row up.
        :return a PuzzleState with the new configuration
        """
        list = self.config[:]

        if self.blank_index < self.n:
            return None
        
        list[self.blank_index], list[self.blank_index - 3] = list[self.blank_index - 3], list[self.blank_index]

        return PuzzleState(list, n = self.n, parent = self, action = "Up", cost = self.cost + 1)
      
    def move_down(self):
        ### STUDENT CODE GOES HERE ###
        """
        Moves the blank tile one row down.
        :return a PuzzleState with the new configuration
        """
        
        list = self.config[:]
        
        if self.blank_index >= self.n * (self.n - 1): 
            return None

        list[self.blank_index], list[self.blank_index + 3] = list[self.blank_index + 3], list[self.blank_index]

        return PuzzleState(list, n = self.n, parent = self, action = "Down", cost = self.cost + 1)

        
      
    def move_left(self):
        ### STUDENT CODE GOES HERE ###
        """
        Moves the blank tile one column to the left.
        :return a PuzzleState with the new configuration
        """
        list = self.config[:]
        
        if self.blank_index in [3*i for i in range(self.n)]:
            return None

        list[self.blank_index], list[self.blank_index - 1] = list[self.blank_index - 1], list[self.blank_index]

        return PuzzleState(list, self.n, parent = self, action = "Left", cost = self.cost + 1)

    def move_right(self):
        ### STUDENT CODE GOES HERE ###
        """
        Moves the blank tile one column to the right.
        :return a PuzzleState with the new configuration
        """
        list = self.config[:]
        
        if self.blank_index in [(self.n - 1 + 3*i) for i in range(self.n)]:
            return None

        list[self.blank_index], list[self.blank_index + 1] = list[self.blank_index + 1], list[self.blank_index]

        return PuzzleState(list, self.n, parent = self, action = "Right", cost = self.cost + 1)
      
    def expand(self):
        """ Generate the child nodes of this node """
        
        # Node has already been expanded
        if len(self.children) != 0:
            return self.children
        
        # Add child nodes in order of UDLR
        children = [
            self.move_up(),
            self.move_down(),
            self.move_left(),
            self.move_right()]

        # Compose self.children of all non-None children states
        self.children = [state for state in children if state is not None]
        return self.children

# Function that Writes to output.txt

### Students need to change the method to have the corresponding parameters
def writeOutput(path, depth, expanded, cost, max_depth, time, ram):
    ### Student Code Goes here
    f = open("output.txt", 'w')
    f.write("path to goal: {path} \n".format(path = path)) 
    f.write("cost_of_path: {cost} \n".format(cost = cost))
    f.write("nodes_expanded: {expanded} \n".format(expanded = expanded))
    f.write("search_depth: {depth} \n".format(depth = depth))
    f.write("max_search_depth: {max_depth} \n".format(max_depth = max_depth))
    f.write("running_time: {running_time} \n".format(running_time = time))
    f.write("max_ram_usage: {ram} \n".format(ram = ram))
    f.close()
    

class Frontier(object):

    def __init__(self):
        self.queue = Q.deque()
        self.dict = {} # for bookeeping & fast lookups

    def pop(self) -> PuzzleState:
        '''LIFO POP'''
        delete = self.queue.pop()
        self.dict.pop(create_index(delete))
        return delete

    def isEmpty(self) -> bool:
        return len(self.queue) == 0

    def popleft(self) -> PuzzleState:
        '''FIFO POP'''
        delete = self.queue.popleft()
        self.dict.pop(create_index(delete))
        return delete

    def append(self, state:PuzzleState):
        self.queue.append(state)
        self.dict[create_index(state)] = 1

    def find(self, state):
        return self.dict.get(create_index(state))


def create_index(state):
    return str(state.config)

def depth (state: PuzzleState) -> int:
    '''
    A helper method to calculate the depth of a state (node).
    '''
    depth = 0
    while state.parent:
        depth += 1
        state = state.parent

    return depth

def path_from_root(state: PuzzleState) -> list:
    '''
    A helper method that, given a state, returns the list of operations executed that led from the root to this state.
    '''
    path = []
    while state.parent:
        path.insert(0, state.action)
        state = state.parent
    
    return path

def bfs_search(initial_state):
    """BFS search"""
    #function performs a BFS search given an initial state until the goal is found; returns final state when finished.
    
    frontier = Frontier()
    frontier.append(initial_state)
    explored = set()
    expanded = 0
    max_depth = 0
    start_time = time.time()

    while frontier: #queue not empty
        state = frontier.popleft() # pop leftmost element in the queue (FIFO)
        explored.add(create_index(state))
        
        if test_goal(state):

            if frontier:
                max_depth = depth(frontier.pop()) # pop the last element to be inserted to the frontier
            else:
                max_depth = depth(state)
            
            end_time = time.time()

            writeOutput(path_from_root(state), depth(state), expanded, state.cost, max_depth, end_time - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 10**(-6))

            return 

        expanded += 1
        
        for child in state.expand():
            idx = create_index(child)
            if not (idx in explored or frontier.find(child)): 
                frontier.append(child)

    return False
    
def dfs_search(initial_state):
    """DFS search"""
    ### STUDENT CODE GOES HERE ###
    
    frontier = Frontier()
    frontier.append(initial_state)
    explored = set()

    expanded = 0
    max_depth = 0
    start_time = time.time()

    while frontier:
        state = frontier.pop() # pop the last element (LIFO)
        explored.add(create_index(state))
        
        if test_goal(state): #current state is goal state

            end_time = time.time()
            writeOutput(path_from_root(state), depth(state), expanded, state.cost, max_depth, end_time - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 10**(-6))
            
            return 
        
        children = reversed(state.expand())
        expanded += 1

        for child in children:
            idx = create_index(child)
            if not (idx in explored or frontier.find(child)): #since I do not pop elements from the Frontier dictionary, it includes the explored ones.
                frontier.append(child)

                if child.cost > max_depth:
                    max_depth = child.cost

    return False

def A_star_search(initial_state):
    """A * search"""
    ### STUDENT CODE GOES HERE ###
    entries = {} # a hashmap of frontier elements for fast membership lookups
    frontier = []
    heappush(frontier, (0, 0, initial_state))
    entries[create_index(initial_state)] = 1
    explored = set()
    count = 0
    expanded = 0
    max_depth = 0

    start_time = time.time()

    while frontier:

        tuple = heappop(frontier)
        state = tuple[2]
        explored.add(create_index(state))
        del entries[create_index(state)]

        if test_goal(state): 
            end_time = time.time()
            writeOutput(path_from_root(state), depth(state), expanded, state.cost, max_depth, end_time - start_time, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 10**(-6))
            return None
        
        children = state.expand()
        expanded += 1

        for child in children:
            idx = create_index(child)
            if not (idx in explored or entries.get(idx)):
                count += 1
                fn = calculate_total_cost(child)
                heappush(frontier, (fn, count, child))
                entries[create_index(child)] = 1
                if child.cost > max_depth:
                    max_depth = child.cost

    return



def calculate_total_cost(state):
    """calculate the total estimated cost of a state"""
    ### STUDENT CODE GOES HERE ###
    config = state.config
    g = state.cost
    h = 0
    for i in range(9):
        if config[i] != 0:
            h += calculate_manhattan_dist(i, config[i], state.n)

    return g + h

def calculate_manhattan_dist(idx, value, n):
    """calculate the manhattan distance of a tile"""
    ### STUDENT CODE GOES HERE ###
    x,y = idx % n, int(idx/n)
    goal_x, goal_y = value % n, int(value/n)
    
    return abs(goal_x - x) + abs(goal_y - y)


def test_goal(puzzle_state):
    """test the state is the goal state or not"""
    ### STUDENT CODE GOES HERE ###
    return puzzle_state.config == list(range(9))

# Main Function that reads in Input and Runs corresponding Algorithm
def main():
    search_mode = sys.argv[1].lower()
    begin_state = sys.argv[2].split(",")
    begin_state = list(map(int, begin_state))
    board_size  = int(math.sqrt(len(begin_state)))
    hard_state  = PuzzleState(begin_state, board_size)
    # hard_state.display()
    
    start_time  = time.time()
    
    if   search_mode == "bfs": bfs_search(hard_state)
    elif search_mode == "dfs": dfs_search(hard_state)
    elif search_mode == "ast": A_star_search(hard_state)
    else: 
        print("Enter valid command arguments !")
        
    end_time = time.time()
    print("Program completed in %.3f second(s)"%(end_time-start_time))


if __name__ == '__main__':
    main()
