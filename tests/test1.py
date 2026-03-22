#!/opt/homebrew/bin/python3
import numpy as np
import random
import matplotlib.pyplot as plt

# -------------------------------
# PARAMETERS
# -------------------------------

STEP_SIZE = 0.4
MAX_ITER = 2000
GOAL_BIAS = 0.1

K_ATT = 1.0
K_REP = 2.0
REP_RADIUS = 1.5


# -------------------------------
# NODE
# -------------------------------

class Node:
    def __init__(self, position, parent=None):
        self.pos = np.array(position)
        self.parent = parent


# -------------------------------
# DISTANCE
# -------------------------------

def dist(a, b):
    return np.linalg.norm(a - b)


# -------------------------------
# RANDOM SAMPLE
# -------------------------------

def sample(goal, bounds):

    if random.random() < GOAL_BIAS:
        return goal

    return np.array([
        random.uniform(bounds[0][0], bounds[0][1]),
        random.uniform(bounds[1][0], bounds[1][1]),
        random.uniform(bounds[2][0], bounds[2][1])
    ])


# -------------------------------
# NEAREST NODE
# -------------------------------

def nearest(tree, point):

    dists = [dist(n.pos, point) for n in tree]
    return tree[np.argmin(dists)]


# -------------------------------
# ATTRACTIVE FORCE
# -------------------------------

def attractive(q, goal):

    return K_ATT * (goal - q)


# -------------------------------
# REPULSIVE FORCE
# -------------------------------

def repulsive(q, obstacles):

    F = np.zeros(3)

    for center, radius in obstacles:

        d = np.linalg.norm(q - center) - radius

        if d < REP_RADIUS:

            direction = (q - center) / np.linalg.norm(q - center)

            mag = K_REP * (1/d - 1/REP_RADIUS) / (d**2)

            F += mag * direction

    return F


# -------------------------------
# STEER WITH APF
# -------------------------------

def steer(q_near, q_rand, goal, obstacles):

    Fatt = attractive(q_near, goal)
    Frep = repulsive(q_near, obstacles)

    direction = (q_rand - q_near) + Fatt + Frep

    direction = direction / np.linalg.norm(direction)

    return q_near + STEP_SIZE * direction


# -------------------------------
# COLLISION CHECK
# -------------------------------

def collision_free(p1, p2, obstacles):

    for t in np.linspace(0,1,10):

        p = p1 + t*(p2 - p1)

        for center, radius in obstacles:

            if np.linalg.norm(p - center) <= radius:
                return False

    return True


# -------------------------------
# EXTRACT PATH
# -------------------------------

def extract_path(node):

    path = []

    while node is not None:
        path.append(node.pos)
        node = node.parent

    return path[::-1]


# -------------------------------
# APF RRT PLANNER
# -------------------------------

def apf_rrt(start, goal, bounds, obstacles):

    tree = [Node(start)]

    for i in range(MAX_ITER):

        q_rand = sample(goal, bounds)

        nearest_node = nearest(tree, q_rand)

        q_new = steer(nearest_node.pos, q_rand, goal, obstacles)

        if collision_free(nearest_node.pos, q_new, obstacles):

            new_node = Node(q_new, nearest_node)
            tree.append(new_node)

            if dist(q_new, goal) < STEP_SIZE:

                print("Goal reached in", i, "iterations")
                return extract_path(new_node), tree

    return None, tree


# -------------------------------
# VISUALIZATION
# -------------------------------

def visualize(tree, path, obstacles, start, goal):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Tree
    for node in tree:
        if node.parent:
            p1 = node.pos
            p2 = node.parent.pos
            ax.plot([p1[0],p2[0]],
                    [p1[1],p2[1]],
                    [p1[2],p2[2]],
                    'gray', linewidth=0.5)

    # Path
    if path:
        path = np.array(path)
        ax.plot(path[:,0],path[:,1],path[:,2],'r',linewidth=3)

    # Obstacles
    for center, radius in obstacles:
        ax.scatter(center[0],center[1],center[2],s=200)

    ax.scatter(*start, color='green', s=100)
    ax.scatter(*goal, color='red', s=100)

    plt.show()


# -------------------------------
# MAIN
# -------------------------------

start = np.array([0,0,0])
goal = np.array([8,8,8])

bounds = [
    (-2,10),
    (-2,10),
    (-2,10)
]

obstacles = [
    (np.array([4,4,4]),1.5),
    (np.array([6,6,5]),1.2),
    (np.array([3,7,6]),1.2)
]

path, tree = apf_rrt(start, goal, bounds, obstacles)

visualize(tree, path, obstacles, start, goal)