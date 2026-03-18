#!/opt/homebrew/bin/python3
import numpy as np
import random
import matplotlib.pyplot as plt


# --------------------------------
# PARAMETERS
# --------------------------------

STEP_SIZE = 0.5
MAX_ITER = 3000
GOAL_SAMPLE_RATE = 0.1
REWIRE_RADIUS = 1.2

K_ATT = 1.0
K_REP = 2.0
REP_INFLUENCE = 2.0

SMOOTH_ITERS = 150


# --------------------------------
# NODE CLASS
# --------------------------------

class Node:

    def __init__(self, pos):

        self.pos = np.array(pos)
        self.parent = None
        self.cost = 0.0


# --------------------------------
# DISTANCE
# --------------------------------

def distance(a,b):
    return np.linalg.norm(a-b)


# --------------------------------
# RANDOM SAMPLING
# --------------------------------

def sample(goal, bounds):

    if random.random() < GOAL_SAMPLE_RATE:
        return goal

    return np.array([
        random.uniform(bounds[0][0], bounds[0][1]),
        random.uniform(bounds[1][0], bounds[1][1]),
        random.uniform(bounds[2][0], bounds[2][1])
    ])


# --------------------------------
# NEAREST NODE
# --------------------------------

def nearest(tree, point):

    d = [distance(n.pos, point) for n in tree]
    return tree[np.argmin(d)]


# --------------------------------
# NEAR NODES (for RRT*)
# --------------------------------

def near_nodes(tree, new_pos):

    nodes = []

    for n in tree:
        if distance(n.pos, new_pos) < REWIRE_RADIUS:
            nodes.append(n)

    return nodes


# --------------------------------
# ATTRACTIVE FORCE
# --------------------------------

def attractive(q, goal):

    return K_ATT * (goal - q)


# --------------------------------
# REPULSIVE FORCE
# --------------------------------

def repulsive(q, obstacles):

    F = np.zeros(3)

    for center, radius in obstacles:

        vec = q - center
        dist = np.linalg.norm(vec) - radius

        if dist < REP_INFLUENCE:

            grad = vec / np.linalg.norm(vec)

            mag = K_REP * (1/dist - 1/REP_INFLUENCE) / (dist**2)

            F += mag * grad

    return F


# --------------------------------
# STEER WITH APF
# --------------------------------

def steer(q_near, q_rand, goal, obstacles):

    F_att = attractive(q_near, goal)
    F_rep = repulsive(q_near, obstacles)

    direction = (q_rand - q_near) + F_att + F_rep

    direction = direction / np.linalg.norm(direction)

    return q_near + STEP_SIZE * direction


# --------------------------------
# COLLISION CHECK
# --------------------------------

def collision_free(p1, p2, obstacles):

    for t in np.linspace(0,1,15):

        p = p1 + t*(p2-p1)

        for center, radius in obstacles:

            if np.linalg.norm(p-center) <= radius:
                return False

    return True


# --------------------------------
# PATH EXTRACTION
# --------------------------------

def extract_path(node):

    path = []

    while node is not None:

        path.append(node.pos)
        node = node.parent

    return path[::-1]


# --------------------------------
# PATH SMOOTHING
# --------------------------------

def smooth_path(path, obstacles):

    if path is None:
        return None

    for _ in range(SMOOTH_ITERS):

        if len(path) < 3:
            break

        i = random.randint(0,len(path)-2)
        j = random.randint(i+1,len(path)-1)

        if collision_free(path[i], path[j], obstacles):

            path = path[:i+1] + path[j:]

    return path


# --------------------------------
# APF RRT* PLANNER
# --------------------------------

def apf_rrt_star(start, goal, bounds, obstacles):

    tree = []
    start_node = Node(start)

    tree.append(start_node)

    for i in range(MAX_ITER):

        q_rand = sample(goal, bounds)

        nearest_node = nearest(tree, q_rand)

        q_new = steer(nearest_node.pos, q_rand, goal, obstacles)

        if not collision_free(nearest_node.pos, q_new, obstacles):
            continue

        new_node = Node(q_new)

        near = near_nodes(tree, q_new)

        best_parent = nearest_node
        best_cost = nearest_node.cost + distance(nearest_node.pos, q_new)

        for n in near:

            if collision_free(n.pos, q_new, obstacles):

                cost = n.cost + distance(n.pos, q_new)

                if cost < best_cost:

                    best_cost = cost
                    best_parent = n

        new_node.parent = best_parent
        new_node.cost = best_cost

        tree.append(new_node)

        # RRT* rewiring
        for n in near:

            if collision_free(new_node.pos, n.pos, obstacles):

                new_cost = new_node.cost + distance(new_node.pos, n.pos)

                if new_cost < n.cost:

                    n.parent = new_node
                    n.cost = new_cost

        if distance(new_node.pos, goal) < STEP_SIZE:

            goal_node = Node(goal)
            goal_node.parent = new_node
            tree.append(goal_node)

            print("Goal reached in iteration:", i)

            return extract_path(goal_node), tree

    return None, tree


# --------------------------------
# DRAW SPHERE
# --------------------------------

def draw_sphere(ax, center, radius):

    u = np.linspace(0,2*np.pi,30)
    v = np.linspace(0,np.pi,30)

    x = radius*np.outer(np.cos(u),np.sin(v)) + center[0]
    y = radius*np.outer(np.sin(u),np.sin(v)) + center[1]
    z = radius*np.outer(np.ones(np.size(u)),np.cos(v)) + center[2]

    ax.plot_surface(x,y,z,alpha=0.3)


# --------------------------------
# VISUALIZATION
# --------------------------------

def visualize(tree, path, obstacles, start, goal):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # tree
    for node in tree:

        if node.parent:

            p1 = node.pos
            p2 = node.parent.pos

            ax.plot(
                [p1[0],p2[0]],
                [p1[1],p2[1]],
                [p1[2],p2[2]],
                linewidth=0.5
            )

    # path
    if path:

        path = np.array(path)

        ax.plot(
            path[:,0],
            path[:,1],
            path[:,2],
            linewidth=4
        )

    # obstacles
    for center, radius in obstacles:

        draw_sphere(ax, center, radius)

    ax.scatter(*start,s=80)
    ax.scatter(*goal,s=80)

    plt.show()


# --------------------------------
# MAIN
# --------------------------------

start = np.array([0,0,0])
goal = np.array([9,9,9])

bounds = [
    (-2,10),
    (-2,10),
    (-2,10)
]

obstacles = [
    (np.array([4,4,4]),0.8),
    (np.array([6,5,6]),0.5),
    (np.array([3,7,6]),0.5),
    (np.array([7,3,5]),0.3)
]


path, tree = apf_rrt_star(start, goal, bounds, obstacles)

path = smooth_path(path, obstacles)

visualize(tree, path, obstacles, start, goal)