#!/opt/homebrew/bin/python3
import pybullet as p
import pybullet_data
import numpy as np
import random
import time

# -------------------------------------------------
# PARAMETERS
# -------------------------------------------------

MAX_ITER = 3500
STEP_SIZE = 0.15
GOAL_SAMPLE_RATE = 0.1
OBSTACLE_SAMPLE_RATE = 0.25
REWIRE_RADIUS = 0.4

K_ATT = 1.2
K_REP = 2.5
REP_RANGE = 0.6

JOINT_LIMIT_LOW = np.array([-2.9,-1.8,-2.9,-3.0,-2.9,-0.1])
JOINT_LIMIT_HIGH = np.array([2.9,1.8,2.9,0.0,2.9,3.7])


# -------------------------------------------------
# NODE
# -------------------------------------------------

class Node:

    def __init__(self,q):

        self.q = np.array(q)
        self.parent = None
        self.cost = 0


# -------------------------------------------------
# ENVIRONMENT
# -------------------------------------------------

def setup_environment():

    p.connect(p.GUI)
    p.setGravity(0,0,-9.81)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.loadURDF("plane.urdf")

    robot = p.loadURDF(
        "franka_panda/panda.urdf",
        useFixedBase=True
    )

    obstacles = []

    for i in range(6):

        pos = np.random.uniform([-0.4,-0.4,0.2],[0.4,0.4,0.6])

        obs = p.loadURDF(
            "sphere_small.urdf",
            pos,
            globalScaling=2
        )

        obstacles.append(obs)

    return robot, obstacles


# -------------------------------------------------
# ROBOT FUNCTIONS
# -------------------------------------------------

def set_configuration(robot,q):

    for i in range(7):
        p.resetJointState(robot,i,q[i])


def get_ee_position(robot,q):

    set_configuration(robot,q)

    link = p.getLinkState(robot,11)

    return np.array(link[0])


def get_jacobian(robot,q):

    zero = [0]*len(q)

    Jlin,_ = p.calculateJacobian(
        robot,
        11,
        [0,0,0],
        q,
        zero,
        zero
    )

    return np.array(Jlin)


# -------------------------------------------------
# OBSTACLE POSITIONS
# -------------------------------------------------

def obstacle_positions(obstacles):

    pos = []

    for o in obstacles:

        p0,_ = p.getBasePositionAndOrientation(o)

        pos.append(np.array(p0))

    return pos


# -------------------------------------------------
# APF FORCE (WORKSPACE)
# -------------------------------------------------

def apf_force(x,goal,obstacles):

    Fatt = K_ATT*(goal-x)

    Frep = np.zeros(3)

    for c in obstacles:

        d = np.linalg.norm(x-c)

        if d < REP_RANGE:

            grad = (x-c)/d

            mag = K_REP*(1/d - 1/REP_RANGE)/(d**2)

            Frep += mag*grad

    return Fatt + Frep


# -------------------------------------------------
# PROJECTED DIRECTION
# -------------------------------------------------

def projected_direction(robot,q,goal,obs):

    x = get_ee_position(robot,q)

    F = apf_force(x,goal,obs)

    J = get_jacobian(robot,q)

    dq = J.T @ F

    if np.linalg.norm(dq) < 1e-6:
        return np.zeros(len(q))

    return dq/np.linalg.norm(dq)


# -------------------------------------------------
# COLLISION CHECK
# -------------------------------------------------

def collision_free(robot,obstacles,q1,q2):

    for t in np.linspace(0,1,10):

        q = q1 + t*(q2-q1)

        set_configuration(robot,q)

        for o in obstacles:

            pts = p.getClosestPoints(
                robot,
                o,
                distance=0.02
            )

            if len(pts)>0:
                return False

    return True


# -------------------------------------------------
# SAMPLING
# -------------------------------------------------

def random_sample():

    return np.random.uniform(
        JOINT_LIMIT_LOW,
        JOINT_LIMIT_HIGH
    )


def obstacle_bias_sample(robot,obs_pos):

    center = random.choice(obs_pos)

    q = random_sample()

    noise = np.random.normal(0,0.1,len(q))

    return q + noise


def sample(robot,goal_q,obs_pos):

    r = random.random()

    if r < GOAL_SAMPLE_RATE:
        return goal_q

    if r < GOAL_SAMPLE_RATE + OBSTACLE_SAMPLE_RATE:
        return obstacle_bias_sample(robot,obs_pos)

    return random_sample()


# -------------------------------------------------
# TREE UTILITIES
# -------------------------------------------------

def nearest(tree,q):

    d = [np.linalg.norm(n.q-q) for n in tree]

    return tree[np.argmin(d)]


def near(tree,q):

    nodes=[]

    for n in tree:

        if np.linalg.norm(n.q-q) < REWIRE_RADIUS:
            nodes.append(n)

    return nodes


# -------------------------------------------------
# STEER
# -------------------------------------------------

def steer(robot,q_near,q_rand,goal,obs):

    dq_apf = projected_direction(robot,q_near,goal,obs)

    dq_rand = q_rand - q_near

    direction = dq_rand + dq_apf

    direction /= np.linalg.norm(direction)

    return q_near + STEP_SIZE*direction


# -------------------------------------------------
# PATH
# -------------------------------------------------

def extract_path(node):

    path=[]

    while node:

        path.append(node.q)

        node=node.parent

    return path[::-1]


# -------------------------------------------------
# RRT* PLANNER
# -------------------------------------------------

def planner(robot,start,goal_q,goal_x,obstacles):

    tree=[Node(start)]

    obs_pos = obstacle_positions(obstacles)

    for i in range(MAX_ITER):

        q_rand = sample(robot,goal_q,obs_pos)

        near_node = nearest(tree,q_rand)

        q_new = steer(robot,near_node.q,q_rand,goal_x,obs_pos)

        if not collision_free(robot,obstacles,near_node.q,q_new):
            continue

        new_node = Node(q_new)

        neighbors = near(tree,q_new)

        best_parent = near_node
        best_cost = near_node.cost + np.linalg.norm(near_node.q-q_new)

        for n in neighbors:

            if collision_free(robot,obstacles,n.q,q_new):

                cost = n.cost + np.linalg.norm(n.q-q_new)

                if cost < best_cost:

                    best_parent = n
                    best_cost = cost

        new_node.parent = best_parent
        new_node.cost = best_cost

        tree.append(new_node)

        # rewiring
        for n in neighbors:

            if collision_free(robot,obstacles,new_node.q,n.q):

                cost = new_node.cost + np.linalg.norm(new_node.q-n.q)

                if cost < n.cost:

                    n.parent = new_node
                    n.cost = cost

        if np.linalg.norm(q_new-goal_q) < STEP_SIZE:

            goal_node = Node(goal_q)

            goal_node.parent = new_node

            tree.append(goal_node)

            print("Goal reached in iteration:",i)

            return extract_path(goal_node)

    return None


# -------------------------------------------------
# EXECUTE PATH
# -------------------------------------------------

def execute(robot,path):

    for q in path:

        set_configuration(robot,q)

        for _ in range(30):
            p.stepSimulation()
            time.sleep(0.01)


# -------------------------------------------------
# MAIN
# -------------------------------------------------

robot,obstacles = setup_environment()

start = np.zeros(7)

goal_position = np.array([0.4,0.3,0.4])

goal_q = p.calculateInverseKinematics(
    robot,
    11,
    goal_position
)

goal_q = np.array(goal_q[:7])

path = planner(robot,start,goal_q,goal_position,obstacles)

if path:

    print("Executing path...")

    execute(robot,path)

else:

    print("No path found")