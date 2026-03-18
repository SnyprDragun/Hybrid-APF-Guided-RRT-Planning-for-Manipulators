#!/opt/homebrew/bin/python3
import numpy as np
import random
import matplotlib.pyplot as plt

# -------------------------
# PARAMETERS
# -------------------------

BASE_STEP = 0.4
MAX_ITER = 4000
GOAL_SAMPLE_RATE = 0.1
REWIRE_RADIUS = 1.5

K_ATT = 1.2
K_REP = 2.5
REP_RANGE = 2.5


# -------------------------
# NODE
# -------------------------

class Node:

    def __init__(self, pos):

        self.pos = np.array(pos)
        self.parent = None
        self.cost = 0


# -------------------------
# DISTANCE
# -------------------------

def dist(a,b):
    return np.linalg.norm(a-b)


# -------------------------
# RANDOM BALL SAMPLE
# -------------------------

def random_unit_ball():

    while True:

        x = np.random.uniform(-1,1,3)

        if np.linalg.norm(x) <= 1:
            return x


# -------------------------
# INFORMED SAMPLING
# -------------------------

def informed_sample(start, goal, c_best, bounds):

    if c_best == np.inf:

        return np.array([
            random.uniform(bounds[0][0],bounds[0][1]),
            random.uniform(bounds[1][0],bounds[1][1]),
            random.uniform(bounds[2][0],bounds[2][1])
        ])

    c_min = dist(start,goal)

    center = (start+goal)/2

    a = c_best/2
    b = np.sqrt(c_best**2 - c_min**2)/2

    L = np.diag([a,b,b])

    x_ball = random_unit_ball()

    return center + L @ x_ball


# -------------------------
# NEAREST
# -------------------------

def nearest(tree,point):

    d=[dist(n.pos,point) for n in tree]
    return tree[np.argmin(d)]


# -------------------------
# NEAR
# -------------------------

def near(tree,pos):

    nodes=[]

    for n in tree:

        if dist(n.pos,pos)<REWIRE_RADIUS:
            nodes.append(n)

    return nodes


# -------------------------
# DISTANCE FIELD
# -------------------------

def distance_field(q,obstacles):

    min_d=np.inf
    grad=np.zeros(3)

    for c,r in obstacles:

        vec=q-c
        d=np.linalg.norm(vec)-r

        if d<min_d:

            min_d=d
            grad=vec/np.linalg.norm(vec)

    return min_d,grad


# -------------------------
# APF FORCE
# -------------------------

def apf_force(q,goal,obstacles):

    Fatt=K_ATT*(goal-q)

    Frep=np.zeros(3)

    d,grad=distance_field(q,obstacles)

    if d<REP_RANGE:

        mag=K_REP*(1/d-1/REP_RANGE)/(d**2)

        Frep=mag*grad

    return Fatt+Frep,d


# -------------------------
# STEER
# -------------------------

def steer(q_near,q_rand,goal,obstacles):

    F,d=apf_force(q_near,goal,obstacles)

    direction=(q_rand-q_near)+F

    direction/=np.linalg.norm(direction)

    step=BASE_STEP*(1+min(d,2))

    return q_near+step*direction


# -------------------------
# COLLISION
# -------------------------

def collision_free(p1,p2,obstacles):

    for t in np.linspace(0,1,15):

        p=p1+t*(p2-p1)

        for c,r in obstacles:

            if np.linalg.norm(p-c)<=r:
                return False

    return True


# -------------------------
# PATH
# -------------------------

def extract(node):

    path=[]

    while node:

        path.append(node.pos)
        node=node.parent

    return path[::-1]


# -------------------------
# PLANNER
# -------------------------

def planner(start,goal,bounds,obstacles):

    tree=[Node(start)]

    c_best=np.inf
    best_path=None

    for i in range(MAX_ITER):

        q_rand=informed_sample(start,goal,c_best,bounds)

        near_node=nearest(tree,q_rand)

        q_new=steer(near_node.pos,q_rand,goal,obstacles)

        if not collision_free(near_node.pos,q_new,obstacles):
            continue

        new_node=Node(q_new)

        neighbors=near(tree,q_new)

        best_parent=near_node
        best_cost=near_node.cost+dist(near_node.pos,q_new)

        for n in neighbors:

            if collision_free(n.pos,q_new,obstacles):

                cost=n.cost+dist(n.pos,q_new)

                if cost<best_cost:

                    best_cost=cost
                    best_parent=n

        new_node.parent=best_parent
        new_node.cost=best_cost

        tree.append(new_node)

        for n in neighbors:

            if collision_free(new_node.pos,n.pos,obstacles):

                cost=new_node.cost+dist(new_node.pos,n.pos)

                if cost<n.cost:

                    n.parent=new_node
                    n.cost=cost

        if dist(q_new,goal)<BASE_STEP:

            goal_node=Node(goal)
            goal_node.parent=new_node
            goal_node.cost=new_node.cost+dist(q_new,goal)

            path=extract(goal_node)

            c_best=goal_node.cost
            best_path=path

            print("Improved path cost:",c_best)

    return best_path,tree


# -------------------------
# SPHERE DRAW
# -------------------------

def draw_sphere(ax,c,r):

    u=np.linspace(0,2*np.pi,25)
    v=np.linspace(0,np.pi,25)

    x=r*np.outer(np.cos(u),np.sin(v))+c[0]
    y=r*np.outer(np.sin(u),np.sin(v))+c[1]
    z=r*np.outer(np.ones(len(u)),np.cos(v))+c[2]

    ax.plot_surface(x,y,z,alpha=0.3)


# -------------------------
# VISUALIZATION
# -------------------------

def visualize(tree,path,obstacles,start,goal):

    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')

    for n in tree:

        if n.parent:

            p1=n.pos
            p2=n.parent.pos

            ax.plot(
                [p1[0],p2[0]],
                [p1[1],p2[1]],
                [p1[2],p2[2]],
                linewidth=0.5
            )

    if path:

        p=np.array(path)

        ax.plot(p[:,0],p[:,1],p[:,2],linewidth=4)

    for c,r in obstacles:

        draw_sphere(ax,c,r)

    ax.scatter(*start,s=100)
    ax.scatter(*goal,s=100)

    plt.show()


# -------------------------
# MAIN
# -------------------------

start=np.array([0,0,0])
goal=np.array([9,9,9])

bounds=[(-2,10),(-2,10),(-2,10)]

obstacles=[
(np.array([4,4,4]),0.2),
(np.array([6,6,5]),0.7),
(np.array([3,7,6]),0.6),
(np.array([7,3,5]),0.4)
]

path,tree=planner(start,goal,bounds,obstacles)

visualize(tree,path,obstacles,start,goal)