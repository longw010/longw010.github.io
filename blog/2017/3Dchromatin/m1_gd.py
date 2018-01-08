# not pairwised
import numpy as np
from scipy import sparse as S
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

np.random.seed(10)

def get_dist(x, y):
    return np.linalg.norm(x-y)

def get_value(A, i, j):
    if i < j:
        return A[j, i]
    else:
        return A[i, j]

def init_posi(r):
    x = np.random.uniform(-r, r, 3)

    while True:
        if x[0] ** 2 + x[1] ** 2 + x[2] ** 2 > r ** 2:
            x = np.random.uniform(-r, r, 3)
        else:
            return x
        
def initialize(num_node, r=4.):
    '''
    arguments
    r : estimated radius
    '''

    # init variables
    node = np.zeros((num_node, 3), dtype=float)
    for i in range(num_node):
        node[i] = init_posi(r)
    return node


def solver(file_name, epochs, init_lr=0.01, isShowed=False):
    ## initialize variables
    loss = []

    ## load data
    with open(file_name, 'r') as f:
        # get the number of nodes
        num_node = int(f.readline().strip())

        # initialize matrics
        real_dist = np.zeros((num_node, num_node), dtype=float) - 1

        for line in f:
            submatrix = line.strip().split()
            idx1 = int(submatrix[0])
            idx2 = int(submatrix[1])
            real_dist[idx1][idx2] = float(submatrix[2])
            real_dist[idx2][idx1] = float(submatrix[2])

    # get initialized
    node_posi = initialize(num_node)
    #node_posi = real_dist
    ## go thru epochs
    for epoch in range(epochs):
        lr = init_lr / np.power(2, (epoch / 200))
        print('---' + str(epoch) + '---')
        # calculate previous loss
        node_dist = pairwise_dist(node_posi)
        current_loss = cal_loss(node_dist, real_dist)
        print('Current loss is ', current_loss)
        loss += [current_loss]

        # calculate gradient
        # 1 gradient cache
        cache = derive_cache(node_dist, real_dist)
        # 2 update x, y, z
        for i in range(num_node):
            dx, dy, dz = 0., 0., 0.
            # calulate updated loss
            count = 0
            for j in range(num_node):
                dx += get_value(cache, i, j) * (node_posi[i][0] - node_posi[j][0])
                dy += get_value(cache, i, j) * (node_posi[i][1] - node_posi[j][1])
                dz += get_value(cache, i, j) * (node_posi[i][2] - node_posi[j][2])
                count += 1
            node_posi[i][0] -= lr * dx / count
            node_posi[i][1] -= lr * dy / count
            node_posi[i][2] -= lr * dz / count
        if isShowed:
            simple_visulization(node_posi)
        #if epoch > 900 and epoch % 10 == 0:
        #simple_visulization(node_posi, epoch)
    return loss, node_posi


def derive_cache(node_dist, real_dist):
    num_node = len(node_dist)
    dist = np.zeros((num_node, num_node), dtype=float)

    for i in range(num_node):
        for j in range(i+1, num_node):
            if real_dist[i][j] > 0:
                cache = 2 * (node_dist[i][j] - real_dist[i][j]) / (real_dist[i][j]**2 * node_dist[i][j])
                dist[i][j] = cache
                dist[j][i] = cache
    return dist


def pairwise_dist(node_posi):
    # init variables
    num_node = len(node_posi)
    dist = np.zeros((num_node, num_node), dtype=float)

    for i in range(num_node):
        for j in range(i+1, num_node):
            current_dist = get_dist(node_posi[i, :], node_posi[j, :])
            dist[i][j] = current_dist
            dist[j][i] = current_dist
    return dist


def cal_loss(node_posi, real_dist):
    num = len(node_posi)
    loss = 0.0
    count = 0
    for i in range(num):
        for j in range(i+1, num):
            if real_dist[i][j] > 0:
                loss += (node_posi[i][j] - real_dist[i][j])**2 / real_dist[i][j]**2
                count += 1
    return 2 * loss / count

def simple_visulization(node_posi, epoch):
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import matplotlib.pyplot as plt

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    N = len(node_posi)
    #for i in range(N-1):
        #https://stackoverflow.com/questions/15617207/line-colour-of-3d-parametric-curve-in-pythons-matplotlib-pyplot
        #ax.plot(node_posi[i:i+2, 0], node_posi[i:i+2, 1], node_posi[i:i+2, 2], color=plt.cm.jet(255*i/N))

    ax.plot(node_posi[:, 0], node_posi[:, 1], node_posi[:, 2],label='parametric curve')
    ax.legend()
    plt.savefig('vis' + str(epoch) +'.JPG')
    #plt.show()


if __name__ == "__main__":
    import time
    start = time.time()
    loss, posi = solver('benchmark/chr1.txt', 250, init_lr = 0.1, isShowed=False)
    #simple_visulization(posi)
    #plt.show()
    print(time.time() - start)
    #plt.figure()
    #plt.plot(loss)
    
    plt.savefig('chr1_loss.pdf')
    #file = open('chr3_posi.txt', 'w')
    #for element in posi:
    #    print('\t'.join(map(str, element)), file=file)
    #file.close()
    #plt.show()
