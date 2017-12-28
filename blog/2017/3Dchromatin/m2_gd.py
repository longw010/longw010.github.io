import numpy as np
import time


def init(n, r=1.5):
    node_posi = np.zeros((n, 3), dtype=float)
    for i in range(n):
        node_posi[i, :] = np.random.uniform(-r, r, 3)
    return node_posi


def weights():
    # weights setting for chr1
    W = np.zeros(4, dtype=float)
    W[0], W[1], W[2], W[3] = 1.0, 0.4, 0.4, 0.4
    return W


def sech2(x):
    """
    sech(x)
    Uses numpy's cosh(x).
    """
    return np.power(1. / np.cosh(x), 2)


def get_dist(x, y):
    return np.linalg.norm(x - y)


def dist_const():
    # distant setting; note that all are in square.
    dist = dict()
    dist['max'] = 30 ** 2
    dist['min'] = 0.2**2
    dist['dc'] = 6 ** 2
    dist['damax'] = 1.5 ** 2
    return dist


def load_data(file_name):
    ## load data
    with open(file_name, 'r') as f:
        # get the number of nodes
        num_node = int(f.readline().strip())-19

        # get the value of totalIF
        #totalIF = float(f.readline().strip())
        totalIF=1000

        # get the value of IFmax
        IFmax = float(f.readline().strip())

        # initialize F
        F = np.zeros((num_node, num_node), dtype=float) - 1

        for line in f:
            submatrix = line.strip().split('\t')
            idx1 = int(submatrix[0])-1
            idx2 = int(submatrix[1])-1
            if idx1 > 121:
                idx1 -= 19
            if idx2 > 121:
                idx2 -= 19
            F[idx1][idx2] = float(submatrix[2])
            F[idx2][idx1] = float(submatrix[2])
        return num_node, totalIF, IFmax, F


def run(filename, dconst, W, epochs=10, lr=0.00001):
    ## load data
    num_node, totalIF, IFmax, F = load_data(filename)

    ## initialize vars
    node_posi = init(num_node)
    loss = []

    ## go thru epoch
    for epoch in range(epochs):
        # calculate pairwise distance
        print('--epoch- ' + str(epoch))
        node_dist = pairwise_dist(node_posi)
        # calculate loss
        current_loss = cal_loss(node_dist, F)
        print(current_loss)
        loss += [current_loss]

        # calculate gradient
        for i in range(num_node):
            df_cache = cal_df_cache(node_posi, node_dist, dconst, W, IFmax,
                                    totalIF, F)
            dx, dy, dz = df_cache[i][0], df_cache[i][1], df_cache[i][2]
            #print(df_cache[i])
            # print (dx, dy, dz)
            node_posi[i][0] += lr * dx
            node_posi[i][1] += lr * dy
            node_posi[i][2] += lr * dz
        #simple_visulization(node_posi)
    return loss, node_posi


def cal_df_cache(node_posi, node_dist, dconst, W, IFmax, totalIF, F):
    num_node = len(node_dist)
    Fn = np.zeros((num_node, 3), dtype=float)
    for i in range(num_node):
        # it is symmetic
        #temp = 0.0
        num_outlink = 0
        for j in range(num_node):
            # check whether it has valid F[i][j]
            if F[i][j] < 0 or F[j][i] < 0 or i == j:
                continue

            i, j = min(i, j), max(i, j)
            num_outlink += 1
            current_dist = node_dist[i][j] ** 2
            # when |i-j| = 1
            if j == i + 1:
                temp = - W[0] * IFmax * sech2(current_dist - dconst['damax']) + W[1] * sech2(current_dist - dconst['min'])
            else:
                # when it is in contact
                if current_dist < dconst['dc']:
                    temp = - W[0] * sech2(current_dist - dconst['dc']) * F[i][j] * 2 * node_dist[i][j] + W[1] * sech2(
                        current_dist - dconst['min'])
                # when it is not in contact
                else:
                    temp = - W[2] * sech2(current_dist - dconst['max']) + W[3] * sech2(current_dist - dconst['dc'])
            # *2 is from gradient
            temp *= 2
            # divide const from d{d_ij} / d{x}
            for coordinate in range(3):
                Fn[i][coordinate] += temp * (node_posi[i][coordinate] - node_posi[j][coordinate])

        # normalize the gradient add
        Fn[i, :] /= num_outlink
    return Fn



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


def cal_loss_obj(node_dist, dconst, W, IFmax, totalIF, F):
    num_node = len(node_dist)
    Fn = 0.0
    count = 0
    for i in range(num_node):
        # it is symmetic
        for j in range(i + 1, num_node):
            # check whether it has valid F[i][j]
            if F[i][j] < 0:
                continue

            current_dist = node_dist[i][j] ** 2
            # print(current_dist)
            # when |i-j| = 1
            if j == i + 1:
                temp = W[0] * IFmax * np.tanh(dconst['damax'] - current_dist) + W[1] * np.tanh(current_dist - dconst['min'])
                # print('1', temp)
            else:
                # when it is in contact
                if current_dist < dconst['dc']:
                    temp = W[0] * np.tanh(dconst['dc'] - current_dist) * F[i][j] + W[1] * np.tanh(current_dist - dconst['min'])
                    # print('2', temp)
                # when it is not in contact
                else:
                    temp = W[2] * np.tanh(dconst['max'] - current_dist) + W[3] * np.tanh(current_dist - dconst['dc'])
                    # print('3', temp)
            Fn += temp
            count += 1
    return Fn * 2


def pairwise_dist(node_posi):
    # init variables
    num_node = len(node_posi)
    dist = np.zeros((num_node, num_node), dtype=float)

    for i in range(num_node):
        for j in range(i + 1, num_node):
            dist[i][j] = get_dist(node_posi[i, :], node_posi[j, :])
    return dist


def simple_visulization(node_posi):
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import matplotlib.pyplot as plt

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    N = len(node_posi)
    # for i in range(N-1):
    # https://stackoverflow.com/questions/15617207/line-colour-of-3d-parametric-curve-in-pythons-matplotlib-pyplot
    # ax.plot(node_posi[i:i+2, 0], node_posi[i:i+2, 1], node_posi[i:i+2, 2], color=plt.cm.jet(255*i/N))

    ax.plot(node_posi[:, 0], node_posi[:, 1], node_posi[:, 2],
            label='parametric curve')
    ax.legend()
    #fig.savefig('test.pdf')  # save the figure to file
    plt.show()


def print_result(node_posi):
    file = open('output.txt', 'w')
    for element in node_posi:
        file.write('\t'.join(map(str, element)) + '\n')
    file.close()
    
if __name__ == "__main__":
    start = time.time()
    W = weights()
    dconst = dist_const()
    filename = 'IFList_Chr_1_1mb.txt'
    _, node_posi = run(filename, dconst, W, lr=0.1, epochs=2)
    #print_result(node_posi)
    print(time.time() - start)