#  NOTES:
#  spsolve changes the matrix -> probably gauss elim or smth: 1st time ~ 0.07s, 2nd,3rd,… time - 0.03s
#

import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import copy

n_vec = [50, 100, 150]  # nr of grid points per side
length = 1  # length of the side of the isosceles triangle
sigma_vec = [0, 1, 10, 50, 100, -1, -10, -50, -100]  # constant space charge ~ f(x,y)

#  Which grid size to use 0-3
index = 0
#  TODO: kako je z razlicnimi grid size-i, oz step-size?
n = n_vec[index]
delta_x = 1 / n  # step size
delta_y = 1 / n
sigma = sigma_vec[8]
M = sum(range(n + 1))  # nr of all points - size of the u (solution) vector
N = sum(range(n + 1))  # the domain is an isosceles triangle.

delta_l = (pow(delta_x, -2) + pow(delta_y, -2)) / 2


def build_matrix_v2():
    # Matrix size = sum(1,N)
    m_size = N
    #print(m_size)

    # L = np.zeros((m_size, m_size))
    diag0 = np.ones((1, m_size)) * -4 * delta_l
    diag1 = np.ones((1, m_size - 1)) * pow(delta_x, -2)
    # diagn1 = np.ones((1, m_size - 1)) * pow(delta_x, -2)
    idx = 0
    for i in range(n, -1, -1):
        idx = idx + i
        if idx < m_size:
            diag1[0][idx-1] = 0

    mrx = np.diag(diag1[0], -1) + np.diag(diag0[0], 0) + np.diag(diag1[0], 1)
    # print(L)

    block_n = n
    counter = 0
    w = pow(delta_y, -2)
    for j in range(0, m_size):
        if counter < block_n - 1:
            mrx[j][j+block_n] = w
            counter = counter + 1
        else:
            block_n = block_n - 1
            counter = 0

    L = np.maximum(mrx, mrx.transpose())
    S = scipy.sparse.csc_matrix(L)
    return S, L


def build_l_sparse_matrix():
    """
    DEPRECATED - use build_matrix_v2
    Build the final L matrix of dimensions N x N by building the BLOCKS first, then building a DENSE matrix
    and convert the DENSE matrix into a SPARSE matrix.
    Need m - number of rows and n - number of columns
    :return:
    """
    l_blocks = []
    lu_blocks = []
    ll_blocks = []

    for i in range(1, n + 1):
        d0 = [np.ones(i) * -4 * delta_l, np.ones(i - 1) * pow(delta_x, -2), np.ones(i - 1) * pow(delta_x, -2)]
        pos = [0, 1, -1]
        l_blocks.append(sp.sparse.diags(d0, pos).toarray())

    # upper diag
    for i in range(1, n + 1):
        lu_blocks.append(np.eye(i - 1) * pow(delta_y, -2))

    # lower diag
    for i in range(1, n + 1):
        ll_blocks.append(np.eye(i) * pow(delta_y, -2))

    # concatenate start with at the (0,0) point in the triangle. j=0, i=1...n
    # leading zeros, lower diag, L, upper diag, zeros
    tmp1 = np.concatenate((lu_blocks[n-1], np.zeros((1, n-1))), axis=0)
    tmp1 = np.concatenate((l_blocks[n-1], tmp1), axis=1)
    L = np.concatenate((tmp1, np.zeros((tmp1.shape[0], N - tmp1.shape[1]))), axis=1)

    for row in range(n-2, 0, -1):
        tmp1 = np.concatenate((lu_blocks[row], np.zeros((1, lu_blocks[row].shape[0]))), axis=0)  # make lu full
        tmp1 = np.concatenate((l_blocks[row], tmp1), axis=1)                                     # add lu to main d (l)
        tmp2 = np.concatenate((ll_blocks[row], np.zeros((ll_blocks[row].shape[0], 1))), axis=1)  # make ll full
        tmp1 = np.concatenate((tmp2, tmp1), axis=1)                                              # add ll to main d (l)
        if n-1 - row > 1:
            # leading zeros
            dim = sum(range(row+3, n+1))
            tmp1 = np.concatenate((np.zeros((tmp1.shape[0], dim)), tmp1),
                                  axis=1)  # sej je to ta prav variable uporabljen?
        # zeros at the end
        tmp1 = np.concatenate((tmp1, np.zeros((tmp1.shape[0], N - tmp1.shape[1]))), axis=1)
        L = np.concatenate((L, tmp1), axis=0)

    # last row
    last_row = np.concatenate((np.zeros((1, 1)), l_blocks[0]), axis=1)
    last_row = np.concatenate((ll_blocks[0], last_row), axis=1)
    last_row = np.concatenate((np.zeros((1, N-last_row.shape[1])), last_row), axis=1)

    L = np.concatenate((L, last_row), axis=0)

    S = scipy.sparse.csc_matrix(L)

    return S, L


def to_m(i, j):
    """
    Different than in the notes! i in j swapped - NE DELA, SPREMEN ZA TRIKOTNIK!
    :param i:
    :param j:
    :return:
    """
    # return i + (j-1)*(n-j/2+1)
    return sum(range(n, n-j+1, -1)) + i


def to_i(m):
    for i in range(m, 0, -1):
        if 1 + (i-1)*(N-i/2+1) <= m:
            return i, m - (i - 1) * (N - i / 1 + 1)


def build_right_hand_side():
    """
    Builds the boundary vector and fills it with values of the boundary condition.
    Builds the f vector - constant space charge.
    Adds them together.
    :return:
    """
    b = np.zeros((N, 1))
    #  i = 1 (0) and iterate over j
    for j in range(1, n+1):
        b[int(round(to_m(1, j)))-1] = +1 * pow(delta_y, -2)
    #  j = 1 (0) and iterate over i
    for i in range(1, n+1):
        b[int(round(to_m(i, 1)))-1] = -1 * pow(delta_x, -2)
    #  diagonal is 0 anyway
    return - np.ones((N, 1)) * sigma - b


def visualize(u):
    """
    Visualize the results.
    :param u:
    :return:
    """
    fig = plt.figure()
    # ax = Axes3D(fig)

    # Make data.
    X = np.arange(0, length+delta_x, delta_x)
    Y = np.arange(0, length+delta_y, delta_y)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros((round(length/delta_x), round(length/delta_y)))
    j = 0
    idx = 0
    for i in range(n, 0, -1):
        Z[j][0:i] = u.tolist()[idx:idx+i]
        idx = idx + i
        j = j + 1

    plt.pcolormesh(X, Y, Z.transpose(), cmap=cm.coolwarm)
    plt.colorbar()
    title = "Potential function - the solution of the Poisson equation\nn = " + str(n) + " | sigma = " + str(sigma)
    plt.title(title)
    plt.show()


def main():
    full_time_dense = 0.0
    full_time_sparse = 0.0

    # Sparse
    for i in range(1, 6):
        t1 = time.time()
        S, L = build_matrix_v2()
        # S1, L1 = build_l_sparse_matrix()
        print("Time | Building Matrix | ", str(time.time() - t1), "s.")
        rhs = build_right_hand_side()
        t1 = time.time()
        u_sparse = linalg.spsolve(S, rhs)
        t2 = time.time()
        full_time_sparse = full_time_sparse + t2-t1
        print("Time | Sparse | Calc | ", t2-t1, "s.")
        t1 = time.time()
        u_dense = np.linalg.solve(L, rhs)
        t2 = time.time()
        print("Time | Dense | Calc | ", t2-t1, "s.")
        full_time_dense = full_time_dense + t2-t1
        print("----------------")

    print("Avg Time - sparse matrix (calculation only): ", full_time_sparse/5, "s.")
    print("Avg Time - dense matrix (calculation only): ", full_time_dense/5, "s.")
    visualize(u_sparse)


if __name__ == '__main__':
    main()
