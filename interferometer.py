import numpy as np
from qoqo import operations, Circuit

def T(m, n, theta, phi, nmax):
    """The Clements T matrix"""
    mat = np.identity(nmax, dtype=np.complex128)
    mat[m, m] = np.exp(1j * phi) * np.cos(theta)
    mat[m, n] = -np.sin(theta)
    mat[n, m] = np.exp(1j * phi) * np.sin(theta)
    mat[n, n] = np.cos(theta)
    return mat


def T_inv(m, n, theta, phi, nmax):
    return np.transpose(T(m, n, theta, -phi, nmax))


def T_null(m, n, U):
    """T matrix to nullify the corresponding U elements"""
    (nmax, mmax) = U.shape

    if nmax != mmax:
        raise ValueError("U is not a square matrix")

    if U[m, n] == 0:
        # nothing here
        theta = 0
        phi = 0
    elif U[m-1, n] == 0:
        # swap in the divide-by-zero case
        theta = np.pi / 2
        phi = 0
    else:
        # No-trivial case
        r = -U[m, n] / U[m-1, n]
        theta = np.arctan(np.abs(r))
        phi = np.angle(r)

    return [m-1, m, theta, phi, nmax]


def inv_T_null(m, n, U):
    """Inverse T matrix to nullify the corresponding U elements"""
    (nmax, mmax) = U.shape

    if nmax != mmax:
        raise ValueError("U must be a square matrix")

    if U[m, n] == 0:
        # no swaps for the identity-like case
        theta = 0
        phi = 0
    elif U[m, n + 1] == 0:
        # swap in the divide-by-zero case
        theta = np.pi / 2
        phi = 0
    else:
        r = U[m, n] / U[m, n + 1]
        theta = np.arctan(np.abs(r))
        phi = np.angle(r)

    return [n, n + 1, theta, phi, nmax]


def interf_rect_decompose(U:np.array):
    # Check whether U is unitary
    mat = np.matrix(U)
    if not np.allclose(np.eye(mat.shape[0]), mat.H @ mat):
        raise ValueError("The interferometer matrix is not unitary")
    else:
        print("The interferometer matrix is unitary")

    n_channels = mat.shape[0]
    print(n_channels)
    print(mat)

    # Initiate two-channel transformations
    # Lists with parameters
    T_list = []
    T_inv_list = []
    for k, i in enumerate(range(n_channels -2, -1, -1)):
        print('k=', k, 'i=', i)
        # Even
        if k % 2 == 0:
            print('even')
            for j in reversed(range(n_channels - 1 - i)):
                print('j=', j)
                # Find nullyfying T_inv
                #t_inv = []
                t_inv = inv_T_null(i + j + 1, j, mat)
                #print(t_inv)
                # Update U
                mat = mat @ T_inv(*t_inv)
                # Update list of inverse T
                T_inv_list.append(t_inv)
                #print('Matrix after T_inv')
                #print(mat)
        else:
            print('odd')
            for j in range(n_channels - 1 - i):
                #print('Mat before T')
                #print(mat)
                # Find nullyfuing T
                #t = []
                t = T_null(i + j + 1, j, mat)
                #print('Transform')
                #print(T(*t))
                # Update U
                mat = T(*t) @ mat
                T_list.append(t)
                #print('Matrix after T')
                #print(mat)

    return T_inv_list, np.diag(mat), T_list

#
# Read the unitaries
#
U1 = np.load('U1.npy')
U2 = np.load('U2.npy')
#
# Decompose
#
T_i, diag, T = interf_rect_decompose(U2)
print('T_i', T_i)
print('diag', diag)
print('T', T)

#
# Convert numbers to operations
#
decomposition_thresh = 1.0e-13

# T_i
ops = []
print('T_inv', len(T_i))
for n, m, theta, phi, _ in T_i:  # [0:len(T_i[0])-1]:
    theta = theta if np.abs(theta) >= decomposition_thresh else 0
    phi = phi if np.abs(phi) >= decomposition_thresh else 0
    print(n, m, "theta = ", theta, "phi = ", phi)
    if phi != 0:
        # Phase shift
        print(operations.PhaseShift(n, phi))
        ops.append(operations.PhaseShift(n, phi))
    if theta != 0:
        # Beam splitter
        print(operations.BeamSplitter(n, m, theta, 0))
        ops.append(operations.BeamSplitter(n, m, theta, 0))

# Diagonal part
print('Diagonal')
for n, expphi in enumerate(diag):
    # Local phase shifts

    if np.abs(expphi - 1) >= decomposition_thresh:
        q = np.log(expphi).imag
    else:
        q = 0
    if (q != 0):
        # print(sf.ops.Rgate(np.mod(q, 2 * np.pi)), n)
        print(operations.PhaseShift(n, np.mod(q, 2*np.pi)))
        ops.append(operations.PhaseShift(n, np.mod(q, 2*np.pi)))

# T
print('T')
for n, m, theta, phi, _ in reversed(T):
    theta = theta if np.abs(theta) >= decomposition_thresh else 0
    phi = phi if np.abs(phi) >= decomposition_thresh else 0
    print(n, m, "theta = ", theta, "phi = ", phi)

    if theta != 0:
        # beamsplitter
        print(operations.BeamSplitter(n, m, -theta, 0))
        ops.append(operations.BeamSplitter(n, m, -theta, 0))

    if phi != 0:
        # phaseshift
        print(operations.PhaseShift(n, -phi))
        ops.append(operations.PhaseShift(n, -phi))

print('Interferometer decomposition')
for op in ops:
    print(op)
