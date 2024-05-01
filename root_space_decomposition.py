from basic import *
from general_linear import *
from cartan_subalgebra import *
import numpy as np

def root_space_decomposition(L: LieAlgebra, H: LieSubalgebra):
    """Decompose Lie algebra into root space

    Args:
        L (LieAlgebra): Lie algebra
        H (LieSubalgebra): Cartan subalgebra

    Returns:
        dict: root space decomposition
    """
    roots = np.zeros((L.dimension, H.dimension))
    for i in range(H.dimension):
        roots[:, i] = np.linalg.eigvals(adjoint(L, H.basis[i]))
    return roots

def regular_vector(roots):
    """Find regular vector in root space

    Args:
        roots (dict): root space decomposition

    Returns:
        np.ndarray: regular vector
    """
    gamma = np.zeros(roots.shape[1])
    while np.all(np.dot(roots, gamma) == 0):
        gamma = np.random.rand(roots.shape[1])
    return gamma

def positive_roots_from_regular(roots, gamma):
    """Find positive roots from regular vector

    Args:
        roots (dict): root space decomposition
        gamma (np.ndarray): regular vector

    Returns:
        np.ndarray: positive roots
    """
    return roots[np.dot(roots, gamma) > 0, :]


def base(roots):
    gamma = regular_vector(roots)
    roots_pr = positive_roots_from_regular(roots, gamma)
    num_roots, dim = roots_pr.shape
    roots_pr_reshaped = roots_pr.reshape(num_roots, 1, dim)
    sum_roots = roots_pr + roots_pr_reshaped
    sum_roots = sum_roots.reshape(-1, dim)
    simple_roots = []
    for i in range(num_roots):
        if not np.any(np.all(sum_roots==roots_pr[i], axis=1)):
            simple_roots.append(roots_pr[i])
    
    simple_roots = np.stack(simple_roots, axis = 0)
    indices = np.lexsort(simple_roots.T)
    simple_roots = simple_roots[indices]

    return simple_roots

def pairing(r1, r2):
    return 2 * np.dot(r1, r2) / np.dot(r2, r2)

def cartan_matrix_from_base(base):
    rank = base.shape[0]
    cartan_matrix = np.zeros((rank, rank))
    for i in range(rank):
        for j in range(rank):
            cartan_matrix[i, j] = pairing(base[i], base[j])
    return cartan_matrix


if __name__ == "__main__":
    basis_so3 = type_B_basis(5)
    GL = GeneralLinear(basis=basis_so3)
    L = LieAlgebra(structure_constant=GL._structure_constant())
    H = cartan_subalgebra(L)
    # print(H.basis)
    # print(adjoint(L, L.basis[1]))
    # for i in range(H.dimension):
    #     print(np.linalg.eigvals(adjoint(L, H.basis[i])))
    roots = root_space_decomposition(L, H)
    gamma = regular_vector(roots)
    # print(gamma)
    prr = positive_roots_from_regular(roots, gamma)
    # print(prr)
    base11 = base(roots)
    
    # print(base11)
    cartan_matrix = cartan_matrix_from_base(base11)
    print(f"Cartan matrix for Type B: \n {cartan_matrix}")
    