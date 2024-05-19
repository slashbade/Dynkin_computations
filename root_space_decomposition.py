from basic import *
from general_linear import *
from cartan_subalgebra import *
import numpy as np

def root_space_decomposition(L: LieAlgebra, H: LieSubalgebra) -> np.ndarray:
    """Decompose Lie algebra into root space

    Args:
        L (LieAlgebra): Lie algebra
        H (LieSubalgebra): Cartan subalgebra

    Returns:
        np.ndarray: root space decomposition
    """
    roots = np.zeros((L.dimension, H.dimension))
    for i in range(H.dimension):
        roots[:, i] = np.linalg.eigvals(adjoint(L, H.basis[i]))
    return roots

def regular_vector(roots):
    """Find regular vector in root space, using randomness

    Args:
        roots (np.ndarray): root space decomposition

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


def base(roots: np.ndarray):
    """Compute simple roots

    Args:
        roots (np.ndarray): root system

    Returns:
        np.ndarray: simple roots
    """
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

def inner_product(L: LieAlgebra, H: LieSubalgebra, alpha: np.ndarray, beta: np.ndarray):
    """Using the nondegenerency to compute inner product (.,.)

    Args:
        L (LieAlgebra): Lie algebra
        H (LieSubalgebra): Cartan Subalgebra
        alpha (np.ndarray): root in H dual
        beta (np.ndarray): root in H dual
    """
    K = Killing_form_restriction_matrix(L, H)
    # Find dual vector with respect to restricted Killing form
    t_alpha = np.linalg.solve(K, alpha)
    t_beta = np.linalg.solve(K, beta)
    prod = Killing_form(L, t_alpha @ H.basis, t_beta @ H.basis)
    return prod

def pairing(L: LieAlgebra, H: LieSubalgebra, alpha, beta):
    """Pairing between roots <.,.>
    """
    return 2 * inner_product(L, H, alpha, beta) / inner_product(L, H, beta, beta)

def sort_simple_roots(matrix: np.ndarray, sorted = []):
    """a simple root sorting method to make cartan matrix look standard

    Args:
        matrix (np.ndarray): Cartan matrix
    """
    negative_counts = np.sum(matrix < 0, axis=1)
    row_index = np.where(negative_counts == 1)[0]
    if row_index.shape[0] == 0:
        row_index_l = np.where(np.sum(matrix > 0, axis=1) > 0)[0]
        return sorted + row_index_l.tolist()
    else:
        row_index = row_index[0]
        matrix[:, row_index] = np.zeros(matrix.shape[0])
        matrix[row_index, :] = np.zeros(matrix.shape[0])
        sorted.append(row_index)
        return sort_simple_roots(matrix, sorted)


def cartan_matrix_from_base(L: LieAlgebra, H: LieSubalgebra, base):
    """Using base and pairing to obtain Cartan matrix
    """
    rank = base.shape[0]
    cartan_matrix = np.zeros((rank, rank))
    for i in range(rank):
        for j in range(rank):
            cartan_matrix[i, j] = pairing(L, H, base[i], base[j])
    return cartan_matrix


if __name__ == "__main__":
    basis_so3 = type_C_basis(4)
    GL = GeneralLinear(basis=basis_so3)
    L = LieAlgebra(structure_constant=GL._structure_constant())
    H = cartan_subalgebra(L)
    # print(H.basis)
    # # print(adjoint(L, L.basis[1]))
    # # for i in range(H.dimension):
    # #     print(np.linalg.eigvals(adjoint(L, H.basis[i])))
    roots = root_space_decomposition(L, H)
    # print(roots)
    # gamma = regular_vector(roots)
    # # print(gamma)
    # prr = positive_roots_from_regular(roots, gamma)
    # # print(prr)
    base11 = base(roots)
    
    print(base11)
    # cartan_matrix = cartan_matrix_from_base(L, H, base11)
    # print(f"Cartan matrix for Type B: \n {cartan_matrix}")
    # sorted_roots = sort_simple_roots(cartan_matrix)
    # print(f"Sorted simple roots: \n {sorted_roots}")
    # print(np.round(cartan_matrix_from_base(L, H, base11[sorted_roots])))
    