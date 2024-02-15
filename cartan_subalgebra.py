from basic import LieAlgebra
from general_linear import *
import numpy as np
import scipy as sp

def generalized_null_space(L: LieAlgebra, A: np.ndarray) -> np.ndarray:
    A_power = np.linalg.matrix_power(A, L.dimension)
    null_subspace_A = sp.linalg.null_space(A_power).transpose()
    dim_null_subspace = null_subspace_A.shape[0]
    subspace_basis = np.concatenate([null_subspace_A, 
                                     np.zeros((L.dimension - dim_null_subspace, L.dimension))], axis=0)
    L_sub = sub_lie_algebra(L, subspace_basis)
    return (dim_null_subspace, L_sub)

def non_nilpotent_element(L: LieAlgebra) -> np.ndarray:
    """Find a non-nilpotent element in the subspace

    Args:
        L (LieAlgebra): Lie algebra
        subspace_basis (np.ndarray): basis of the subspace

    Returns:
        np.ndarray: non-nilpotent element
    """
    for i in range(L.dimension):
        if not is_ad_nilpotent(L, L.basis[i, :]):
            return L.basis[i, :]
    
    for i in range(L.dimension):
        for j in range(i + 1, L.dimension):
            combined = L.basis[i, :] + L.basis[j, :]
            if not is_ad_nilpotent(L, combined):
                return combined
    return None
    
def cartan_subalgebra(L: LieAlgebra):
    """Find the Cartan subalgebra of a Lie algebra

    Args:
        L (LieAlgebra): Lie algebra

    Returns:
        np.ndarray: Cartan subalgebra basis
    """
    a = non_nilpotent_element(L)
    
    if a is None:
        return L
    # print(L.basis, a)
    ada = adjoint(L, a)
    
    ada = np.linalg.matrix_power(ada, L.dimension)
    null_subspace_ada = sp.linalg.null_space(ada).transpose()
    subspace_basis = np.concatenate([null_subspace_ada, np.zeros((L.dimension - null_subspace_ada.shape[0], L.dimension))], axis=0)
    # print(null_subspace_ada.shape)
    K = sub_lie_algebra(L, subspace_basis)
    # print(subspace_basis)
    b = non_nilpotent_element(K)
    ready = b is None
    while not ready:
        found = False
        c = 0
        while not found:
            c += 1
            new_elment = a + c * (b - a)
            ad_new = adjoint(K, new_elment)
            rank_ad_new = np.linalg.matrix_rank(ad_new)
            ad_new = np.linalg.matrix_power(ad_new, rank_ad_new)
            null_subspace_ad_new = sp.linalg.null_space(ad_new).transpose()
            found = null_subspace_ad_new.shape[0] < subspace_basis.shape[0]
            for i in range(null_subspace_ad_new.shape[0]):
                if not is_in_subspace(K, subspace_basis, null_subspace_ad_new[i, :]):
                    found = False
            a = new_elment
            subspace_basis = np.concatenate([null_subspace_ad_new, np.zeros((L.dimension - null_subspace_ad_new.shape[0], L.dimension))], axis=0)
            K = sub_lie_algebra(L, subspace_basis)
            b = non_nilpotent_element(K)
            ready = b is None
    return K
            

if __name__ == "__main__":
    basis_sl2 = np.stack([E_unit(2, 0, 0) - E_unit(2, 1, 1), 
                          E_unit(2, 0, 1), E_unit(2, 1, 0)])
    basis_sl3 = np.stack([E_unit(3, 0, 0) - E_unit(3, 1, 1),
                        E_unit(3, 1, 1) - E_unit(3, 2, 2),                        
                        E_unit(3, 0, 1),
                        E_unit(3, 0, 2),
                        E_unit(3, 1, 2),
                        E_unit(3, 1, 0),
                        E_unit(3, 2, 0),
                        E_unit(3, 2, 1)])

    GL = GeneralLinear(basis=basis_sl3)
    L = LieAlgebra(structure_constant=GL._structure_constant())
    x_mat = np.array([[1, 2, 3], 
                    [4, 50, 6],
                    [7, 8,-51]])

    y_mat = np.array([[2, 3, 4],
                    [5, 60, 7],
                    [8, 9,-62]])

    adx = np.array([[0, 1, 2],
                    [0, 0, 1],
                    [0, 0, 0]])
    basis_num = GL.basis.shape[0]
    # print(L.structure_constant)
    # print(np.linalg.matrix_rank(B))
    # print(is_nilpotent_end(adx))
    
    subspace_basis = np.array(
        [[1, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]]
    )
    b = non_nilpotent_element(L)
    print(b)
    CSA = cartan_subalgebra(L)
    print(CSA.basis)
    # new_coord = np.array([0, 100, 0, 0, 0, 0, 0, 0])
    # new_b = find_bracket_closed_b(GL, subspace_basis, 2)
    # # print(bracket(L._as_matrix(subspace_basis[1, :]), L._as_matrix(new_b)))
    # print(find_cartan_subalgebra(GL))
    # print(is_in_subspace(subspace_basis=subspace_basis, new_coord=new_coord))
    