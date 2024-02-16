from basic import LieAlgebra
from general_linear import *
import numpy as np
import scipy as sp

def generalized_null_space(L: LieAlgebra, a: np.ndarray) -> np.ndarray:
    A = adjoint(L, a)
    A_power = np.linalg.matrix_power(A, L.dimension)
    subspace_basis = sp.linalg.null_space(A_power).transpose()
    return subspace_basis

def non_nilpotent_element(L: LieAlgebra) -> np.ndarray:
    """Find a non-nilpotent element in the subspace

    Args:
        L (LieAlgebra): Lie algebra
        subspace_basis (np.ndarray): basis of the subspace

    Returns:
        np.ndarray: non-nilpotent element
    """
    for i in range(L.dimension):
        if not is_ad_nilpotent(L, L.basis[i]):
            # print("testadn", i, L.basis[i], presentation(L, L.basis[i]), adjoint(L, L.basis[i]), "endtestadn")
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
    # print("sss", ada)
    ada = np.linalg.matrix_power(ada, L.dimension)
    null_subspace_ada = sp.linalg.null_space(ada).transpose()
    subspace_basis = null_subspace_ada 
    # print("p", subspace_basis.shape)
    K = SubLieAlgebra(L, subspace_basis)
    b = non_nilpotent_element(K)
    # print(b)
    
    ready = (b is None)
    # print(ready)
    while not ready:
        found = False
        c = 0
        while not found:
            c += 1
            
            new_elment = a + c * (b - a)
            ad_new = adjoint(K, new_elment)
            # print(new_elment, ad_new)
            ad_new = np.linalg.matrix_power(ad_new.transpose(), K.dimension)
            
            null_subspace_ad_new = sp.linalg.null_space(ad_new).transpose()
            # print(null_subspace_ad_new.shape, K.basis.shape)
            null_subspace_ad_new = null_subspace_ad_new @ K.basis
            
            found = null_subspace_ad_new.shape[0] < subspace_basis.shape[0]
            for i in range(null_subspace_ad_new.shape[0]):
                if found:
                    if not is_in_subspace(K, subspace_basis, null_subspace_ad_new[i, :]):
                        found = False
                        # print("sss")
            # print(c, subspace_basis.shape, null_subspace_ad_new.shape, found)
        a = new_elment
        # print("p", null_subspace_ad_new.shape)
        subspace_basis = null_subspace_ad_new
        print(subspace_basis)
        K = SubLieAlgebra(L, subspace_basis)
        b = non_nilpotent_element(K)
        # print(K.basis)
        ready = (b is None)
    return K

def GAP_command_for_creation(structure_constants: np.ndarray) -> str:
    """Generate GAP command for creation of Lie algebra

    Args:
        structure_constants (np.ndarray): structure constants

    Returns:
        str: GAP command
    """
    basis_num = structure_constants.shape[0]
    str = f"T:= EmptySCTable({basis_num}, 0);;\n"
    for i in range(structure_constants.shape[0]):
        for j in range(structure_constants.shape[1]):
            line = f"SetEntrySCTable(T, {i+1}, {j+1}, ["
            linek = ""
            for k in range(structure_constants.shape[2]):
                if structure_constants[i, j, k] != 0:
                    linek += f"{int(structure_constants[i, j, k])}, {k+1}, "
            if linek != "":
                line += linek[:-2]
                line += "]);"
                str += line + "\n"
    return str

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
    basis_so3 = type_B_basis(3)
    GL = GeneralLinear(basis=basis_so3)
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
    matrix_num = GL.basis.shape[1]
    print(L.structure_constant)
    str = ""
    for i in range(L.structure_constant.shape[0]):
        for j in range(L.structure_constant.shape[1]):
            line = f"SetEntrySCTable(T, {i+1}, {j+1}, ["
            linek = ""
            for k in range(L.structure_constant.shape[2]):
                if L.structure_constant[i, j, k] != 0:
                    linek += f"{int(L.structure_constant[i, j, k])}, {k+1}, "
            if linek != "":
                line += linek[:-2]
                line += "]);"
                str += line + "\n"
    print(GAP_command_for_creation(L.structure_constant))
    # print(np.linalg.matrix_rank(B))
    # print(is_nilpotent_end(adx))
    
    # subspace_basis = np.array(
    #     [[1, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 1, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0]]
    # )
    # b = non_nilpotent_element(L)
    # # print(GL._as_matrix(np.array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])))
    # print(b)
    # print(is_sub_Lie_algebra(L, L.basis))
    CSA = cartan_subalgebra(L)
    print(CSA.basis)
    # # new_coord = np.array([0, 100, 0, 0, 0, 0, 0, 0])
    # # new_b = find_bracket_closed_b(GL, subspace_basis, 2)
    # # # print(bracket(L._as_matrix(subspace_basis[1, :]), L._as_matrix(new_b)))
    # # print(find_cartan_subalgebra(GL))
    # # print(is_in_subspace(subspace_basis=subspace_basis, new_coord=new_coord))
    