from basic import LieAlgebra
from general_linear import *
import numpy as np
import scipy as sp

def generalized_null_space(L: LieAlgebra, a: np.ndarray) -> np.ndarray:
    A = adjoint(L, a)
    A_power = np.linalg.matrix_power(A, L.dimension)
    subspace_basis = sp.linalg.null_space(A_power).transpose() @ L.basis
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
    subspace_basis = generalized_null_space(L, a)
    K = LieSubalgebra(L, subspace_basis)
    b = non_nilpotent_element(K)
    
    ready = (b is None)
    while not ready:
        found = False
        c = 0
        k = K.dimension
        while not found:
            c += 1
            
            new_elment = a + c * (b - a)
            null_subspace_ad_new = generalized_null_space(K, new_elment)
            
            found = null_subspace_ad_new.shape[0] < k
            for i in range(null_subspace_ad_new.shape[0]):
                if found and not is_in_subspace(K, null_subspace_ad_new[i]):
                    found = False
        a = new_elment
        # subspace_basis = null_subspace_ad_new
        K = LieSubalgebra(L, null_subspace_ad_new)
        b = non_nilpotent_element(K)
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
    basis_so3 = type_A_basis(4)
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

    CSA = cartan_subalgebra(L)
    print(CSA.basis)
    km = Killing_form_restriction_matrix(L, CSA)
    print(km)

    