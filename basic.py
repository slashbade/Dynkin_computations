import numpy as np


class LieAlgebra:
    def __init__(self, structure_constant: np.ndarray, basis: np.ndarray = None) -> None:
        """Initialize Lie algebra with structure constant

        Args:
            structure_constant (np.ndarray): structure constant in the shape (basis_num, basis_num, basis_num)
        """
        self.structure_constant = structure_constant
        self.dimension = structure_constant.shape[0]
        self.basis = np.eye(self.dimension)
        if basis is not None:
            self.basis = basis
            
    
def adjoint(L: LieAlgebra, x: np.ndarray) -> np.ndarray:
    """Obtain adjoint representation in the matrix form

    Args:
        x (np.ndarray): matrix in GLn

    Returns:
        np.ndarray: adjoint representation of shape (basis_num, basis_num)
    """
    adjoint_mat = np.zeros((L.dimension, L.dimension))
    for i in range(L.dimension):
        adjoint_mat[i, :] = np.dot(L.structure_constant[:, :, i].transpose(), x)
    return adjoint_mat

def bracket(L: LieAlgebra, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Computes the bracket of two vectors

    Args:
        x (np.ndarray): vector in L
        y (np.ndarray): vector in L

    Returns:
        np.ndarray: bracket of x and y
    """
    bracket_prod = np.zeros(L.dimension)
    for i in range(L.dimension):
        bracket_prod[i] = L.structure_constant[:, :, i] @ y @ x
    
    return bracket_prod

def is_ad_nilpotent(L: LieAlgebra, x: np.ndarray) -> bool:
    """Justify if a matrix is nilpotent

    Args:
        x (np.ndarray): matrix in GLn

    Returns:
        bool: nilpotency of end
    """
    tol = 1e-8
    adx = adjoint(L, x)
    eigvals = np.linalg.eigvals(adx)
    if all(np.abs(eigvals) <= tol):
        return True
    else:
        return False

def is_in_subspace(L: LieAlgebra, subspace_basis: np.ndarray, new_coord: np.ndarray) -> bool:
    """Justify if a vector is in the subspace spanned by basis

    Args:
        subspace_basis (np.ndarray): basis of subspace, each line is a basis
        new_coord (np.ndarray): new vectot

    Returns:
        bool: if vector in the subspace
    """
    subspace_basis = subspace_basis.transpose()
    matrix_rank = np.linalg.matrix_rank(subspace_basis)
    ext_matrix = np.concatenate([subspace_basis, new_coord.reshape((-1, 1))], axis=1)
    ext_matrix_rank = np.linalg.matrix_rank(ext_matrix)
    if matrix_rank >= ext_matrix_rank:
        return True
    return False

def is_sub_lie_algebra(L: LieAlgebra, subspace_basis: np.ndarray) -> bool:
    """Justify if a subspace is a sub Lie algebra

    Args:
        subspace_basis (np.ndarray): basis of subspace, each line is a basis

    Returns:
        bool: if subspace is a sub Lie algebra
    """
    basis_num = subspace_basis.shape[0]
    for i in range(basis_num):
        for j in range(i, basis_num):
            if not L.is_in_subspace(subspace_basis, bracket(L, subspace_basis[i], subspace_basis[j])):
                return False
    return True

def sub_lie_algebra(L: LieAlgebra, subspace_basis: np.ndarray) -> LieAlgebra:
    """Get the sub Lie algebra of a subspace

    Args:
        subspace_basis (np.ndarray): basis of subspace, each line is a basis

    Returns:
        LieAlgebra: sub Lie algebra
    """
    basis_num = L.dimension
    structure_constant = np.zeros((basis_num, basis_num, basis_num))
    for i in range(basis_num):
        for j in range(i, basis_num):
            structure_constant[i, j, :] = bracket(L, subspace_basis[i], subspace_basis[j])
    return LieAlgebra(structure_constant, subspace_basis)
    
def Killing_form(L: LieAlgebra, x: np.ndarray, y: np.ndarray):
    pass
