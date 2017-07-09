import numpy as np

def hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
        - x : ndarray
    Returns:
        an array of shape (x.dim, x.ndim) + x.shape
        where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x) 
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian

def intersect(A, B):
    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)], \
            'formats':ncols * [A.dtype]}
    C1 = A.copy()
    C2 = B.copy()
    C = np.intersect1d(C1.view(dtype), C2.view(dtype))
    return C.view(A.dtype).reshape(-1, ncols)


def diff(A, B):
    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)], \
            'formats':ncols * [A.dtype]}
    C1 = A.copy()
    C2 = B.copy()
    C = np.setdiff1d(C1.view(dtype), C2.view(dtype))
    return C.view(A.dtype).reshape(-1, ncols)

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)\
            .swapaxes(1,2)\
            .reshape(-1, nrows, ncols))
