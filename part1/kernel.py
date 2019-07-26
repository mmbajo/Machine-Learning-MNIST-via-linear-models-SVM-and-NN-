import numpy as np

### Functions for you to fill in ###

# pragma: coderesponse template


def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    kernel = np.matmul(X, Y.T) + c
    kernel_matrix = kernel ** p
    return kernel_matrix
    #raise NotImplementedError
# pragma: coderesponse end

# pragma: coderesponse template


def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    difference_norm = np.linalg.norm((X[:, None, :] - Y[:, :]), axis = 2) ** 2
    kernel_matrix = np.exp(-gamma * difference_norm)
    #print(X.shape, Y.shape, kernel_matrix.shape)
    # n = X.shape[0]
    # m = Y.shape[0]
    # kernel_matrix = np.zeros((n ,m))
    # for i in range(n):
    #     for j in range(m):
    #         kernel_matrix[i][j] = np.exp(-gamma * (np.linalg.norm((X[i] - Y[j]), ord = 2)**2))
    return kernel_matrix
    #raise NotImplementedError
# pragma: coderesponse end
