import numpy as np


def mean_upper_triangular(arr) -> float:
    # Create a mask for the upper triangular elements
    mask = np.triu(np.ones(arr.shape), k=1).astype(bool)
    # Use the mask to get the upper triangular elements
    upper_triangular = arr[mask]
    # Calculate the mean of the upper triangular elements
    mean = np.mean(upper_triangular)
    return float(mean)
