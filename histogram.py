from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import laplacian_kernel, rbf_kernel

def histogram_intersection(histogram_1, histogram_2, histogram_kernel, histogram_size):
    return sum([min(histogram_1[k], histogram_2[k]) for k in range(histogram_size)])


def histogram_intersection_pairwise(X, Y):
    return pairwise_distances(X, Y, metric=histogram_intersection)


def combined_kernel(X, Y, histogram_size):
    histograms_X = X[:, :histogram_size]
    histograms_Y = Y[:, :histogram_size]
    classic_X = X[:, histogram_size:]
    classic_Y = Y[:, histogram_size:]
    return histogram_intersection_pairwise(histograms_X, histograms_Y) + laplacian_kernel(classic_X, classic_Y)

