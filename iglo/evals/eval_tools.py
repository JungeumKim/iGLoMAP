from collections import Counter
from sklearn.neighbors import  NearestNeighbors

# code modified from pacmap

def knn_clf(nbr_vec, y):
    '''
    Helper function to generate knn classification result.
    '''
    y_vec = y[nbr_vec]
    c = Counter(y_vec)
    return c.most_common(1)[0][0]


def knn_clf_seq(K, pairwise_X, Y):
    nbrs = NearestNeighbors(n_neighbors=K + 1, metric="precomputed").fit(pairwise_X)
    distances, indices = nbrs.kneighbors(pairwise_X)

    def classifier(K, indices=indices, distances=distances):
        sum_acc = 0
        max_acc = pairwise_X.shape[0]
        indices = indices[:, 1:(K + 1)]
        distances = distances[:, 1:(K + 1)]
        for i in range(pairwise_X.shape[0]):
            result = knn_clf(indices[i], Y)
            if result == Y[i]:
                sum_acc += 1
        avg_acc = sum_acc / max_acc
        print(avg_acc)
        return avg_acc

    return classifier