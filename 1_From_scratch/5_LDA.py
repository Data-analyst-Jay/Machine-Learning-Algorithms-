import numpy as np

X = np.array([[2, 3], [3, 4], [4, 5], [5, 6], [5, 8], [6, 7], [7, 8], [8, 9]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

def lda(X,y):
    labels = np.unique(y)
    mean_vectors = []
    for i in labels:
        mean_vectors.append(np.mean(X[y==i],axis=0))

    S_W = np.zeros((X.shape[1],X.shape[1]))

    for i, j in zip(labels, mean_vectors):
        class_sc_mat = np.zeros((X.shape[1],X.shape[1]))
        for row in X[y == i]:
            row, mean_vec = row.reshape(X.shape[1],1), j.reshape(X.shape[1],1)
            class_sc_mat += (row-mean_vec).dot((row-mean_vec).T)
        S_W += class_sc_mat

    overall_mean = np.mean(X, axis=0)
    S_B = np.zeros((X.shape[1],X.shape[1]))
    for i, mean_vec in enumerate(mean_vectors):
        n = X[y==i].shape[0]
        mean_vec = mean_vec.reshape(X.shape[1],1)
        overall_mean = overall_mean.reshape(X.shape[1],1)
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

    W = eig_pairs[0][1].reshape(X.shape[1],1)

    X_reduced = X.dot(W)
    return X_reduced

ans = lda(X,y)
print(ans)