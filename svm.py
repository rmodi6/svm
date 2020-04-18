import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

# Hyperparameters
lambd = 0.1
T = 500000  # max_epochs


def train(X, y):
    max_fval = np.max(X)  # Maximum feature value
    eta = 0.001 * max_fval  # Initial learning rate
    w = np.zeros(X.shape[1])  # Weight vector

    # For each epoch
    for t in range(1, T + 1):
        # Get a random index from features
        i = np.random.choice(X_train.shape[0])
        # Get the corresponding random feature and label
        X_i, y_i = X[i], y[i]

        # Compute hinge loss
        v = 1 - (y_i * np.dot(X_i, w))

        # Compute the partial gradient
        if v <= 0:
            dw = lambd * w
        else:
            dw = lambd * w - y_i * X_i

        # Update the weight vector
        w = w - eta * dw

        # Reduce the learning rate but don't reduce it too much
        eta = eta / 10 if eta > 0.00001 else eta

        if t % (T / 10) == 0:
            print('#Epoch: {}/{}'.format(t, T))

    # Return the weight vector of the final epoch
    return w


def decision_function(w, X):
    # Add one to the X values to incorporate bias
    if len(w) == len(X[0]) + 1:
        X = np.c_[np.ones((X.shape[0])), X]
    # Compute the output of the svm weight vector
    return np.dot(X, w)


def draw(w, X, y):
    # Plot the features
    plt.scatter(X[:, 1], X[:, 2], c=y, s=30, cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = decision_function(w, xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    plt.show()


def test(w, X, y):
    # Compute the number of misclassified points
    error = np.sum(np.where(y != np.sign(decision_function(w, X)), 1, 0))
    print('Total number of test data points: {}'.format(len(X)))
    print('Number of misclassified points: {}'.format(error))


if __name__ == '__main__':
    # Generate sample data
    features, labels = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.05, random_state=10)

    # Add one to the feature values to incorporate bias
    features = np.c_[np.ones((features.shape[0])), features]
    # Change labels from (0,1) -> (-1,1)
    labels = np.where(labels == 1, 1, -1)

    # Train Test Split
    X_train, X_test, y_train, y_test = features[:80], features[-20:], labels[:80], labels[-20:]

    weights = train(X_train, y_train)
    draw(weights, X_train, y_train)
    test(weights, X_test, y_test)
