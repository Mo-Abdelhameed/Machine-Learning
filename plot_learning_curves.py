from sklearn.model_selection import ShuffleSplit, learning_curve
import matplotlib.pyplot as plt
import numpy as np
def plot_learning_curve(model, X, Y):
    cv = ShuffleSplit(n_splits=50, test_size=0.2)

    train_sizes, train_scores, test_scores = learning_curve(model, X, Y, cv=cv)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r",
    )

    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")

    plt.fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g",
    )
    plt.plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")
    plt.grid()
    plt.legend()