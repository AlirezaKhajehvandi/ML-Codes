import numpy as np
# Imports
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from time import sleep

def unit_step_func(x):
    return np.where(x > 0 , 1, 0)

class Perceptron:

    def __init__(self, learning_rate=0.01, n_iters=10):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None

    def plot_decision_boundary(self, weights, bias, xlim=np.array([-1, 5]), ylim=np.array([-20, 20]), figure=plt.figure):
        x = xlim
        y = ((x*weights[0]) + bias) / (weights[1] * (-1))
        plt.ylim(np.min(ylim), np.max(ylim))
        ax.plot(x, y, c="green")

    def points(self, X, y, figure=plt.figure):
        for i in range(len(y)):
            if y[i] == 0:
                ax.scatter(X[i, 0], X[i, 1], c="r", marker=".")
            else:
                ax.scatter(X[i, 0], X[i, 1], c="b", marker=".") 
    def points_selected(self, X, figure=plt.figure):
        for i in range(len(y)):
            ax.scatter(X[0], X[1], c="black", marker="*")
            



    def fit(self, X, y, figure=plt.figure):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.total_error = []
        num_iteration = []
        y_ = np.where(y > 0 , 1, 0)
        x_lim_min = np.min(X[:, 0]) - 1
        x_lim_max = np.max(X[:, 0]) + 1
        y_lim_min = np.min(X[:, 1]) - 1
        y_lim_max = np.max(X[:, 1]) + 1
        plt.xlim(x_lim_min, x_lim_max)
        plt.ylim(y_lim_min, y_lim_max)
        p.plot_decision_boundary(weights=p.weights, bias=p.bias, 
                                 xlim=np.array([x_lim_min, x_lim_max]), 
                                 ylim=np.array([y_lim_min, y_lim_max]),
                                 figure=fig)
        p.points(X=X, y=y, figure=fig)
        # learn weights
        for num_iter in range(self.n_iters):
            error = 0
            
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                self.points_selected(X=x_i,  figure=fig)
                # Perceptron update rule
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
                error +=  np.sum(np.abs(y_[idx] - y_predicted))
                plt.pause(0.1)
            print (num_iter)
            num_iteration.append(num_iter+1)
            self.total_error.append(error)
            # ax.plot(num_iteration, self.total_error)
            plt.cla()
            self.plot_decision_boundary(weights=self.weights, bias=self.bias, 
                                        xlim=np.array([x_lim_min, x_lim_max]), 
                                        ylim=np.array([y_lim_min, y_lim_max]),
                                        figure=fig)
            self.points(X=X, y=y, figure=fig)
            plt.pause(0.1)
            print ("before if")
            print (self.total_error)
            if (self.total_error[num_iter] == 0):
                print ("after if")
                print (self.total_error)
                break



    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted
    

      


# Testing
if __name__ == "__main__":
    

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    X, y = datasets.make_blobs(
        n_samples=40, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    # x_lim_min = np.min(X[:, 0]) - 1
    # x_lim_max = np.max(X[:, 0]) + 1
    # y_lim_min = np.min(X[:, 1]) - 1
    # y_lim_max = np.max(X[:, 1]) + 1
    

    fig, ax = plt.subplots(figsize=(5, 5))
    # plt.xlim(x_lim_min, x_lim_max)
    # plt.ylim(y_lim_min, y_lim_max)
    p = Perceptron(learning_rate=0.01, n_iters=10)
    p.fit(X, y, figure=fig)
    predictions = p.predict(X_test)
    # plt.xlim((0, 20))
    print("Perceptron classification accuracy", accuracy(y_test, predictions))

    

    # p.plot_decision_boundary(weights=p.weights, bias=p.bias, xlim=np.array([-1, 5]), figure=fig)
    # p.points(X=X, y=y, figure=fig)


    plt.show()