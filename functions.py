import numpy as np
from common.functions import sigmoid

'''
통계적 분석기법 함수
-로지스틱 회귀: fit(), predict()
-LDA: fit(), predict()
-나이브 베이즈 분류기: fit(), predict()
'''


#로지스틱 회귀 분석 함수
class LogisticRegression:
    def __init__(self, lr=0.001, num_iter=100):
        self.learning_rate = lr
        self.num_iter = num_iter
        self.weights = None
        self.bias = None
    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.random.rand(n) 
        self.bias = np.random.rand(1) 
        for i in range(self.num_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear_model)
            self.weights += self.learning_rate * ((1 / m) * np.dot(X.T, (y_pred - y))) *-1
            self.bias += self.learning_rate * ((1 / m) * np.sum(y_pred - y))*-1
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_pred]

#LDA 함수
class LinearDiscriminantAnalysis:
    def __init__(self):
        self.means,self.priors,self.covar = None,None,None

    def fit(self, X, y):
        self.means = {}
        self.priors = {}
        classes = np.unique(y)
        n = X.shape[1]
        self.covar = np.zeros((n, n))

        for cls in classes:
            X_c = X[y == cls]
            self.means[cls] = np.mean(X_c, axis=0)
            self.priors[cls] = X_c.shape[0] / X.shape[0]
            self.covar += np.cov(X_c, rowvar=False) * (X_c.shape[0] - 1)
        self.covar /= (X.shape[0] - len(classes))

    def predict(self, X):
        inv_cov = np.linalg.inv(self.covar)
        scores = []
        for x in X:
            class_scores = {}
            for cls in self.means:
                vec = self.means[cls]
                score = x @ inv_cov @ vec - 0.5 * vec.T @ inv_cov @ vec + np.log(self.priors[cls])
                class_scores[cls] = score
            scores.append(max(class_scores, key=class_scores.get))
        return np.array(scores)
        

#나이브 베이즈 분류기
class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.means = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = {}
        self.var = {}
        self.priors = {}

        for cls in self.classes:
            X_c = X[y == cls]
            self.means[cls] = np.mean(X_c, axis=0)
            self.var[cls] = np.var(X_c, axis=0)
            self.priors[cls] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        result = []

        for x in X:
            temp = []
            for cls in self.classes:
                prior = np.log(self.priors[cls])
                cond = np.sum(np.log(2. * np.pi * self.var[cls])) - 0.5 * np.sum(((x - self.means[cls]) ** 2) / (self.var[cls])) * -0.5
                temp.append(prior + cond)
            result.append(self.classes[np.argmax(temp)])

        return np.array(result)
