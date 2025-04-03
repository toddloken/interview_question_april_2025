import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


class KNNModel:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_n_neighbors = 5
        self.model = None

    def find_best_k(self, Ks=10):
        """Find the best k value based on accuracy."""
        mean_acc = []

        for n in range(2, Ks):
            neigh = KNeighborsClassifier(n_neighbors=n).fit(self.X_train, self.y_train)
            yhat = neigh.predict(self.X_test)
            mean_acc.append(metrics.accuracy_score(self.y_test, yhat))

        print('Neighbor Accuracy List')
        print(mean_acc)

        self.best_n_neighbors = mean_acc.index(max(mean_acc)) + 2

    def train_model(self):
        """Train the KNN model with the best k value."""
        self.find_best_k()
        self.model = KNeighborsClassifier(n_neighbors=self.best_n_neighbors)
        self.model.fit(self.X_train, self.y_train)
        predKNN = self.model.predict(self.X_test)

        self.save_model()
        return predKNN

    def save_model(self, filename='model.pkl'):
        """Save the trained KNN model."""
        joblib.dump(self.model, filename)
        print(f'Model saved as {filename}')