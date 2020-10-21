from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def svc(x_train, x_test, y_train, y_test):
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'), verbose=True)
    print("Go")
    clf.fit(x_train, y_train)
    print("Guessing:")
    print(clf.score(x_test, y_test))


def k_neighbours(x_train, x_test, y_train, y_test):
    neigh = KNeighborsClassifier(n_neighbors=5)
    print("go")
    neigh.fit(x_train, y_train)
    print("Guessing:")
    print(neigh.score(x_test, y_test))


def random_forest_classifier(x_train, x_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=10)
    print("go")
    clf.fit(x_train, y_train)
    print("Guessing:")
    print(clf.score(x_test, y_test))