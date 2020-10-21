import numpy as np
from sklearn.model_selection import train_test_split
from models import svc, k_neighbours, random_forest_classifier

def main():
    print("hei")
    x_data = np.genfromtxt('handwritten_digits_images.csv', dtype='int', delimiter=',')
    print("hei2")
    x_data2 = x_data.reshape(x_data.shape[0], 28, 28)

    y_data = np.genfromtxt('handwritten_digits_labels.csv', dtype='int', delimiter='\n')

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=42)

    #svc(x_train, x_test, y_train, y_test)
    #k_neighbours(x_train, x_test, y_train, y_test)
    random_forest_classifier(x_train, x_test, y_train, y_test )



main()
