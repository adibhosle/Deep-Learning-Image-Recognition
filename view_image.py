from keras.datasets import cifar10
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)

class_names = {
    0: "Plane",
    1: "Car",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Boat",
    9: "Truck"
}

for i in range(1000):
    # print(x_train[i], class_names[y_train[i][0]])
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i][0]])
    plt.show()