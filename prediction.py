from keras.models import model_from_json
from pathlib import Path

from keras.preprocessing import image
from pandas import np

classes = [
    "Plane",
    "Car",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Boat",
    "Truck"
]

f = Path("model_structure.json")
model_structure = f.read_text()
model = model_from_json(model_structure)
model.load_weights("model_weights.h5")

img = image.load_img("tractor-3-1386656.jpg", target_size=(32, 32))
img_arry = image.img_to_array(img) / 255 

list_img = np.expand_dims(img_arry, axis=0)

res = model.predict(list_img)

single = res[0]
most_likely = int(np.argmax(single))
class_like = single[most_likely]

class_name = classes[most_likely]

print("The image is {} with accuracy: {:2f}".format(class_name, class_like))