#from sklearn import cross_validation
from sklearn import model_selection

from PIL import Image
import os, glob
import numpy as np

# select the object category
caltech_dir = "/data/101_ObjectCategories"
categories = ["chair","cup","Faces_easy","lamp","laptop","scissors","soccer_ball","stapler","watch","yin_yang"]

nb_classes = len(categories)

# specify the image size
image_w = 32
image_h = 32
pixels = image_w * image_h * 3

# loading the images
X = []
Y = []
for idx, cat in enumerate(categories):
    # label preparation
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    # image data preparation
    image_dir = caltech_dir + "/" + cat
    files = glob.glob(image_dir+"/*.jpg")
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        X.append(data)
        Y.append(label)
        if i % 10 == 0:
            print(i, "\n", data)
X = np.array(X)
Y = np.array(Y)

# make training data and test data set
X_train, X_test, y_train, y_test = \
    model_selection.train_test_split(X, Y)

xy = (X_train, X_test, y_train, y_test)
np.save("/data/101_Caltech_npy/32-32img-10obj.npy", xy)

print("ok,", len(Y))
