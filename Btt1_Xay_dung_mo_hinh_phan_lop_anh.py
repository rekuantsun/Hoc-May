
import numpy as np
import os
from skimage import io
from skimage.transform import resize
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

def check_corrupted_image(img_file):
    try:
        with Image.open(img_file) as img:
            img.verify()
            img_new = io.imread(os.path.join(img_file))
        return False
    except Exception as e:
        print(e)
        return True

def read_img_data(path,label, size):
    X = []
    y = []
    files = os.listdir(path)
    for img_file in files:
        if not(check_corrupted_image(os.path.join(path,img_file))):
            img = io.imread(os.path.join(path, img_file), as_gray=True)
            img = resize(img, size)
            img_vector = img.flatten()
            X.append(img_vector)
            y.append(label)
    X = np.array(X)
    return X,y

def main():
    X,y = read_img_data('A:/Study/HocMay/Đồ án/PetImages/Cat','cat', (32,32))
    X_dog, y_dog = read_img_data('A:/Study/HocMay/Đồ án/PetImages/Dog','dog', (32,32))
    X = np.concatenate(X , X_dog)
    y = np.concatenate(y , y_dog)
    X = np.array(X)
    y = LabelBinarizer().fit_transform(y)
    print("x shape", X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state =1)

if __name__ == '__main__':
    main()