# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import cv2
from skimage import io
import numpy as np


model1 = tf.keras.models.load_model('model_resnet.hdf5')
model2 = tf.keras.models.load_model('vgg16-model-19-0.71.hdf5')

labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 
            'Edema', 'Effusion', 'Emphysema', 
            'Fibrosis', 'Infiltration', 
            'Mass', 'No Finding', 'Nodule', 
            'Pleural_Thickening', 'Pneumothorax',
            'Covid','Hernia','Normal','Pneumonia']

def main(img_path):

    pred_label = []
    img = io.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img/255
    img = np.reshape(img,(1, 224, 224, 3))
    img = np.array(img, dtype=np.float64)

    prediction1 = model1.predict(img)
    prediction2 = model2.predict(img)
    arr = np.concatenate((prediction1[0],prediction2[0]))
    pred_class = arr.argmax(axis=-1)
    pred_label.append(labels[pred_class])
    return pred_label

if __name__ == "__main__":
    main(img_path)
