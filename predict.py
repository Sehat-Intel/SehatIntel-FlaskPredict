# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import cv2
from skimage import io


alex = tf.keras.models.load_model('alexnet_model.hdf5') 

labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 
            'Edema', 'Effusion', 'Emphysema', 
            'Fibrosis', 'Hernia', 'Infiltration', 
            'Mass', 'No Finding', 'Nodule', 
            'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

def main(img_path):

    pred_label = []
    img = io.imread(img_path)
    img_resized = cv2.resize(img, (224, 224))
    img_preprocessed = preprocess_input(img_resized)
    img_reshaped = img_preprocessed.reshape((1, 224, 224, 3))
    prediction = alex.predict(img_reshaped)
    pred_class = prediction.argmax(axis=-1)
    pred_label.append(labels[pred_class[0]])
    print('PREDICTION LABEL:',pred_label)


    """
    print('IMAGE PATH: ', img_path)
    predict_datagen = ImageDataGenerator(rescale=1. / 255)
    predict = predict_datagen.flow_from_directory(
        'static/', 
        target_size=(224,224), 
        batch_size = 1,
        class_mode='categorical')
    pred = alex.predict_generator(predict)
    prediction = os.listdir(labels)[np.argmax(pred)]
    """
    return pred_label



if __name__ == "__main__":
    main(img_path)
