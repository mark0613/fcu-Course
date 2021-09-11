import numpy as np
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Captcha_OCR:
    def __init__(self, char_list, captcha_len):
        self.char_list = char_list
        self.captcha_length = captcha_len
        self.batch_size = 1
        self.char_to_num = layers.experimental.preprocessing.StringLookup(
            vocabulary=self.char_list, mask_token=None
        )
        self.num_to_char = layers.experimental.preprocessing.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )
    def process_image(self, img_folder_path, img_width, img_height):
        img = os.listdir(img_folder_path)
        self.images = [f"{img_folder_path}/{img[0]}"]
        self.labels = ["0000"]
        self.img_width = img_width
        self.img_height = img_height
    def predict(self, validation_dataset):
        for batch in validation_dataset.take(1):
            batch_images = batch["image"]
            preds = self.prediction_model.predict(batch_images)
            pred_texts = self.decode_batch_predictions(preds)
            return pred_texts[0]
    def load_model(self, model_path):
        model =  keras.models.load_model(model_path)
        self.prediction_model = keras.models.Model(
            model.get_layer(name="image").input, model.get_layer(name="dense2").output
        )
    def recognize(self):
        x_valid, y_valid = self.split_data(np.array(self.images), np.array(self.labels))
        validation_dataset = self.get_dataset(x_valid, y_valid)
        return self.predict(validation_dataset)

    def split_data(self, images, labels):
        indices = np.arange(len(images))
        x_valid, y_valid = images[indices], labels[indices]
        return x_valid, y_valid
    def encode_single_sample(self, img_path, label):
        img = tf.io.read_file(img_path)
        img = tf.io.decode_jpeg(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.transpose(img, perm=[1, 0, 2])
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        return {"image": img, "label": label}
    def get_dataset(self, x, y):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = (
            dataset.map(
                self.encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            .batch(self.batch_size)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )
        return dataset
    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
            :, :self.captcha_length
        ]
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(self.num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text
