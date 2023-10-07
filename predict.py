#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from keras.models import load_model
from keras.preprocessing import image

class model:
    def __init__(self,filename):
        self.filename =filename


    def prediction(self):
        # load model
        model = load_model('densenet_model.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        labels  = {0: 'burger', 1: 'butter_naan', 2: 'chai', 3: 'chapati', 4: 'chole_bhature', 5: 'dal_makhani',
                        6: 'dhokla', 7: 'fried_rice', 8: 'idli', 9: 'jalebi', 10: 'kaathi_rolls',
                        11: 'kadai_paneer', 12: 'kulfi', 13: 'masala_dosa', 14: 'momos', 15: 'paani_puri',
                        16: 'pakode', 17: 'pav_bhaji', 18: 'pizza', 19: 'samosa'}

        predicted_label = labels[np.argmax(result[0])]  # lable[0]
        return [{ "Prediction" : predicted_label}]




