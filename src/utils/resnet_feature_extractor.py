from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
import numpy as np
import os


model = ResNet50(weights='imagenet', include_top=False)
model.summary()

dir_path = '/home/hima/hima/data/Insight-MVT_Annotation_Train/'
for dir_ in os.listdir(dir_path):
    tmp_dir = dir_path + dir_ + "/"
    print(tmp_dir)
    for file in os.listdir(tmp_dir):
        img = image.load_img(tmp_dir + file, target_size=(224, 224, 3))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        resnet_feature = model.predict(img_data)
        print(resnet_feature.shape)
        np.save(open('/home/hima/hima/data/resnet_features_insight/'+tmp_dir+"_"+file))
