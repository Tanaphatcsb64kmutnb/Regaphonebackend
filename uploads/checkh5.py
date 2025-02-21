import tensorflow as tf
from tensorflow import keras
import h5py
import pydot
import graphviz

# โหลดโมเดล
model = keras.models.load_model('myyogamodel.h5')

# วาดโครงสร้างโมเดล
keras.utils.plot_model(model, 
                      to_file='model_structure.png',
                      show_shapes=True,  # แสดง shape ของแต่ละ layer
                      show_layer_names=True,  # แสดงชื่อ layer
                      show_dtype=True,  # แสดง data type
                      rankdir='TB')  # TB = top to bottom, LR = left to right

# ถ้าอยากดูแบบ summary ใน terminal ก็ใช้
model.summary()