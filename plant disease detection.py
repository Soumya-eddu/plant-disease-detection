pip install keras.utils 
# Import Libraries 
import warnings 
warnings.filterwarnings("ignore") 
import os 
import glob 
import matplotlib.pyplot as plt 
# Keras API 
import tensorflow as tf 
from tensorflow import keras 
import keras 
from keras.models import Sequential 
from keras.layers import Dense,Dropout,Flatten 
from keras.layers import Conv2D,MaxPooling2D,Activation,AveragePooloing2D 
from  keras.layers import BatchNormalization 
from keras.preprocessing.image import ImageDataGenerator 
from keras.utils import np_utils 
# My dataset path 
test_dir = "D:\AD\Datasett\Test" 
train_dir = "D:\AD\Datasett\Train" 
# function to get count of images 
def get_files(directory): 
  if not os.path.exists(directory): 
    return 0 
  count=0 
  for current_path,dirs,files in os.walk(directory): 
    for dr in dirs: 
      count+= len(glob.glob(os.path.join(current_path,dr+"/*"))) 
  return count 
#To get the no of images 
train_samples =get_files(train_dir) 
num_classes=len(glob.glob(train_dir+"/*")) 
test_samples=get_files(test_dir) 
print(num_classes,"Classes") 
print(train_samples,"Train images") 
print(test_samples,"Test images") 
# Preprocessing data, Data Augmentation 
train_datagen= ImageDataGenerator(rescale=1/255, 
                                   shear_range=0.2, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True) 
test_datagen=ImageDataGenerator(rescale=1/255) 
img_width,img_height =256,256 
input_shape=(img_width,img_height,3) 
batch_size =32 
train_generator =train_datagen.flow_from_directory(train_dir, 
                                                   target_size=(img_width,img_height), 
                                                   batch_size=batch_size) 
test_generator=test_datagen.flow_from_directory(test_dir,shuffle=True, 
                                                   target_size=(img_width,img_height), 
                                                   batch_size=batch_size) 
# The name of the 3 diseases. 
train_generator.class_indices 
# CNN building. 
model = Sequential() 
model.add(Conv2D(32, (5, 5),input_shape=input_shape,activation='relu')) 
model.add(MaxPooling2D(pool_size=(3, 3))) 
model.add(Conv2D(32, (3, 3),activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Conv2D(64, (3, 3),activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Flatten()) 
model.add(Dense(512,activation='relu')) 
model.add(Dropout(0.25)) 
model.add(Dense(128,activation='relu')) 
model.add(Dense(num_classes,activation='softmax')) 
model.summary() 
# Check the names of the layers in the model 
model_layers = [ layer.name for layer in model.layers] 
print('layer name : ',model_layers) 
# Take one image to visualize it's changes after every layer 
from keras.preprocessing import image 
import numpy as np 
img1 = image.load_img('D:/AD/Datasett/Train/Healthy/803bcb65f486e39a.jpg') 
plt.imshow(img1); 
#preprocess image 
img1 = image.load_img('D:/AD/Datasett/Train/Healthy/803bcb65f486e39a.jpg', target_size=(256, 
256)) 
img = image.img_to_array(img1) 
img = img/255 
img = np.expand_dims(img, axis=0) 
# Visualizing output after every layer. 
from keras.models import Model 
conv2d_1_output = Model(inputs=model.input, outputs=model.get_layer('conv2d_1').output) 
max_pooling2d_1_output = 
Model(inputs=model.input,outputs=model.get_layer('max_pooling2d_1').output) 
conv2d_2_output = Model(inputs=model.input,outputs=model.get_layer('conv2d_1').output) 
max_pooling2d_2_output = 
Model(inputs=model.input,outputs=model.get_layer('max_pooling2d_1').output) 
max_pooling2d_1_features = max_pooling2d_1_output.predict(img) 
conv2d_2_features = conv2d_2_output.predict(img) 
max_pooling2d_2_features = max_pooling2d_2_output.predict(img) 
import matplotlib.image as mpimg 
fig=plt.figure(figsize=(14,7) 
columns = 8 
rows = 4 
for i in range(columns*rows): 
    #img = mpimg.imread() 
    fig.add_subplot(rows, columns, i+1) 
plt.axis('off') 
plt.title('filter'+str(i)) 
plt.imshow(conv2d_1_features[0, :, :, i], cmap='viridis') # Visualizing in color mode. 
plt.show() 
import matplotlib.image as mpimg 
fig=plt.figure(figsize=(14,7)) 
columns = 8 
rows = 4 
for i in range(columns*rows): 
#img = mpimg.imread() 
fig.add_subplot(rows, columns, i+1) 
plt.axis('off') 
plt.title('filter'+str(i)) 
import matplotlib.image as mpimg 
fig=plt.figure(figsize=(14,7)) 
plt.imshow(conv2d_1_features[0, :, :, i], cmap='viridis') # Visualizing in color mode. 
plt.show() 
import matplotlib.image as mpimg 
fig=plt.figure(figsize=(14,7)) 
columns = 8 
rows = 4 
for i in range(columns*rows): 
#img = mpimg.imread() 
fig.add_subplot(rows, columns, i+1) 
plt.axis('off') 
plt.title('filter'+str(i)) 
plt.imshow(conv2d_2_features[0, :, :, i], cmap='viridis') 
plt.show() 
# we can also visualize in color mode. 
columns = 8 
rows = 4 
for i in range(columns*rows): 
    #img = mpimg.imread() 
    fig.add_subplot(rows, columns, i+1) 
    plt.axis('off') 
    plt.title('filter'+str(i)) 
    plt.imshow(max_pooling2d_2_features[0, :, :, i], cmap='viridis') 
plt.show() 
# validation data. 
validation_generator = train_datagen.flow_from_directory( 
    train_dir, # same directory as training data 
    target_size=(img_height, img_width), 
    batch_size=batch_size) 
# Model building to get trained with parameters. 
opt=tf.keras.optimizers.Adam(lr=0.001) 
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy']) 
train=model.fit_generator(train_generator, 
                          epochs=15, 
                          steps_per_epoch=train_generator.samples // batch_size, 
                          validation_data=validation_generator, 
                        verbose=1) 
from keras.callbacks import History 
#history = History() 
acc = train.history['accuracy'] 
val_acc = train.history['val_accuracy'] 
loss = train.history['loss'] 
val_loss = train.history['val_loss'] 
epochs = range(1, len(acc) + 1) 
#Train and validation accuracy 
plt.plot(epochs, acc, 'b', label='Training accurarcy') 
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy') 
plt.title('Training and Validation accurarcy') 
plt.legend() 
plt.figure() 
#Train and validation loss 
plt.plot(epochs, loss, 'b', label='Training loss') 
plt.plot(epochs, val_loss, 'r', label='Validation loss') 
plt.title('Training and Validation loss') 
plt.legend() 
plt.show() 
score,accuracy =model.evaluate(test_generator,verbose=1) 
print("Test score is {}".format(score)) 
print("Test accuracy is {}".format(accuracy)) 
# Save entire model with optimizer, architecture, weights and training configuration. 
from keras.models import load_model 
model.save('plant.h5') 
import numpy as np 
from keras.preprocessing import image 
from keras.models import load_model 
import matplotlib.pyplot as plt 
# Load the saved model 
model = load_model('plant.h5') 
# Function to preprocess the image 
def preprocess_image(img_path, target_size): 
    img = image.load_img(img_path, target_size=target_size) 
    img_array = image.img_to_array(img) 
    img_array = img_array / 255.0  # Normalize the image 
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match the model's 
input shape 
    return img_array 
# Path to the new image 
img_path = 'D:\AD\Datasett\Test\Rust\93b7273db49054ca.jpg' 
# Preprocess the image 
img_array = preprocess_image(img_path, target_size=(256, 256)) 
# Make a prediction 
predictions = model.predict(img_array) 
predicted_class = np.argmax(predictions, axis=1) 
# Mapping of class indices to class names 
class_indices = { 
    0: 'Healthy', 
    1: 'Powdery', 
    2: 'Rust', 
    } 
# Get the class name of the predicted class 
predicted_class_name = class_indices[predicted_class[0]] 
# Print the prediction 
print(f'The model predicts that the image is: {predicted_class_name}') 
# Optionally, display the image with the prediction 
img = image.load_img(img_path) 
plt.imshow(img) 
plt.title(f'Prediction: {predicted_class_name}') 
plt.axis ('off') 
plt.show() 
if(predicted_class_name == 'Powdery'): 
    print('These fertlizers can be used Powdery Leaves: sulfur,chlorothalonil,horticultural 
oil,potassium bicarbonate,Bacillus subtilis') 
elif(predicted_class_name == 'Rust'): 
    print('These fertlizers can be used for Rust Leaves : Foliar sprays with fungicides containing 
propiconazole or triazole  ') 
else: 
print('No Fertlizers are required : ')
