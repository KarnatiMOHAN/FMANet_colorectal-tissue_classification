import os,cv2
import numpy as np
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
# from keras.utils import np_utils
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras import initializers
from keras.models import Model
from keras.regularizers import l2
import h5py
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam, SGD
from keras.layers import Input, Conv2D, LSTM, Add, Multiply, Concatenate, MaxPooling2D, AveragePooling2D, Flatten, Lambda
from keras.layers import SeparableConv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization


data_path = '.../Colorectal_tissue_dataset/Kather_texture_2016_image_tiles_5000'
data_dir_list = ['01_TUMOR', '02_STROMA', '03_COMPLEX', '04_LYMPHO', '05_DEBRIS', '06_MUCOSA', '07_ADIPOSE', '08_EMPTY']


folder_mapping = {
    '01_TUMOR': '0',
    '02_STROMA': '1',
    '03_COMPLEX': '2',
    '04_LYMPHO': '3',
    '05_DEBRIS': '4',
    '06_MUCOSA': '5',
    '07_ADIPOSE': '6',
    '08_EMPTY': '7'
}

img_data_list = []
labels = []

for dataset in data_dir_list:
    img_list = os.listdir(os.path.join(data_path, dataset))
    print('Loaded the images of dataset-{}\n'.format(folder_mapping[dataset]))
    for img in img_list:
        input_img = cv2.imread(os.path.join(data_path, dataset, img))
        labels.append(folder_mapping[dataset])
        input_img_resize = cv2.resize(input_img, (128, 128))
        img_data_list.append(input_img_resize)

label = np.array(labels)
data = np.array(img_data_list)

data = data.astype('float32')
data = data / 255.0

print("Data shape:", data.shape)



"""# **Network**"""
def Cosin_similarity(input):
    dot1 = K.batch_dot(input[0], input[1], axes=[1, 1])
    dot2 = K.batch_dot(input[0], input[0], axes=[1, 1])
    dot3 = K.batch_dot(input[1], input[1], axes=[1, 1])
    max = K.maximum(K.sqrt(dot2 * dot3), K.epsilon())
    value = dot1 / max
    return K.tanh(value)

def Bund(input):
    alpha_1 = input[0]
    alpha_2 = input[1]

    alpha_l = alpha_1/(alpha_1+alpha_2)
    alpha_g = alpha_2/(alpha_1+alpha_2)

    return alpha_l, alpha_g

#spatial attention
def residual_csa_module(input_tensor, reduction_ratio=16):
    channels = input_tensor.shape[-1]

    # Convolutional layer for feature extraction
    conv1 = Conv2D(filters=channels // reduction_ratio, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
    conv2 = Conv2D(filters=channels // reduction_ratio, kernel_size=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=channels, kernel_size=(1, 1), padding='same', activation='relu')(conv2)

    # Spatial attention mechanism
    attention_map = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='sigmoid')(conv3)

    # Multiply attention map with input tensor
    attention = Multiply()([attention_map, input_tensor])

    # Residual connection
    output = Add()([attention, input_tensor])

    return output

#channel attention
def eca_module(input_tensor, gamma=2, b=1):
    channels = input_tensor.shape[-1]
    print(channels)

    squeeze = GlobalAveragePooling2D()(input_tensor)
    excitation = Dense(units=channels // gamma)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=channels)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1, 1, channels))(excitation)

    scale = Multiply()([input_tensor, excitation])

    output = Add()([scale, input_tensor])
    return output

def FMANet(input1, num_classes):
    eps = 1.1e-5
    input1 = Input((128, 128, 3))
    
    x1 = Conv2D(64, (3, 3), padding = 'same', activation='relu')(input1)
    x1 = BatchNormalization(axis=-1, epsilon=eps)(x1)
    x2 = Conv2D(64, (3, 3), padding = 'same', activation='relu')(x1)
    x2 = BatchNormalization(axis=-1, epsilon=eps)(x2)
    x3 = Conv2D(64, (3, 3), padding = 'same', activation='relu')(x2)
    x3 = BatchNormalization(axis=-1, epsilon=eps)(x3)
    c1 = Concatenate()([x1, x2, x3])
    
    x11 = Conv2D(192, (3,3), dilation_rate = (1, 1), padding = 'same', activation='relu')(c1)
    x11 = BatchNormalization(axis=-1, epsilon=eps)(x11)
    x21 = Conv2D(192, (3,3), dilation_rate = (2, 2), padding = 'same', activation='relu')(c1)
    x21 = BatchNormalization(axis=-1, epsilon=eps)(x21)
    x31 = Conv2D(192, (3,3), dilation_rate = (3, 3) , padding = 'same', activation='relu')(c1)
    x31 = BatchNormalization(axis=-1, epsilon=eps)(x31)
    M1 = Multiply()([x11, x21])
    M2 = Multiply()([x11, x31])
    M3 = Multiply()([x21, x31])
    x111 = Conv2D(192, (3,3),dilation_rate = (1, 1), padding = 'same', activation='relu')(M1)
    x111 = BatchNormalization(axis=-1, epsilon=eps)(x111)
    x211 = Conv2D(192, (3,3),dilation_rate = (2, 2), padding = 'same', activation='relu')(M2)
    x211 = BatchNormalization(axis=-1, epsilon=eps)(x211)
    x311 = Conv2D(192, (3,3),dilation_rate = (3, 3), padding = 'same', activation='relu')(M3)
    x311 = BatchNormalization(axis=-1, epsilon=eps)(x311)
    M4 = Multiply()([x111, x211, x311])
    X44 = Conv2D(192, (1,1), padding = 'same', activation='relu')(M4)
    X44 = BatchNormalization(axis=-1, epsilon=eps)(X44)
    A1 = Add()([X44, c1])
    
    #spatial attention
    res_csa_module1 = residual_csa_module(A1)
    
    x4 = Conv2D(64, (3, 3), padding = 'same', activation='relu')(x3)
    x4 = BatchNormalization(axis=-1, epsilon=eps)(x4)
    x5 = Conv2D(64, (3, 3), padding = 'same', activation='relu')(x4)
    x5 = BatchNormalization(axis=-1, epsilon=eps)(x5)
    x6 = Conv2D(64, (3, 3), padding = 'same', activation='relu')(x5)
    x6 = BatchNormalization(axis=-1, epsilon=eps)(x6)
    c2 = Concatenate()([x4, x5, x6])
    
    x51 = Conv2D(192, (3,3), dilation_rate = (1, 1), padding = 'same', activation='relu')(c2)
    x51 = BatchNormalization(axis=-1, epsilon=eps)(x51)
    x61 = Conv2D(192, (3,3),dilation_rate = (2, 2), padding = 'same', activation='relu')(c2)
    x61 = BatchNormalization(axis=-1, epsilon=eps)(x61)
    x71 = Conv2D(192, (3,3),dilation_rate = (3, 3), padding = 'same', activation='relu')(c2)
    x71 = BatchNormalization(axis=-1, epsilon=eps)(x71)
    M6 = Multiply()([x51, x61])
    M7 = Multiply()([x51, x71])
    M8 = Multiply()([x61, x71])
    x511 = Conv2D(192, (3,3), dilation_rate = (1, 1), padding = 'same', activation='relu')(M6)
    x511 = BatchNormalization(axis=-1, epsilon=eps)(x511)
    x611 = Conv2D(192, (3,3),dilation_rate = (2, 2), padding = 'same', activation='relu')(M7)
    x611 = BatchNormalization(axis=-1, epsilon=eps)(x611)
    x711 = Conv2D(192, (3,3), dilation_rate = (3, 3),padding = 'same', activation='relu')(M8)
    x711 = BatchNormalization(axis=-1, epsilon=eps)(x711)
    M9 = Multiply()([x511, x611, x711])
    X81 = Conv2D(192, (1,1), padding = 'same', activation='relu')(M9)
    X81 = BatchNormalization(axis=-1, epsilon=eps)(X81)
    A2 = Add()([X81, c2])
    
    #channel attention
    eca_module1 = eca_module(A2)
    
    x_l = A1
    x_g = A2
    
    # x_lg = Multiply()([A1, A2])
    # x_gl = Add()([A1, A2])
    x_c = Concatenate()([x_l, x_g])
    
    x_l = GlobalAveragePooling2D()(x_l)
    x_g = GlobalAveragePooling2D()(x_g)
    x_c = GlobalAveragePooling2D()(x_c)
    
    x_l = Dense(units=384, activation='relu')(x_l)
    x_g = Dense(units=384, activation='relu')(x_g)
    
    share_1 = Dense(units=768, activation='relu')
    share_2 = Dense(units=768, activation='relu')
    
    x_l = share_1(x_l)
    x_g = share_1(x_g)
    x_c = share_1(x_c)
    
    x_l = share_2(x_l)
    x_g = share_2(x_g)
    x_c = share_2(x_c)

    x_l = Dense(units=8, activation=softmax)(x_l)
    x_g = Dense(units=8, activation=softmax)(x_g)
    x_c = Dense(units=8, activation=softmax)(x_c)
    
    alpha_1 = Lambda(Cosin_similarity, output_shape= (None, 1))([x_l, x_c])
    alpha_2 = Lambda(Cosin_similarity, output_shape= (None, 1))([x_g, x_c])
    alpha_l, alpha_g = Lambda(Bund)([alpha_1, alpha_2])
    out_l = Multiply()([alpha_l, x_l])
    out_g = Multiply()([alpha_g, x_g])
    
    out = Concatenate()([out_l, out_g])
    out = Dense(units = 64, activation = 'relu')(out)
    
    out = Dense(units=num_classes, activation="softmax")(out)
    model = Model(inputs=input1, outputs=out)
    return model



from sklearn.model_selection import StratifiedKFold
input = Input(shape=(128, 128, 3), name='input')
# Initialize your model
num_classes = 8
model = FMANet(input, num_classes)

# Initialize 10-fold cross-validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Lists to store the performance metrics from each fold
accuracy_scores = []

# Cross-validation loop
for train_index, test_index in kfold.split(x, y):
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    opt = Adam(learning_rate=0.0001)
    model.compile(optimizer= opt, loss='categorical_crossentropy', metrics=['accuracy'])
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    early_stopping = EarlyStopping(
                              patience=5,
                              min_delta=0.001,
                              monitor="val_loss",
                              restore_best_weights=True
                              )
    checkpoint = ModelCheckpoint('../Dataset_1'+'-{epoch:02d}.h5', monitor='val_accuracy', save_best_only=True)
    history = model.fit(X_train, y_train, batch_size= 16, epochs= 100,
                    verbose=1, callbacks=[early_stopping, checkpoint], validation_data=(X_test, y_test), shuffle = True) #, callbacks=[early_stopping, checkpoint]


    pred = model.predict(X_test)
    y_pred = np.argmax(pred, axis = 1)
    y_true = np.argmax(y_test, axis = 1)

    if len(y_pred) != len(y_true):
        raise ValueError("Lengths of predictions and ground_truth lists must be the same.")
    
    correct_count = 0
    total_count = len(y_pred)
    
    for pred, truth in zip(y_pred, y_true):
        if pred == truth:
            correct_count += 1

    accuracy = correct_count / total_count * 100

    print(accuracy)

    accuracy_scores.append(accuracy)

# Calculate and display the average performance across all folds
mean_accuracy = sum(accuracy_scores) / len(accuracy_scores)

print(f"Mean Accuracy: {mean_accuracy:.2f}")
