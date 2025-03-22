from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense

def build_msrcnn(input_shape=(224, 224, 3), num_classes=3):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    x = inception_res_block(x, 64)
    x = MaxPooling2D((2,2))(x)

    x = inception_res_block(x, 128)
    x = MaxPooling2D((2,2))(x)

    x = inception_res_block(x, 256)
    x = MaxPooling2D((2,2))(x)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model
