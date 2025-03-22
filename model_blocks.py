from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Add, ReLU, Input
from tensorflow.keras.models import Model

def inception_res_block(x, filters):
    # 1x1 conv
    path1 = Conv2D(filters, (1, 1), padding='same', activation='relu')(x)

    # 3x3 conv
    path2 = Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
    path2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(path2)

    # 5x5 conv
    path3 = Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
    path3 = Conv2D(filters, (5, 5), padding='same', activation='relu')(path3)

    # Merge
    merged = concatenate([path1, path2, path3], axis=-1)
    bottleneck = Conv2D(filters, (1, 1), padding='same')(merged)

    shortcut = Conv2D(filters, (1, 1), padding='same')(x)
    out = Add()([bottleneck, shortcut])
    return ReLU()(out)
