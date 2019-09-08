from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense,\
                         GlobalAveragePooling2D, Input

from common.config import Config    

# Model based on https://github.com/tomasz-oponowicz/spoken_language_identification

def build_model(conf: Config, space, input_shape) -> Model:
    input_data = Input(shape=input_shape)
    x = Conv2D(
        filters=space['filters_conv2d_1'],
        kernel_size=3,
        activation='relu',
        data_format='channels_last',
        padding=('same' if space['maxpooling_1'] is not None else
                 'valid')
    )(input_data)
    if space['maxpooling_1'] is not None:
        x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(space['dropout1'])(x)

    if space['filters_conv2d_2'] is not None:
        x = Conv2D(
        filters=space['filters_conv2d_2'],
                kernel_size=3,
                activation='relu',
                padding=('same' if space['maxpooling_2'] is not None else
                         'valid')
        )(x)
        if space['maxpooling_2'] is not None:
            x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(space['dropout2'])(x)

    if space['filters_conv2d_3'] is not None:
        x = Conv2D(
        filters=space['filters_conv2d_3'],
                kernel_size=3,
                activation='relu',
                padding=('same' if space['maxpooling_3'] is not None else
                         'valid')
        )(x)
        if space['maxpooling_3'] is not None:
            x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.2)(x)
    
    if space['filters_conv2d_4'] is not None:
        x = Conv2D(
        filters=space['filters_conv2d_4'],
                kernel_size=3,
                activation='relu',
                padding=('same' if space['maxpooling_4'] is not None else
                         'valid')
        )(x)
        if space['maxpooling_4'] is not None:
            x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.2)(x)
    
    if space['filters_conv2d_5'] is not None:
        x = Conv2D(
        filters=space['filters_conv2d_5'],
                kernel_size=3,
                activation='relu',
                padding=('same' if space['maxpooling_5'] is not None else
                         'valid')
        )(x)
        if space['maxpooling_5'] is not None:
            x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.2)(x)
    
    if space['filters_conv2d_6'] is not None:
        x = Conv2D(
        filters=space['filters_conv2d_6'],
                kernel_size=3,
                activation='relu',
                padding=('same' if space['maxpooling_6'] is not None else
                         'valid')
        )(x)
        if space['maxpooling_6'] is not None:
            x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.2)(x)
    
    if space['filters_conv2d_7'] is not None:
        x = Conv2D(
        filters=space['filters_conv2d_7'],
                kernel_size=3,
                activation='relu',
                padding=('same' if space['maxpooling_7'] is not None else
                        'valid')
        )(x)
        if space['maxpooling_7'] is not None:
            x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.2)(x)
    
    if space['filters_conv2d_8'] is not None:
        x = Conv2D(
        filters=space['filters_conv2d_8'],
                kernel_size=3,
                activation='relu',
                padding=('same' if space['maxpooling_8'] is not None else
                        'valid')
        )(x)
        if space['maxpooling_8'] is not None:
            x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.2)(x)

    x = GlobalAveragePooling2D()(x)

    if space['dense_1'] is not None:
        x = Dense(space['dense_1'], activation='relu')(x)
    if space['dense_2'] is not None:
        x = Dense(space['dense_2'], activation='relu')(x)

    x = Dense(conf.num_classes, activation='softmax')(x)
    model = Model(input_data, x)
    opt = space['opt'](lr=space['lr'])
    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer=opt)
    return model
