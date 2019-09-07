from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense,\
                         GlobalAveragePooling2D, Input

# Model based on https://github.com/tomasz-oponowicz/spoken_language_identification

def build_model(conf: Config, input_shape) -> Model:
    input_data = Input(shape=input_shape)
    x = Conv2D(
        filters=500,
        kernel_size=3,
        activation='relu',
        data_format='channels_last'
    )(input_data)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    x = Conv2D(
        filters=500,
        kernel_size=3,
        activation='relu'
    )(x)
    x = Conv2D(
        filters=500,
        kernel_size=3,
        activation='relu'
    )(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    x = Conv2D(
        filters=64,
        kernel_size=3,
        activation='relu'
    )(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    x = Conv2D(
        filters=128,
        kernel_size=3,
        activation='relu'
    )(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    x = GlobalAveragePooling2D()(x)

    if conf.use_tb_embeddings:
        x = Dense(128, activation='relu', name='features')(x)

    x = Dense(conf.num_classes, activation='softmax')(x)
    model = Model(input_data, x)

    opt = Adam(lr=conf.learning_rate)
    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer=opt)
    return model
