import keras

SHAKE_VALUE = 20

params = {
    'batch_size': [8, 16, 32],
    'filters_conv2d_1': [32, 64, 100, 200, 300, 400, 500],
    'maxpooling_1': [None, 2],
    'dropout1': [0.2, 0.3, 0.4, 0.5],
    'filters_conv2d_2': [None, 32, 64, 100, 200, 300, 400, 500],
    'maxpooling_2': [None, 2],
    'dropout2': [0.2, 0.3, 0.4, 0.5],
    'filters_conv2d_3': [None, 32, 64, 100, 200, 300, 400, 500],
    'maxpooling_3': [None, 2],
    'dropout3': [0.2, 0.3, 0.4, 0.5],
    'filters_conv2d_4': [None, 32, 64, 100, 200, 300, 400, 500],
    'maxpooling_4': [None, 2],
    'dropout4': [0.2, 0.3, 0.4, 0.5],
    'filters_conv2d_5': [None, 32, 64, 100, 200, 300, 400, 500],
    'maxpooling_5': [None, 2],
    'dropout5': [0.2, 0.3, 0.4, 0.5],
    'filters_conv2d_6': [None, 32, 64, 100, 200, 300, 400, 500],
    'maxpooling_6': [None, 2],
    'dropout6': [0.2, 0.3, 0.4, 0.5],
    'filters_conv2d_7': [None, 32, 64, 100, 200, 300, 400, 500],
    'maxpooling_7': [None, 2],
    'dropout7': [0.2, 0.3, 0.4, 0.5],
    'filters_conv2d_8': [None, 32, 64, 100, 200, 300, 400, 500],
    'maxpooling_8': [None, 2],
    'dropout8': [0.2, 0.3, 0.4, 0.5],
    'dense_1': [None, 128, 256, 512],
    'dense_2': [None, 128, 256, 512],
    'opt': [keras.optimizers.Adam, keras.optimizers.RMSprop,
            keras.optimizers.Adagrad],
    'lr': [0.000001, 0.00001, 0.0001, 0.001]
}

can_shake = ['filters_conv2d_1', 'filters_conv2d_2', 'filters_conv2d_3',
             'filters_conv2d_4', 'filters_conv2d_5', 'filters_conv2d_6',
             'filters_conv2d_7', 'filters_conv2d_8', 'dense_1', 'dense_2']

import random

assert(params['filters_conv2d_1'] is not None)

def create_space(length=1):
    space = []
    for _ in range(length):
        combination = {}
        for k in params:
            combination[k] = random.choice(params[k])
            if k in can_shake and combination[k] is not None:
                shaked = combination[k] + random.randint(-SHAKE_VALUE,
                                                         SHAKE_VALUE)
                if shaked > 0:
                    combination[k] = shaked
        space.append(combination)
    return space if len(space) > 1 else space[0]


def params_keys_comma_separated():
    s = ''
    for k in params:
        s += k + ','
    return s[:-1]


def params_values(p):
    values = []
    for k in p:
        values.append(str(p[k] if not callable(p[k]) else p[k].__name__))
    return values


def params_values_comma_separated(p):
    s = ''
    for k in p:
        s = str(p[k] if not callable(p[k]) else p[k].__name__) + ','
    return s[:-1]


if __name__ == "__main__":
    space = create_space()
    s = ''
    for k in space:
            s += str(space[k] if not callable(space[k]) else
                     space[k].__name__) + ','
    print(s[:-1])