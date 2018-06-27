import numpy as np
from keras.models import model_from_json, model_from_yaml
from mir_eval import melody

TIMESTEP = 128
SUBDIVISION = 8


def freq2midi(f):
    return 69 + 12*np.log2(f/440)


def midi2freq(m):
    return 2**((m - 69)/ 12) * 440


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def note_res_downsampling(score):
    # filter
    f = [0.1, 0.2, 0.4, 0.2, 0.1]
    r = len(f) // 2

    new_score = np.zeros((score.shape[0], 88))

    pad = np.zeros((new_score.shape[0], 2))
    score = np.concatenate([pad, score], axis=1)

    f_aug = np.tile(f, (new_score.shape[0], 1))

    for i in range(0, 352, 4):
        cent = i + 2
        lb = max(0, cent - r)
        ub = min(353, (cent + 1) + r)
        new_score[:, i // 4] = np.sum(score[:, lb:ub] * f_aug, axis=1)

    return new_score


def padding(x,
            feature_num,
            timesteps,
            dimension=False):

    extended_chorale = np.array(x)

    if (((feature_num - x.shape[1]) % 2) == 0):
        p_t = (feature_num - x.shape[1]) // 2
        p_b = p_t

    else:
        p_t = (feature_num - x.shape[1]) // 2
        p_b = p_t + 1

    top = np.zeros((extended_chorale.shape[0], p_t))
    bottom = np.zeros((extended_chorale.shape[0], p_b))
    extended_chorale = np.concatenate([top, extended_chorale, bottom], axis=1)

    padding_dimensions = (timesteps,) + extended_chorale.shape[1:]

    padding_start = np.zeros(padding_dimensions)
    padding_end = np.zeros(padding_dimensions)

    padding_start[:, :p_t] = 1
    padding_end[:, -p_b:] = 1

    extended_chorale = np.concatenate((padding_start,
                                       extended_chorale,
                                       padding_end),
                                      axis=0)

    if (dimension):
        return extended_chorale, p_t, p_b
    else:
        return extended_chorale


def matrix_parser(m):
    x = np.zeros(shape=(m.shape[0], 2))

    for i in range(len(m)):
        if (np.sum(m[i]) != 0):
            x[i][0] = 1
            x[i][1] = midi2freq(np.argmax(m[i]) / 4 + 21)

    x[:, 1] = melody.hz2cents(x[:, 1])

    return x


def load_model(model_name):
    """


    """
    ext = '.yaml'
    model = model_from_yaml(open(model_name + ext).read())
    model.load_weights(model_name + '_weights.h5')

    print("model " + model_name + " loaded")
    return model


def save_model(model, model_name, overwrite=False):
    # SAVE MODEL

    string = model.to_yaml()
    ext = '.yaml'

    open(model_name + ext, 'w').write(string)
    model.save_weights(model_name + '_weights.h5', overwrite=overwrite)
    print("model " + model_name + " saved")


def model_copy(origin, target):

    for index, layer in enumerate(target.layers):
        if layer.__class__.__name__ == 'LSTM':
            weights = origin.layers[index].get_weights()
            units = weights[1].shape[0]
            bias = weights[2]
            if len(bias) == units * 8:
                # reshape the kernels
                kernels = np.split(weights[0], 4, axis=1)
                kernels = [kernel.reshape(-1).reshape(kernel.shape, order='F') for kernel in kernels]
                weights[0] = np.concatenate(kernels, axis=1)

                # transpose the recurrent kernels
                recurrent_kernels = np.split(weights[1], 4, axis=1)
                recurrent_kernels = [kernel.T for kernel in recurrent_kernels]
                weights[1] = np.concatenate(recurrent_kernels, axis=1)

                # split the bias into half and merge
                weights[2] = bias[:units * 4] + bias[units * 4:]
                layer.set_weights(weights)
                print("Set success")
        else:
            layer.set_weights(origin.layers[index].get_weights())