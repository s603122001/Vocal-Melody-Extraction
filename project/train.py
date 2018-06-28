import pickle
import numpy as np

from keras.utils.np_utils import to_categorical

from project.utils import note_res_downsampling, padding

def generator_audio(batch_size, timesteps, dataset, dataset_label,
                   phase='train', percentage_train=0.8,
                   constraint=False
                   ):
    X_48 = []
    X_12 = []
    # dataset loding
    for d in dataset:
        part = pickle.load(open(d, 'rb'))
        X_48 = np.concatenate([X_48, part])

    Y = []
    # label loading
    for d in dataset_label:
        part = pickle.load(open(d, 'rb'))
        Y = np.concatenate([Y, part])

    # Set chorale_indices
    if phase == 'train':
        chorale_indices = np.arange(int(len(X_48) * percentage_train))
    if phase == 'test':
        chorale_indices = np.arange(int(len(X_48) * percentage_train), len(X_48))
    if phase == 'all':
        chorale_indices = np.arange(int(len(X_48)))

    for a in range(len(X_48)):
        if (a in chorale_indices):
            new_x = np.array(X_48[a][:, :, 0])

            new_x_12 = note_res_downsampling(new_x)
            new_x_12 = padding(new_x_12, 128, timesteps)
            X_12.append(new_x_12)

            new_x_48 = padding(new_x, 384, timesteps)
            X_48[a] = new_x_48

            new_y = np.array(Y[a])
            Y[a] = padding(new_y, 384, timesteps)

        else:
            X_48[a] = 0
            X_12.append(0)
            Y[a] = 0

    features_48 = []
    features_12 = []
    labels = []

    batch = 0

    while True:
        # control the training percentage between datasets(we use 1/2 mir1k and 1/2 medleydb)
        if (np.random.choice(np.arange(10)) < 5 and phase == 'train'):
            chorale_index = np.random.choice(np.arange(48))
        else:
            chorale_index = np.random.choice(chorale_indices)
        chorale = np.array(X_48[chorale_index])
        chorale_length = len(chorale)
        time_index = np.random.randint(0, chorale_length - timesteps)

        feature_48 = (X_48[chorale_index][time_index: time_index + timesteps])
        feature_48 = np.reshape(feature_48, (timesteps, 384, 1))

        feature_12 = (X_12[chorale_index][time_index: time_index + timesteps])
        feature_12 = np.reshape(feature_12, (timesteps, 128, 1))

        label = Y[chorale_index][time_index: time_index + timesteps]
        label = to_categorical(label, num_classes=2)

        features_48.append(feature_48)
        features_12.append(feature_12)
        labels.append(label)

        batch += 1

        # if there is a full batch
        if batch == batch_size:
            next_element = (
                np.array(features_48, dtype=np.float32),
                np.array(features_12, dtype=np.float32),
                np.array(labels, dtype=np.float32),
            )

            yield next_element

            batch = 0

            features_48 = []
            features_12 = []
            labels = []


def train_audio(model, timesteps, dataset_path, label_path, epoch=5, steps=6000, batch_size=12):
    generator_train = (({'input_score_48': features_48,
                         'input_score_12': features_12
                         },
                        {'prediction': labels}
                        )
                       for (features_48,
                            features_12,
                            labels
                            ) in generator_audio(batch_size, timesteps, dataset_path, label_path))

    generator_val = (({'input_score_48': features_48,
                       'input_score_12': features_12
                       },
                      {'prediction': labels}
                      )
                     for (features_48,
                          features_12,
                          labels
                          ) in generator_audio(batch_size, timesteps, dataset_path, label_path,
                                               phase='test',
                                              ))

    model.fit_generator(generator_train, steps_per_epoch=steps,
                        epochs=epoch, verbose=1, validation_data=generator_val,
                        validation_steps=200,
                        use_multiprocessing=False,
                        max_queue_size=100,
                        workers=1,
                        )


    return model