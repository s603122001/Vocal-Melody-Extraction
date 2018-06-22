import argparse
import os

import numpy as np

import MelodyExt
from utils import load_model, matrix_parser
from test import inference


def main():
    # Arguments


    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name',
                        help = 'path to existing model (default: %(default)s',
                        type = str, default = 'transfer_audio_directly')
    parser.add_argument('-i', '--input_file',
                        help='path to input file (default: %(default)s',
                        type=str, default='train01.wav')

    args = parser.parse_args()
    print(args)


    # load wav
    song = args.input_file

    # Feature extraction
    feature = MelodyExt.feature_extraction(song)
    feature = np.transpose(feature[0:4], axes=(2, 1, 0))

    # load model
    model = load_model(args.model_name)

    # Inference
    print(feature[:,:,0].shape)
    extract_result = inference(feature= feature[:,:,0],
                               model = model,
                               channel=1)

    # Output
    r = matrix_parser(extract_result)

    np.savetxt("out_seg.txt", r)


if __name__ == '__main__':
    main()