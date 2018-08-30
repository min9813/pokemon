import numpy as np
import pandas as pd
import chainer
import re
import sys
import argparse
from chainer import iterators, training
from chainer import links as L
from chainer import functions as F
from chainer.backends import cuda
from chainer.backends.cuda import cupy as cp
from chainer.training import extensions
from chainer.datasets import split_dataset_random
from model_and_function import load_dataset, type_accuracy, VGG16Model, DeepCNN
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
FutureWarning("ignore")


POKEMON_TYPE_NUMBER = 19
MODEL_LIST = ["vgg16", "normal"]


def train(model=None,
          ground_data=None,
          disabled_update_last_layer="conv4",
          max_epoch=30,
          batch_size=64,
          gpu_id=0,
          _8th_data=False,
          image_types=["original"],
          out_file="vgg16"):
    if gpu_id >= 0:
        chainer.cuda.set_max_workspace_size(512 * 1024 * 1024)
        chainer.config.autotune = True

    if _8th_data:
        train_data, valid_data = load_dataset(
            image_types=image_types, label_type="int", ground_data=ground_data)
        print(train_data[0])
        print("train data size :", len(train_data))
        print("valid data size :", len(valid_data))
        train_iter = iterators.SerialIterator(
            train_data, batch_size=batch_size)
        valid_iter = iterators.SerialIterator(
            valid_data, batch_size=batch_size, repeat=False, shuffle=False)

    else:
        train_val, test_data = load_dataset(
            image_types=image_types, label_type="int", model="vgg", ground_data=ground_data)
        train_size = int(len(train_val) * 0.9)
        train_data, valid_data = split_dataset_random(
            train_val, train_size, seed=0)
        print("train data size :", len(train_data))
        print("valid data size :", len(valid_data))
        print("test data size :", len(test_data))
        train_iter = iterators.SerialIterator(
            train_data, batch_size=batch_size)
        valid_iter = iterators.SerialIterator(
            valid_data, batch_size=batch_size, repeat=False, shuffle=False)
        # test_iter = iterators.SerialIterator(
        #     test_data, batch_size=batch_size, repeat=False, shuffle=False)

    model = L.Classifier(
        model, lossfun=F.sigmoid_cross_entropy, accfun=type_accuracy)
    optimizer = chainer.optimizers.Adam(alpha=1e-4)
    optimizer.setup(model)

    stop_conv_layer_int = disabled_update_last_layer.find("conv")
    stop_fc_layer_int = disabled_update_last_layer.find("fc")
    if stop_conv_layer_int >= 0:
        last_layer_int = disabled_update_last_layer[stop_conv_layer_int + 4]
        detect_stop_layer_str = "(1"
        for layer in range(2, last_layer_int + 1):
            detect_stop_layer_str += ("|" + str(layer))
        detect_stop_layer_str += ")_"
    if stop_conv_layer_int >= 0:
        last_layer_int = disabled_update_last_layer[stop_fc_layer_int + 2]
        if last_layer_int >= 8:
            sys.exit("stop layer must before fc8 layer")
        detect_stop_layer_str = "(1|2|3|4|5"
        for layer in range(6, last_layer_int + 1):
            detect_stop_layer_str += ("|" + str(layer))
        detect_stop_layer_str += ")_"
    detect_stop_layer_re = re.compile(detect_stop_layer_str)
    print("* disable updating layer *")
    for layer in model.predictor.base._children:
        if re.search(detect_stop_layer_re, layer) is not None:
            print(layer)
            model.predictor.base[layer].disable_update()
    print()

    if gpu_id >= 0:
        model.to_gpu(gpu_id)

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)

    trainer = training.Trainer(updater, (max_epoch, "epoch"), out=out_file)

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.snapshot(
        filename='snapshot_epoch-{.updater.epoch}'))
    trainer.extend(extensions.Evaluator(
        valid_iter, model, device=gpu_id), name='val')
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(
        ['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()


def infer(test_data, infer_model, type_columns, gpu_id=0):
    if gpu_id >= 0:
        infer_model.to_gpu(gpu_id)

    x, t = test_data

    # plt.imshow(x, cmap="gray")
    # plt.show()
    x = cp.array(L.model.vision.vgg.prepare(x, size=(224, 224)))
    x = cp.expand_dims(x, 0)
    print(x.shape)

    print("start infering ...")
    with chainer.using_config("train", False), chainer.using_config("enable_backprop", False):
        y = infer_model(x)
    print("finish infering")
    y = y.array

    y = cuda.to_cpu(y)

    t_label = np.array(type_columns)[t == 1]
    t_label = list(t_label)
    p_label = np.array(type_columns)[np.sort(
        np.argpartition(-y.reshape(-1), 2)[:2])]
    p_label = list(p_label)
    return t_label[::-1], p_label[::-1]


parser = argparse.ArgumentParser(
    description="This file is used to train model")
parser.add_argument("model",
                    help="what model to use",
                    choices=MODEL_LIST,
                    type=str)
parser.add_argument(
    "-t", "--train", help="flag to train model", action="store_true")
parser.add_argument("-m", "--max_epoch",
                    help="max epoch time", type=int, default=3)
parser.add_argument("-f", "--output_file",
                    help="file to output the training data", type=str, default="TEST")
parser.add_argument("-p", "--pokemon", help="# of pokemon image to infer", type=int,
                    default=1)

args = parser.parse_args()

if __name__ == "__main__":
    pokemon_data = pd.read_csv("pokemon_data.csv")
    POKEMON_TYPE_COLUMNS = pokemon_data.columns.tolist()
    POKEMON_TYPE_COLUMNS = POKEMON_TYPE_COLUMNS[POKEMON_TYPE_COLUMNS.index(
        "None"):]
    pokemon_type_data = pokemon_data[POKEMON_TYPE_COLUMNS]
    image_types = ["gaussian",
                   "high_contrast",
                   "gaussian_noise",
                   "high_gamma",
                   "low_gamma",
                   "original",
                   "x_reverse",
                   "y_reverse"]
    if args.model == "vgg16":
        model = VGG16Model(POKEMON_TYPE_NUMBER)
    else:
        model = DeepCNN(POKEMON_TYPE_NUMBER)
    if args.train:
        print("Training Mode")
        train(model=model,
              image_types=image_types,
              ground_data=pokemon_type_data,
              max_epoch=3,
              out_file=args.output_file)
    else:
        print("Infering Mode")
        load_file = "./pokemon_img/daisukiclub/original/{}.png".format(
            str(args.pokemon))
        if args.model == "vgg16":
            load_model_path = "./VGG16/snapshot_epoch-4"
        image = Image.open(load_file)
        chainer.serializers.load_npz(load_model_path,
                                     model,
                                     path="updater/model:main/predictor/")
        test_data = (image, pokemon_type_data.iloc[args.pokemon - 1, :])
        t_label, p_label = infer(
            test_data, model, POKEMON_TYPE_COLUMNS)

        print("Name of pokemon:",
              pokemon_data["Name"][pokemon_data["#"] == args.pokemon])
        print("Type of the pokemon:")
        for _id_type, _type in enumerate(t_label):
            print("True Type {}:{} | Predict Type {}:{}".format(
                str(_id_type + 1), _type, str(_id_type + 1), p_label[_id_type]))
