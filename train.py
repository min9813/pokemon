# import numpy as np
import pandas as pd
import chainer
import re
import sys
from chainer import iterators, training
from chainer import Link as L
from chainer import Function as F
# from chainer.backends.cuda import cupy as cp
from chainer.training import extensions
from chainer.datasets import split_dataset_random
from model_and_function import load_dataset, type_accuracy, VGG16Model


def train(model=None,
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
            image_types=image_types, label_type="int")
        print(train_data[0])
        print("train data size :", len(train_data))
        print("valid data size :", len(valid_data))
        train_iter = iterators.SerialIterator(
            train_data, batch_size=batch_size)
        valid_iter = iterators.SerialIterator(
            valid_data, batch_size=batch_size, repeat=False, shuffle=False)

    else:
        train_val, test_data = load_dataset(
            image_types=image_types, label_type="int", model="vgg")
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

    model = L.Classifier(model, lossfun=F.sigmoid_cross_entropy, accfun=type_accuracy)
    optimizer = chainer.optimizers.Adam(alpha=1e-4)
    optimizer.setup(model)

    stop_conv_layer_int = disabled_update_last_layer.find("conv")
    stop_fc_layer_int = disabled_update_last_layer.find("fc")
    if stop_conv_layer_int >= 0:
        last_layer_int = disabled_update_last_layer[stop_conv_layer_int+4]
        detect_stop_layer_str = "(1"
        for layer in range(2, last_layer_int+1):
            detect_stop_layer_str += ("|"+str(layer))
        detect_stop_layer_str += ")_"
    if stop_conv_layer_int >= 0:
        last_layer_int = disabled_update_last_layer[stop_fc_layer_int+2]
        if last_layer_int >= 8:
            sys.exit("stop layer must before fc8 layer")
        detect_stop_layer_str = "(1|2|3|4|5"
        for layer in range(6, last_layer_int+1):
            detect_stop_layer_str += ("|"+str(layer))
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


if __name__ == "__main__":
    pokemon_data = pd.read_csv("pokemon_data.csv")
    image_types = ["gaussian",
                   "high_contrast",
                   "gaussian_noise",
                   "high_gamma",
                   "low_gamma",
                   "original",
                   "x_reverse",
                   "y_reverse"]
    model = VGG16Model(len(pokemon_data.columns[12:]))
    train(model=model, image_types=image_types, max_epoch=3, out_file="vgg16_test")
