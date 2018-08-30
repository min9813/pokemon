import numpy as np
import pandas as pd
import chainer
import sys
from chainer import links as L
from chainer import functions as F
from PIL import Image
from chainer.backends.cuda import cupy as cp


class ConvBlock(chainer.Chain):

    def __init__(self, n_ch, pool_drop=False):
        w = chainer.initializers.HeNormal()
        super(ConvBlock, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(
                None, n_ch, 3, 1, 1, nobias=True, initialW=w)
            self.bn = L.BatchNormalization(n_ch)
        self.pool_drop = pool_drop

    def __call__(self, x):
        h = F.relu(self.bn(self.conv(x)))
        if self.pool_drop:
            h = F.max_pooling_2d(h, 2, 2)
            h = F.dropout(h, ratio=0.25)
        return h


class LinearBlock(chainer.Chain):

    def __init__(self, drop=False):
        w = chainer.initializers.HeNormal()
        super(LinearBlock, self).__init__()
        with self.init_scope():
            self.fc = L.Linear(None, 1024, initialW=w)
        self.drop = drop

    def __call__(self, x):
        h = F.relu(self.fc(x))
        if self.drop:
            h = F.dropout(h, ratio=0.3)
        return h


class DeepCNN(chainer.ChainList):

    def __init__(self, n_output):
        super(DeepCNN, self).__init__(
            ConvBlock(64),
            ConvBlock(64, True),
            ConvBlock(128),
            ConvBlock(128, True),
            ConvBlock(256),
            ConvBlock(256),
            ConvBlock(256),
            ConvBlock(256, True),
            LinearBlock(),
            LinearBlock(True),
            L.Linear(None, n_output),
        )

    def __call__(self, x):
        for f in self:
            x = f(x)
        return x


class VGG16Model(chainer.Chain):
    def __init__(self, out_size):
        super(VGG16Model, self).__init__(
            base=L.VGG16Layers(),
            fc=L.Linear(None, out_size)
        )

    def __call__(self, x):
        h = self.base(x, layers=["fc7"])
        y = self.fc(h["fc7"])
        return y


def load_dataset(image_types=["original"], model=None, label_type="float", image_size=(96, 96), ground_data=None):
    def make_one_data(number, load_path, type_data, label_type, _start_index, model=None, image_size=(96, 96)):
        pokemon_image = Image.open(load_path)
        if model is None:
            if (pokemon_image.size[0] != image_size[0] or pokemon_image.size[1] != image_size[1]):
                pokemon_image = pokemon_image.resize(image_size)
            pokemon_image = np.asarray(pokemon_image, dtype=np.float32)
            pokemon_image = pokemon_image / np.max(pokemon_image)
        elif model == "vgg":
            pokemon_image = L.model.vision.vgg.prepare(
                pokemon_image, size=(224, 224))
        else:
            sys.exit("model must be [None, \"vgg\"]")
        pokemon_type = type_data.iloc[number - 1, _start_index:].astype(label_type)
        pokemon_type = np.asarray(pokemon_type, dtype=label_type)
        pokemon_data = (pokemon_image, pokemon_type)
        return pokemon_data
    if label_type == "float":
        label_type = np.float32
    elif label_type == "int":
        label_type = np.int32
    else:
        sys.exit("label_type argument mast be either \"float\" or \"int\"")
    print("start loading data ...")
    if ground_data is None:
        type_data = pd.read_csv("pokemon_data.csv")
        start_index = type_data.columns.tolist().index("None")
        type_data = type_data.iloc[:, start_index:]
    else:
        type_data = ground_data
    train_data = []
    test_data = []
    for image_type in image_types:
        print("start loading {} image ...".format(image_type))
        for i in range(1, 722):
            if model is "vgg":
                image_file_path = "./pokemon_img/daisukiclub/{}/{}.png".format(
                    image_type, str(i))
            else:
                image_file_path = "./pokemon_img/daisuki_resize/{}/{}.png".format(
                    image_type, str(i))
            pokemon_data = make_one_data(number=i,
                                         load_path=image_file_path,
                                         type_data=type_data,
                                         label_type=label_type,
                                         _start_index=start_index,
                                         model=model,
                                         image_size=image_size,
                                         )
            train_data.append(pokemon_data)
            image_file_path = "./pokemon_img/yakkun/{}/{}.png".format(
                image_type, str(i))
            pokemon_data = make_one_data(number=i,
                                         load_path=image_file_path,
                                         type_data=type_data,
                                         label_type=label_type,
                                         _start_index=start_index,
                                         model=model,
                                         image_size=image_size)
            train_data.append(pokemon_data)
        for i in range(722, 803):
            if model is "vgg":
                image_file_path = "./pokemon_img/daisukiclub/{}/{}.png".format(
                    image_type, str(i))
            else:
                image_file_path = "./pokemon_img/daisuki_resize/{}/{}.png".format(
                    image_type, str(i))
            pokemon_data = make_one_data(number=i,
                                         load_path=image_file_path,
                                         type_data=type_data,
                                         label_type=label_type,
                                         _start_index=start_index,
                                         model=model,
                                         image_size=image_size)
            test_data.append(pokemon_data)
        print("finish loading {} image".format(image_type))
    print("finish loading data")
    return train_data, test_data


def type_accuracy(x, t):
    # x is Chainer.Variable, t is cupy.array
    p_max_indices = cp.argpartition(-x.array, 2)[:, :2]
    t_max_indices = cp.argpartition(-t, 2)[:, :2]
    accuracy_rate = cp.sum(p_max_indices == t_max_indices) / \
        (2 * p_max_indices.shape[0])
    return accuracy_rate


if __name__ == "__main__":
    pass
