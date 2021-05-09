import struct

import numpy as np
from numpy import expand_dims
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Input,
    LeakyReLU,
    UpSampling2D,
    ZeroPadding2D,
    add,
    concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.python.keras.saving.save import load_model


def _conv_block(inp, convs, skip=True):
    x = inp
    count = 0
    for conv in convs:
        if count == (len(convs) - 2) and skip:
            skip_connection = x
        count += 1
        if conv["stride"] > 1:
            x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        x = Conv2D(
            conv["filter"],
            conv["kernel"],
            strides=conv["stride"],
            padding="valid" if conv["stride"] > 1 else "same",
            name="conv_" + str(conv["layer_idx"]),
            use_bias=False if conv["bnorm"] else True,
        )(x)
        if conv["bnorm"]:
            x = BatchNormalization(
                epsilon=0.001, name="bnorm_" + str(conv["layer_idx"])
            )(x)
        if conv["leaky"]:
            x = LeakyReLU(alpha=0.1, name="leaky_" + str(conv["layer_idx"]))(x)
    return add([skip_connection, x]) if skip else x


def make_yolov3_model():
    input_image = Input(shape=(None, None, 3))
    x = _conv_block(
        input_image,
        [
            {
                "filter": 32,
                "kernel": 3,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 0,
            },
            {
                "filter": 64,
                "kernel": 3,
                "stride": 2,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 1,
            },
            {
                "filter": 32,
                "kernel": 1,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 2,
            },
            {
                "filter": 64,
                "kernel": 3,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 3,
            },
        ],
    )
    x = _conv_block(
        x,
        [
            {
                "filter": 128,
                "kernel": 3,
                "stride": 2,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 5,
            },
            {
                "filter": 64,
                "kernel": 1,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 6,
            },
            {
                "filter": 128,
                "kernel": 3,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 7,
            },
        ],
    )
    x = _conv_block(
        x,
        [
            {
                "filter": 64,
                "kernel": 1,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 9,
            },
            {
                "filter": 128,
                "kernel": 3,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 10,
            },
        ],
    )
    x = _conv_block(
        x,
        [
            {
                "filter": 256,
                "kernel": 3,
                "stride": 2,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 12,
            },
            {
                "filter": 128,
                "kernel": 1,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 13,
            },
            {
                "filter": 256,
                "kernel": 3,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 14,
            },
        ],
    )
    for i in range(7):
        x = _conv_block(
            x,
            [
                {
                    "filter": 128,
                    "kernel": 1,
                    "stride": 1,
                    "bnorm": True,
                    "leaky": True,
                    "layer_idx": 16 + i * 3,
                },
                {
                    "filter": 256,
                    "kernel": 3,
                    "stride": 1,
                    "bnorm": True,
                    "leaky": True,
                    "layer_idx": 17 + i * 3,
                },
            ],
        )
    skip_36 = x
    x = _conv_block(
        x,
        [
            {
                "filter": 512,
                "kernel": 3,
                "stride": 2,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 37,
            },
            {
                "filter": 256,
                "kernel": 1,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 38,
            },
            {
                "filter": 512,
                "kernel": 3,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 39,
            },
        ],
    )
    for i in range(7):
        x = _conv_block(
            x,
            [
                {
                    "filter": 256,
                    "kernel": 1,
                    "stride": 1,
                    "bnorm": True,
                    "leaky": True,
                    "layer_idx": 41 + i * 3,
                },
                {
                    "filter": 512,
                    "kernel": 3,
                    "stride": 1,
                    "bnorm": True,
                    "leaky": True,
                    "layer_idx": 42 + i * 3,
                },
            ],
        )
    skip_61 = x
    x = _conv_block(
        x,
        [
            {
                "filter": 1024,
                "kernel": 3,
                "stride": 2,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 62,
            },
            {
                "filter": 512,
                "kernel": 1,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 63,
            },
            {
                "filter": 1024,
                "kernel": 3,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 64,
            },
        ],
    )
    for i in range(3):
        x = _conv_block(
            x,
            [
                {
                    "filter": 512,
                    "kernel": 1,
                    "stride": 1,
                    "bnorm": True,
                    "leaky": True,
                    "layer_idx": 66 + i * 3,
                },
                {
                    "filter": 1024,
                    "kernel": 3,
                    "stride": 1,
                    "bnorm": True,
                    "leaky": True,
                    "layer_idx": 67 + i * 3,
                },
            ],
        )
    x = _conv_block(
        x,
        [
            {
                "filter": 512,
                "kernel": 1,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 75,
            },
            {
                "filter": 1024,
                "kernel": 3,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 76,
            },
            {
                "filter": 512,
                "kernel": 1,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 77,
            },
            {
                "filter": 1024,
                "kernel": 3,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 78,
            },
            {
                "filter": 512,
                "kernel": 1,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 79,
            },
        ],
        skip=False,
    )
    yolo_82 = _conv_block(
        x,
        [
            {
                "filter": 1024,
                "kernel": 3,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 80,
            },
            {
                "filter": 255,
                "kernel": 1,
                "stride": 1,
                "bnorm": False,
                "leaky": False,
                "layer_idx": 81,
            },
        ],
        skip=False,
    )
    x = _conv_block(
        x,
        [
            {
                "filter": 256,
                "kernel": 1,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 84,
            }
        ],
        skip=False,
    )
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_61])
    x = _conv_block(
        x,
        [
            {
                "filter": 256,
                "kernel": 1,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 87,
            },
            {
                "filter": 512,
                "kernel": 3,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 88,
            },
            {
                "filter": 256,
                "kernel": 1,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 89,
            },
            {
                "filter": 512,
                "kernel": 3,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 90,
            },
            {
                "filter": 256,
                "kernel": 1,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 91,
            },
        ],
        skip=False,
    )
    yolo_94 = _conv_block(
        x,
        [
            {
                "filter": 512,
                "kernel": 3,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 92,
            },
            {
                "filter": 255,
                "kernel": 1,
                "stride": 1,
                "bnorm": False,
                "leaky": False,
                "layer_idx": 93,
            },
        ],
        skip=False,
    )
    x = _conv_block(
        x,
        [
            {
                "filter": 128,
                "kernel": 1,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 96,
            }
        ],
        skip=False,
    )
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_36])
    yolo_106 = _conv_block(
        x,
        [
            {
                "filter": 128,
                "kernel": 1,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 99,
            },
            {
                "filter": 256,
                "kernel": 3,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 100,
            },
            {
                "filter": 128,
                "kernel": 1,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 101,
            },
            {
                "filter": 256,
                "kernel": 3,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 102,
            },
            {
                "filter": 128,
                "kernel": 1,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 103,
            },
            {
                "filter": 256,
                "kernel": 3,
                "stride": 1,
                "bnorm": True,
                "leaky": True,
                "layer_idx": 104,
            },
            {
                "filter": 255,
                "kernel": 1,
                "stride": 1,
                "bnorm": False,
                "leaky": False,
                "layer_idx": 105,
            },
        ],
        skip=False,
    )
    model = Model(input_image, [yolo_82, yolo_94, yolo_106])
    return model


class WeightReader:
    def __init__(self, weight_file):
        with open(weight_file, "rb") as w_f:
            (max,) = struct.unpack("i", w_f.read(4))
            (min,) = struct.unpack("i", w_f.read(4))
            (rev,) = struct.unpack("i", w_f.read(4))
            if (max * 10 + min) >= 2 and max < 1000 and min < 1000:
                w_f.read(8)
            else:
                w_f.read(4)
            transpose = (max > 1000) or (min > 1000)
            binary = w_f.read()
        self.offset = 0
        self.all_weights = np.frombuffer(binary, dtype="float32")

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size : self.offset]

    def load_weights(self, model):
        for i in range(106):
            try:
                conv_layer = model.get_layer("conv_" + str(i))
                print("loading weights of convolution #" + str(i))
                if i not in [81, 93, 105]:
                    norm_layer = model.get_layer("bnorm_" + str(i))
                    size = np.prod(norm_layer.get_weights()[0].shape)
                    beta = self.read_bytes(size)
                    gamma = self.read_bytes(size)
                    mean = self.read_bytes(size)
                    var = self.read_bytes(size)
                    weights = norm_layer.set_weights([gamma, beta, mean, var])
                if len(conv_layer.get_weights()) > 1:
                    bias = self.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    kernel = kernel.reshape(
                        list(reversed(conv_layer.get_weights()[0].shape))
                    )
                    kernel = kernel.transpose([2, 3, 1, 0])
                    conv_layer.set_weights([kernel, bias])
                else:
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    kernel = kernel.reshape(
                        list(reversed(conv_layer.get_weights()[0].shape))
                    )
                    kernel = kernel.transpose([2, 3, 1, 0])
                    conv_layer.set_weights([kernel])
            except ValueError:
                print("no convolution #" + str(i))

    def reset(self):
        self.offset = 0


# Here we will create a new model, get the weights and
# compane the two to make a new model
model = make_yolov3_model()
weight_reader = WeightReader("yolov3.weights")
weight_reader.load_weights(model)
model.save("model.h5")


# Function that will handle loading and prepare an
# an image for the model
def load_image_pixels(filename, shape):
    image = load_img(filename)
    width, height = image.size
    image = load_img(filename, target_size=shape)
    image = img_to_array(image)
    image = image.astype("float32")
    image /= 255.0
    image = expand_dims(image, 0)
    return image, width, height


# load the yolov3 model that was created in the last
# step of 01_save_model.py
model = load_model("model.h5", compile=False)
input_w, input_h = 416, 416
photo_filename = "vehicles.jpg"
image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))
pred = model.predict(image)
print([a.shape for a in pred])
