# 텐서보드를 사용해서 Projector를 구현할 때 오류가 있음
# 이 오류를 해결하기 위해서 작성해야 할 것
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
import torchvision
import matplotlib.pyplot as plt
import numpy as np
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
# console: tensorboard --logdir=runs --bind_all
# tensorboard --logdir=runs --bind_all > /dev/null 2>&1 &


# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def set_tensorboard_writer(name):
    writer = SummaryWriter(name) # 'runs/fashion_mnist_experiment_1'
    return writer


def inspect_model(writer, model, data):
    writer.add_graph(model, data)


def close_tensorboard_writer(writer):
    writer.close()


def add_image_on_tensorboard(writer, dataloader, desc="dataset"):
    tdataiter = iter(dataloader)
    images, labels = tdataiter.next()
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image('{}/images of {}'.format(desc, desc), img_grid)

