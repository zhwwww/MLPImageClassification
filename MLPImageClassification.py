import gzip, struct
import os
import numpy as np
import argparse
import fit
import mxnet as mx
import math
def read_data(label, image):
    label = os.path.join('data',label)
    image = os.path.join('data',image)
    with gzip.open(os.path.join(label)) as f1:
        # head info, big edian , 2 integers
        magic, num = struct.unpack(">II", f1.read(8))
        label = np.fromstring(f1.read(), dtype=np.int8)
    with gzip.open(os.path.join(image), 'rb') as fimg:
        # head info , big edian , 4 integers
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        # reshape to num*rows*cols
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label,image)

def to4d(img):
    """
    reshape to 4D arrays
    """
    # 1D value array reshape to num*1*28*28 and normalize to [0,1]
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255

def get_mnist_iter(args, kv):
    """
    create data iterator with NDArrayIter
    """
    (train_lbl, train_img) = read_data(
            'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz')
    (val_lbl, val_img) = read_data(
            't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')
    # Returns an iterator
    train = mx.io.NDArrayIter(
        to4d(train_img), train_lbl, args.batch_size, shuffle=True)
    val = mx.io.NDArrayIter(
        to4d(val_img), val_lbl, args.batch_size)
    return (train, val)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train mnist',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-classes', type=int, default=10,
                        help='the number of classes')
    parser.add_argument('--num-examples', type=int, default=60000,
                        help='the number of training examples')
    parser.add_argument('--add_stn',  action="store_true", default=False, help='Add Spatial Transformer Network Layer (lenet only)')
    parser.add_argument('--image_shape', default='1, 28, 28', help='shape of training images')
    fit.add_fit_args(parser)
    parser.set_defaults(
        # network
        network        = 'mlp',
        # train
        gpus           = None,
        batch_size     = 64,
        disp_batches   = 100,
        num_epochs     = 20,
        lr             = .05,
        lr_step_epochs = '10',
    )
    args = parser.parse_args()

    # load network
    from importlib import import_module
    net = import_module(args.network)
    sym = net.get_symbol(**vars(args))

    # train
    # data_loader : get_mnist_iter 
    # return : (train, val)
    progressBar = [mx.callback.ProgressBar(math.ceil(int(args.num_examples / 1) / args.batch_size))]
    fit.fit(args, sym, get_mnist_iter,batch_end_callback=progressBar)
