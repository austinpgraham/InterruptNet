from pdb import set_trace
import copy
import json
import torch

from random import shuffle

from vgg16 import VGG16
from vgg16 import load_cifar_data

from eigen_remove import select_gpu

from matplotlib import pyplot as plt


TOTAL_EPOCHS = 50
# ORDERING = ["c{}".format(i) for i in range(2, 11)] + \
#     ["f{}".format(j) for j in range(1, 3)]
# shuffle(ORDERING)
ORDERING = json.load(open('orderings.json', 'r'))
SIZES = [1, 1.5, 2, 2.5]


def plot_losses(losses):
    plt.plot(list(range(1, TOTAL_EPOCHS)), losses)
    plt.title('Losses Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Batch Loss')
    plt.show()


def do_one_iter(train_loader, test_loader, size_ratio):
    layers = copy.deepcopy(ORDERING)
    print("Building network...")
    # Build network
    network = VGG16()
    # initial test
    network.test_net(test_loader)
    # Track total losses
    total_losses = []
    spectral_norms = []
    obj_rations = {'losses': {}, 'final_acc': None}
    for epoch in range(TOTAL_EPOCHS):
        if epoch % 2 == 1:
            if layers:
                op = layers.pop()
                _type, _int = op[0], int(op[1:])
                if _type == 'c':
                    print("Expanding conv{}...".format(_int))
                    curr_size = getattr(
                        network, "conv{}".format(_int)).out_channels
                    network.expand_conv_layer(
                        _int, int(curr_size + (curr_size * size_ratio)))
                else:
                    print("Expanding fc{}...".format(_int))
                    curr_size = getattr(
                        network, "fc{}".format(_int)).out_features
                    network.expand_fc_layer(
                        _int, int(curr_size + (curr_size * size_ratio)))
        # Train then test at every epoch
        train_losses, first = network.train_net(epoch, train_loader)
        acc = network.test_net(test_loader)
        if epoch % 2 == 1:
            # Track the jump in loss
            if layers:
                obj_rations[op] = first / total_losses[-1]
        total_losses.extend(train_losses)
    obj_rations['final_acc'] = acc.item()
    return obj_rations


def main():
    torch.backends.cudnn.enabled = True
    torch.manual_seed(1)
    torch.cuda.empty_cache()
    results = [{} for _ in range(5)]
    with select_gpu(0) as _:
        # Grab data
        train_loader, test_loader = load_cifar_data()
        for ratio in SIZES:
            for i in range(5):
                print("-----{}-----".format(ratio))
                iter_result = do_one_iter(train_loader, test_loader, ratio)
                results[i][str(round(ratio, 3))] = iter_result
    json.dump(results, open("we_results.json", "w"))


if __name__ == '__main__':
    main()
