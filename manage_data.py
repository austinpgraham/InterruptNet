import os
import sys
import json

from statistics import mean
from statistics import median
from statistics import variance

import matplotlib.pyplot as plt


def epochvsoptimes(results):
    xs = ['0.75', '0.667', '0.5', '0.333', '0.25'][::-1]
    plt.title('Op and Epoch Times at Various Changes')
    plt.plot(xs, [mean(val['epoch_time'])
                  for key, val in results.items()][::-1], label="Epoch Time")
    plt.plot(xs, [mean(val['op_time'])
                  for key, val in results.items()][::-1], label="Op Time")
    plt.legend()
    plt.xlabel('Neuron Delta')
    plt.ylabel('Total Time (ms)')
    plt.show()


def main():
    file_name = sys.argv[1]
    data = json.load(open(os.path.expanduser(file_name), 'r'))
    results = {x: {'losses': [], 'acc': [], 'epoch_time': [], 'op_time': []}
               for x in ('0.75', '0.667', '0.5', '0.333', '0.25')}
    for _iter in data:
        for key, d in _iter.items():
            d.pop('losses')
            d.pop('tot_time')
            etimes = d.pop('epoch_time')
            otimes = d.pop('op_time')
            acc = d.pop('final_acc')
            results[key]['acc'].append(acc)
            results[key]['losses'].extend(d.values())
            results[key]['epoch_time'].extend(etimes)
            results[key]['op_time'].extend(otimes)
    loss_avgs = []
    for key in results.keys():
        # avg_acc = sum(results[key]['acc']) / len(results[key]['acc'])
        # avg_loss = sum(results[key]['losses']) / \
        #     len(results[key]['losses'])
        avg_acc = median(results[key]['acc'])
        avg_loss = median(results[key]['losses'])
        loss_avgs.append(avg_loss)
        print("{}: accuracy: {} loss: {}".format(key, avg_acc, avg_loss))
    print("Variance of losses: {}".format(variance(loss_avgs)))
    epochvsoptimes(results)


if __name__ == '__main__':
    main()
