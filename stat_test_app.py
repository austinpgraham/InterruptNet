import os
import sys
import json

from statistics import mean
from statistics import stdev
from statistics import median

import matplotlib.pyplot as plt

from scipy import stats


def get_losses(data):
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
    return results


def plot_losses(results1, results2):
    plt.title('Median Loss Increase for WeakExpand and Zero Weight Expansion')
    xs = [float(x) for x in results1.keys()]
    plt.plot(xs, [median(results1[x]['losses'])
                  for x in results1], label="WeakExpand", linestyle="dashed", marker="^", color="red")
    plt.plot(xs, [median(results2[x]['losses'])
                  for x in results2], label="Zero Weight Expansion", linestyle="dashed", marker="o", color="blue")
    plt.fill_between(xs, [median(results1[key]['losses']) - stdev(val['losses'])
                          for key, val in results1.items()], [median(results1[key]['losses']) + stdev(val['losses'])
                                                              for key, val in results1.items()], facecolor="red", alpha=0.3)
    plt.fill_between(xs, [median(results2[key]['losses']) - stdev(val['losses'])
                          for key, val in results2.items()], [median(results2[key]['losses']) + stdev(val['losses'])
                                                              for key, val in results2.items()], facecolor="blue", alpha=0.3)
    plt.ylabel('Median Loss')
    plt.xlabel('Neuron Delta')
    plt.legend()
    plt.show()


def main():
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    data1 = json.load(open(os.path.expanduser(file1), 'r'))
    data2 = json.load(open(os.path.expanduser(file2), 'r'))
    results1 = get_losses(data1)
    results2 = get_losses(data2)
    plot_losses(results1, results2)
    losses1 = []
    losses2 = []
    for key in results1:
        losses1.extend(results1[key]['acc'])
        losses2.extend(results2[key]['acc'])
    _, p = stats.ttest_ind(losses1, losses2)
    print(p)


if __name__ == '__main__':
    main()
