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
    plt.title('Final Training Accuracy for EigenRemove and Minimum Weight Selection')
    xs = [float(x) for x in results1.keys()]
    plt.errorbar(xs, [median(results1[x]['acc'])
                      for x in results1], yerr=[stdev(val['acc']) for key, val in results1.items()], label="EigenRemove", linestyle="dashed", marker="^", color="red", capsize=5)
    plt.errorbar(xs, [median(results2[x]['acc'])
                      for x in results2], yerr=[stdev(val['acc']) for key, val in results2.items()], label="Minimum Weight Selection", linestyle="dashed", marker="o", color="blue", capsize=5)
    plt.ylabel('Final Accuracy (%)')
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
