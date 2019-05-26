import os
import sys
import json

from statistics import median
from statistics import variance


def main():
    file_name = sys.argv[1]
    data = json.load(open(os.path.expanduser(file_name), 'r'))
    results = {x: {'losses': [], 'acc': []}
               for x in ('0.75', '0.667', '0.5', '0.333', '0.25')}
    for _iter in data:
        for key, d in _iter.items():
            d.pop('losses')
            acc = d.pop('final_acc')
            results[key]['acc'].append(acc)
            results[key]['losses'].extend(d.values())
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


if __name__ == '__main__':
    main()
