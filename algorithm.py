import os
from collections import defaultdict
from itertools import accumulate
from random import choices
from typing import List, Dict

from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')


class ProbsAlgo:
    def __init__(self, data_path: str, probs: List[float], n: int) -> None:
        self.data_path = data_path

        self.probs = probs
        assert sum(probs) == 1, 'Probabilities must sum to 1'

        self.true_labels = self.read_file(data_path)
        self.n = n

        self.preds = self.make_predictions()
        self.metrics = self.get_final_metrics()

    def read_file(self, path: str) -> List[int]:
        assert os.path.isfile(path), 'There is no such file in the directory'
        with open(path, 'r') as file:
            return [int(line) for line in file]

    def make_predictions(self) -> List[List[int]]:
        predictions = []
        classes = sorted(set(self.true_labels))

        for i in range(self.n):
            predictions.append(choices(classes, self.probs, k=len(self.true_labels)))

        assert len(predictions) == self.n
        for pred in predictions:
            assert len(pred) == len(self.true_labels)
        return predictions

    @staticmethod
    def accuracy(true_labels: List[int], predictions: List[int]) -> float:
        assert len(true_labels) == len(predictions) != 0

        return sum(pred == true_pred for pred, true_pred in zip(true_labels, predictions)) \
               / len(predictions)

    @staticmethod
    def precision(true_labels: List[int], predictions: List[int], class_number: int) -> float:
        assert len(true_labels) == len(predictions) != 0

        tp = sum(pred == true_pred == class_number\
                 for pred, true_pred in zip(predictions, true_labels))

        fp = sum(class_number == pred != true_pred\
                 for pred, true_pred in zip(predictions, true_labels))

        return tp / (tp + fp)

    @staticmethod
    def recall(true_labels: List[int], predictions: List[int], class_number: int) -> float:
        assert len(true_labels) == len(predictions) != 0

        tp = sum(pred == true_pred == class_number \
                 for pred, true_pred in zip(predictions, true_labels))

        fn = sum(class_number == true_pred != pred\
                 for pred, true_pred in zip(predictions, true_labels))

        return tp / (tp + fn)

    def cumulative_average(self, metrics: List[int]) -> List[int]:
        current_sum = 0
        for num, val in enumerate(metrics, start=1):
            current_sum += val
            metrics[num-1] = current_sum / num
        return metrics

    def get_final_metrics(self) -> Dict[str, List[float]]:

        metrics = defaultdict(list)
        for i in range(self.n):
            metrics['accuracy'].append(self.accuracy(self.true_labels,
                                                     self.preds[i]))

            metrics['precision_class_0'].append(self.precision(self.true_labels,
                                                               self.preds[i],
                                                               class_number=0))

            metrics['precision_class_1'].append(self.precision(self.true_labels,
                                                               self.preds[i],
                                                               class_number=1))

            metrics['precision_class_2'].append(self.precision(self.true_labels,
                                                               self.preds[i],
                                                               class_number=2))

            metrics['recall_class_0'].append(self.recall(self.true_labels,
                                                         self.preds[i],
                                                         class_number=0))

            metrics['recall_class_1'].append(self.recall(self.true_labels,
                                                         self.preds[i],
                                                         class_number=1))

            metrics['recall_class_2'].append(self.recall(self.true_labels,
                                                         self.preds[i],
                                                         class_number=2))
        for metric in metrics:
            metrics[metric] = self.cumulative_average(metrics[metric])

        assert len(metrics) == 7
        for metric in metrics.values():
            assert len(metric) == self.n
        return metrics

    def plot_and_save_result(self, output_path: str) -> None:
        metrics = self.get_final_metrics()
        fig, axs = plt.subplots(len(metrics), figsize=(16, 12))

        i = 0
        for metric in metrics:
            axs[i].plot(metrics[metric])
            axs[i].title.set_text(f'{metric}')
            i += 1

        plt.subplots_adjust()
        plt.savefig(rf'{output_path}\final_results.png')
