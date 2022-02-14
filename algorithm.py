from typing import List, Dict
import random
from random import choices
import csv
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from itertools import accumulate

class ProbsAlgo:
    def __init__(self, data_path: str, probs: List[float], n: int) -> None:
        self.data_path = data_path
        self.true_labels = self.read_file(data_path)
        self.probs = probs
        self.n = n

        self.preds = self.make_predictions()
        self.metrics = self.get_final_metrics()

    def read_file(self, path: str) -> List[int]:
        with open(path, 'r') as file:
            return [int(line.rstrip('\n')) for line in file]

    def make_predictions(self) -> List[List[int]]:
        predictions = []
        classes = list(set(self.true_labels))

        for i in range(self.n):
            predictions.append(choices(classes, self.probs, k=len(self.true_labels)))

        assert len(predictions) == self.n
        for pred in predictions:
            assert len(pred) == len(self.true_labels)
        return predictions

    @staticmethod
    def accuracy(true_labels: List[int], predictions: List[int]) -> float:
        count = 0
        for i in range(len(predictions)):
            if predictions[i] == true_labels[i]:
                count += 1
        return count / len(predictions)

    @staticmethod
    def precision(true_labels: List[int], predictions: List[int], class_number: int) -> float:
        tp, fp = 0, 0
        for i in range(len(predictions)):
            if predictions[i] == class_number and true_labels[i] == class_number:
                tp += 1
            if predictions[i] == class_number and true_labels[i] != class_number:
                fp += 1
        return tp / (tp + fp)

    @staticmethod
    def recall(true_labels: List[int], predictions: List[int], class_number: int) -> float:
        tp, fn = 0, 0
        for i in range(len(predictions)):
            if predictions[i] == class_number and true_labels[i] == class_number:
                tp += 1
            if predictions[i] != class_number and true_labels[i] == class_number:
                fn += 1
        return tp / (tp + fn)

    @staticmethod
    def cumulative_average(n: int, metrics: Dict[str, List[float]]) -> Dict[str, List[float]]:

        accuracy_cumsum = list(accumulate(metrics['accuracy']))
        precision_class_0_cumsum = list(accumulate(metrics['precision_class_0']))
        precision_class_1_cumsum = list(accumulate(metrics['precision_class_1']))
        precision_class_2_cumsum = list(accumulate(metrics['precision_class_2']))
        recall_class_0_cumsum = list(accumulate(metrics['recall_class_0']))
        recall_class_1_cumsum = list(accumulate(metrics['recall_class_1']))
        recall_class_2_cumsum = list(accumulate(metrics['recall_class_2']))

        for i in range(n):
            if i == 0:
                continue
            metrics['accuracy'][i] = accuracy_cumsum[i] / (i + 1)
            metrics['precision_class_0'][i] = precision_class_0_cumsum[i] / (i + 1)
            metrics['precision_class_1'][i] = precision_class_1_cumsum[i] / (i + 1)
            metrics['precision_class_2'][i] = precision_class_2_cumsum[i] / (i + 1)
            metrics['recall_class_0'][i] = recall_class_0_cumsum[i] / (i + 1)
            metrics['recall_class_1'][i] = recall_class_1_cumsum[i] / (i + 1)
            metrics['recall_class_2'][i] = recall_class_2_cumsum[i] / (i + 1)

        # # for i in range(n):
        #     if i == 0:
        #         accuracy = metrics['accuracy'][i]
        #         precision_class_0 = metrics['precision_class_0'][i]
        #         precision_class_1 = metrics['precision_class_1'][i]
        #         precision_class_2 = metrics['precision_class_2'][i]
        #         recall_class_0 = metrics['recall_class_0'][i]
        #         recall_class_1 = metrics['recall_class_1'][i]
        #         recall_class_2 = metrics['recall_class_2'][i]
        #         continue

            # accuracy += metrics['accuracy'][i]
            # metrics['accuracy'][i] = accuracy / (i + 1)
            #
            # precision_class_0 += metrics['precision_class_0'][i]
            # metrics['precision_class_0'][i] = precision_class_0 / (i + 1)
            #
            # precision_class_1 += metrics['precision_class_1'][i]
            # metrics['precision_class_1'][i] = precision_class_1 / (i + 1)
            #
            # precision_class_2 += metrics['precision_class_2'][i]
            # metrics['precision_class_2'][i] = precision_class_2 / (i + 1)
            #
            # recall_class_0 += metrics['recall_class_0'][i]
            # metrics['recall_class_0'][i] = recall_class_0 / (i + 1)
            #
            # recall_class_1 += metrics['recall_class_1'][i]
            # metrics['recall_class_1'][i] = recall_class_1 / (i + 1)
            #
            # recall_class_2 += metrics['recall_class_2'][i]
            # metrics['recall_class_2'][i] = recall_class_2 / (i + 1)

        return metrics

    def get_final_metrics(self) -> Dict[str, List[float]]:

        metrics = {
            'accuracy': [],
            'precision_class_0': [],
            'precision_class_1': [],
            'precision_class_2': [],
            'recall_class_0': [],
            'recall_class_1': [],
            'recall_class_2': []
        }
        for i in range(self.n):
            metrics['accuracy'].append(ProbsAlgo.accuracy(self.true_labels,
                                                          self.preds[i]))

            metrics['precision_class_0'].append(ProbsAlgo.precision(self.true_labels,
                                                                    self.preds[i],
                                                                    class_number=0))

            metrics['precision_class_1'].append(ProbsAlgo.precision(self.true_labels,
                                                                    self.preds[i],
                                                                    class_number=1))

            metrics['precision_class_2'].append(ProbsAlgo.precision(self.true_labels,
                                                                    self.preds[i],
                                                                    class_number=2))

            metrics['recall_class_0'].append(ProbsAlgo.recall(self.true_labels,
                                                              self.preds[i],
                                                              class_number=0))

            metrics['recall_class_1'].append(ProbsAlgo.recall(self.true_labels,
                                                              self.preds[i],
                                                              class_number=1))

            metrics['recall_class_2'].append(ProbsAlgo.recall(self.true_labels,
                                                              self.preds[i],
                                                              class_number=2))
        metrics = ProbsAlgo.cumulative_average(self.n, metrics)

        assert len(metrics) == 7
        for metric in metrics.values():
            assert len(metric) == self.n
        return metrics

    def plot_and_save_result(self, output_path: str) -> None:
        metrics = self.get_final_metrics()
        fig, axs = plt.subplots(len(metrics), figsize=(16, 14))

        i = 0
        for metric in metrics:
            axs[i].plot(metrics[metric])
            axs[i].title.set_text(f'{metric}')
            i += 1

        plt.savefig(rf'{output_path}\final_results.png')