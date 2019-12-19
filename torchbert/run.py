# coding: UTF-8
import time
import torch
import numpy as np
from torchbert.train_eval import train, init_network
from importlib import import_module
import argparse
from torchbert.utils import build_dataset, build_iterator, get_time_dif, DatasetIterater

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'Data'  # 数据集

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, (sentence_id, test_data) = build_dataset(config)
    print(len(dev_data), len(test_data))
    assert len(sentence_id) == len(test_data)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter, sentence_id)


### bert
# Test Loss: 0.083,  Test Acc: 98.40%,  Test ROC AUC:  0.767
# Precision, Recall and F1-Score...
#               precision    recall  f1-score   support
#
#         True     0.9723    0.9967    0.9843      1797
#         Fake     0.9965    0.9712    0.9837      1769
#
#     accuracy                         0.9840      3566
#    macro avg     0.9844    0.9839    0.9840      3566
# weighted avg     0.9843    0.9840    0.9840      3566
#
# Confusion Matrix...
# [[1791    6]
#  [  51 1718]]


### bert_CNN
# Test Loss: 0.077,  Test Acc: 98.57%,  Test ROC AUC:  0.788
# Precision, Recall and F1-Score...
#               precision    recall  f1-score   support
#
#         True     0.9802    0.9917    0.9859      1797
#         Fake     0.9914    0.9796    0.9855      1769
#
#     accuracy                         0.9857      3566
#    macro avg     0.9858    0.9857    0.9857      3566
# weighted avg     0.9858    0.9857    0.9857      3566
#
# Confusion Matrix...
# [[1782   15]
#  [  36 1733]]
# Time usage: 0:00:51

