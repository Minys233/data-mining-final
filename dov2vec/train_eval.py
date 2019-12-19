import gensim
import logging
from scipy.special import softmax
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
import jieba
import argparse

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def readtokenize(fname, tokensonly=False):
    if isinstance(fname, list):
        for i in fname:
            yield readtokenize(i, tokensonly=tokensonly)
    with open(fname) as fin:
        for line in fin:
            if len(line.split('\t')) != 2:
                print(fname)
                print(line)
                exit(0)
            text, label = line.split('\t')
            tokens = list(jieba.cut(text, cut_all=True))
            if tokensonly:
                yield tokens, int(label)
            else:
                yield gensim.models.doc2vec.TaggedDocument(tokens, [int(label)])

def KNNclassify(model, corpus, label):
    predict, prob = [], []
    for seq in corpus:
        seq_vec = model.infer_vector(seq)
        raw = model.docvecs.most_similar([seq_vec])
        s = softmax([i[1] for i in raw])
        if s[0] > s[1]:
            predict.append(0)
            prob.append(s[0])
        else:
            predict.append(1)
            prob.append(s[1])
    predict, prob, label = np.array(predict), np.array(prob), np.array(label)
    evaluate(predict, prob, label)
    return np.array(predict), np.array(prob), np.array(label)


def vec_for_learning(model, sents):
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return np.array(targets), np.array(regressors)


def SVCclassify(model, corpus_train, corpus_test):
    svc = SVC(gamma='scale', probability=True)
    y_train, X_train = vec_for_learning(model, corpus_train)
    y_test, X_test = vec_for_learning(model, corpus_test)
    svc.fit(X_train, y_train)
    raw_result = svc.predict_proba(X_test)
    predic = np.argmax(raw_result, axis=1)
    prob = raw_result[range(len(raw_result)), predic]
    evaluate(predic, prob, y_test)


def evaluate(predict, prob, label):
    acc = metrics.accuracy_score(label, predict)
    roc = metrics.roc_auc_score(label, prob)
    report = metrics.classification_report(label, predict, target_names=['True', 'Fake'], digits=4)
    confusion = metrics.confusion_matrix(label, predict)
    msg = f'Acc: {acc:>6.2%},  ROC AUC: {roc:>6.3}'
    print(msg)
    print("Precision, Recall and F1-Score...")
    print(report)
    print("Confusion Matrix...")
    print(confusion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'load'])
    parser.add_argument('--model', default="None")
    args = parser.parse_args()
    train_corpus = list(readtokenize('../data/raw_train.txt'))
    test_corpus = list(readtokenize(['../data/raw_test.txt', '../data/raw_dev.txt']))
    if args.action == 'train':
        model = gensim.models.doc2vec.Doc2Vec(train_corpus, dm=0, vector_size=300, negative=5, hs=0, min_count=2,
                                              sample=0, workers=16)
        model.save('doc2vec.model')
    else:
        model = gensim.models.doc2vec.Doc2Vec.load('doc2vec.model')

    SVCclassify(model, train_corpus, test_corpus)


### Acc: 90.72%,  ROC AUC:  0.449
# Precision, Recall and F1-Score...
#               precision    recall  f1-score   support
#
#         True     0.9033    0.9140    0.9086      1941
#         Fake     0.9114    0.9004    0.9058      1907
#
#     accuracy                         0.9072      3848
#    macro avg     0.9073    0.9072    0.9072      3848
# weighted avg     0.9073    0.9072    0.9072      3848
#
# Confusion Matrix...
# [[1774  167]
#  [ 190 1717]]