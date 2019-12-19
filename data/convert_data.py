from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
import random


def output(raw_text, labels, fname, length=30, step=5):
    text, label = [], []
    for raw, l in zip(raw_text, labels):
        raw = raw.strip()
        raw = raw.replace('\t', ' ')
        raw = raw.replace('\n', ' ')
        raw = raw.replace('\r', ' ')
        for i in range((len(raw)-length) // step):
            text.append(raw[i*step:i*step+length])
            label.append(l)
    tmp = list(zip(text, label))
    random.shuffle(tmp)
    text, label = list(zip(*tmp))
    with open(fname, 'w') as fout:
        for t, l in zip(text, label):
            print(f"{t}\t{l}", file=fout)


def output_test(raw_text, labels, fname, length=30, step=5):
    text, label, index = [], [], []
    for idx, (raw, l) in enumerate(zip(raw_text, labels)):
        raw = raw.strip()
        raw = raw.replace('\t', ' ')
        raw = raw.replace('\n', ' ')
        raw = raw.replace('\r', ' ')
        for i in range((len(raw) - length) // step):
            text.append(raw[i * step:i * step + length])
            label.append(l)
            index.append(idx)
    tmp = list(zip(index, text, label))
    random.shuffle(tmp)
    index, text, label = list(zip(*tmp))
    with open(fname, 'w') as fout:
        for i, t, l in zip(index, text, label):
            print(f"{i}\t{t}\t{l}", file=fout)


def rawdata(raw_text, labels, fname):
    with open(fname, 'w') as fout:
        for raw, l in zip(raw_text, labels):
            raw = raw.strip()
            raw = raw.replace('\t', ' ')
            raw = raw.replace('\n', ' ')
            raw = raw.replace('\r', ' ')
            print(f"{raw}\t{l}", file=fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", type=str)
    parser.add_argument('--train', type=float, default=0.8)
    args = parser.parse_args()

    df = pd.read_csv(args.datafile)
    portion = 1-args.train
    text, label = df['text'].tolist(), df['label'].tolist()
    X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=portion, random_state=20191219)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=20191219)
    print(f"Total entries: {len(df)}")
    print(f"Train data size: {len(X_train)}")
    print(f"Validation data size: {len(X_valid)}")
    print(f"Test data size: {len(X_test)}")
    output(X_train, y_train, fname='../torchbert/Data/data/train.txt')
    output(X_valid, y_valid, fname='../torchbert/Data/data/dev.txt')
    output_test(X_test, y_test, fname='../torchbert/Data/data/test.txt')
    with open('../torchbert/Data/data/class.txt', 'w') as fout:
        print("True", file=fout)
        print("Fake", file=fout)

    # write raw data
    rawdata(X_train, y_train, 'raw_train.txt')
    rawdata(X_valid, y_valid, 'raw_dev.txt')
    rawdata(X_test, y_test, 'raw_test.txt')

