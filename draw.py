from matplotlib import pyplot as plt
from scipy.stats.stats import pearsonr
import numpy as np

max_sentence = 500

def least_squares(x, y):
    x = np.array(x)
    y = np.array(y)
    x_ = x.mean()
    y_ = y.mean()
    m = np.zeros(1)
    n = np.zeros(1)
    for i in np.arange(max_sentence):
        m += (x[i] - x_) * (y[i] - y_)
        n += np.square(x[i] - x_)
    k = m / n
    b = y_ - k * x_
    return k, b

def draw(x, y, xlabel):
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel("human")
    k, b = least_squares(x, y)
    r = pearsonr(x, y)[0]
    y = k * x + b
    plt.plot(x, y, 'r-', lw = 2, markersize = 6)
    plt.title("%s: r = %f" % (xlabel, r))
    plt.autoscale(tight = True)
    plt.grid()
    plt.savefig('imgs/%s.png' % xlabel)
    plt.clf()

def getHuman():
    with open("result/human_result.txt") as f:
        lis = f.readlines()
        return [float(i.strip().split()[1]) for i in lis]

def getScore(filename):
    with open(filename) as f:
        lis = f.readlines()
        return [float(i.strip()) for i in lis]

if __name__ == "__main__":
    test_lis = {
        "Bleu1" : "result/ngram/bleu1.txt",
        "Bleu2" : "result/ngram/bleu2.txt",
        "Bleu3" : "result/ngram/bleu3.txt",
        "Bleu4" : "result/ngram/bleu4.txt",
        "Embedding Average" : "result/embedding/average.txt",
        "Vector Extrema" : "result/embedding/extrema.txt",
        "Greedy Match" : "result/embedding/greedy.txt",
        "BERT" : "result/embedding/raw.txt",
        "Q-Bleu3" : "result/q-ngram/Qbleu3.txt",
        "Q-Bleu4" : "result/q-ngram/Qbleu4.txt",
        "Q-BERT-original" : "result/new/bert_original.txt",
        "Q-BERT-best" : "result/new/bert_best.txt",
    }

    for test in test_lis:
        draw(getScore(test_lis[test]), getHuman(), test)