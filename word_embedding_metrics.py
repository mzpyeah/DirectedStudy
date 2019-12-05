import numpy as np
from scipy.spatial.distance import cosine
from bert_serving.client import BertClient

max_sentence = 500
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]'  # This has a vocab id, which is used at the end of untruncated target sequences

class Vocab(object):
    """Vocabulary class for mapping between words and ids (integers)"""

    def __init__(self, vocab_file, max_size):
        """Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.

        Args:
          vocab_file: path to the vocab file, which is assumed to contain "<word> <frequency>" on each line, sorted with most frequent word first. This code doesn't actually use the frequencies, though.
          max_size: integer. The maximum size of the resulting Vocabulary."""
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # keeps track of total number of words in the Vocab
        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
        for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file + '.from', 'r') as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 1:
                    print('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
                    continue
                w = pieces[0]
                if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception(
                        '<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count))
                    break
        print("Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count - 1]))

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count

    def write_metadata(self, fpath):
        """Writes metadata file for Tensorboard word embedding visualizer as described here:
          https://www.tensorflow.org/get_started/embedding_viz
        Args:
          fpath: place to write the metadata file
        """
        print("Writing word embedding metadata file to %s..." % (fpath))
        with open(fpath, "w") as f:
            fieldnames = ['word']
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            for i in xrange(self.size()):
                writer.writerow({"word": self._id_to_word[i]})

class Embedding():
    def __init__(self):
        # self.embed = np.loadtxt('data/embedding/bert_embed.txt') # 19616 * 768
        # self.vocab = Vocab('data/embedding/vocab20000', 20000)
        self.bc = BertClient()

    # def getEmbedding(self, word):
        # return self.embed[self.vocab.word2id(self.clean(word))]
    def getEmbedding(self, sentences):
        return self.bc.encode(sentences)

def greedy_matching(ref, hypo):
    def _greedy_matching(r_em, h_em):
        res = 0
        for h in h_em:
            mini = float('inf')
            for r in r_em:
                mini = min(mini, cosine(h, r))
            res += (1 - mini)
        return res * 1.0 / len(h_em)
    return (_greedy_matching(ref, hypo) + _greedy_matching(hypo, ref)) * 1.0 / 2

def embedding_average(ref, hypo):
    ref_avg = np.sum(ref, axis = 0) * 1.0 / ref.shape[1]
    hypo_avg = np.sum(hypo, axis = 0) * 1.0 / hypo.shape[1]
    return 1 - cosine(ref_avg, hypo_avg)

def vector_extrema(ref, hypo):
    def _extrema(vector):
        res = []
        for col in range(vector.shape[1]):
            v = vector[:, col]
            mini, maxi = min(v), max(v)
            if maxi > abs(mini): res.append(maxi)
            else: res.append(mini)
        return np.array(res)
    return 1 - cosine(_extrema(ref), _extrema(hypo))

def test_embedding_metrics(reference, hypothesis):
    embed = Embedding()
    ref_ems = embed.getEmbedding(reference)
    hypo_ems = embed.getEmbedding(hypothesis)
    # greedy, average, extrema = [], [], []
    raw = []
    ct = 0
    # for ref, hypo in zip(reference, hypothesis):
    for ref, hypo in zip(ref_ems, hypo_ems):
        if ct % 100 == 0: print(ct)
        raw.append(1 - cosine(ref, hypo))
        # ref = ref.split()
        # hypo = hypo.split()
        # ref_em = np.array([embed.getEmbedding(word) for word in ref])
        # hypo_em = np.array([embed.getEmbedding(word) for word in hypo])
        # greedy.append(greedy_matching(ref_em, hypo_em))
        # average.append(embedding_average(ref_em, hypo_em))
        # extrema.append(vector_extrema(ref_em, hypo_em))
        ct += 1
    # return greedy, average, extrema
    return raw

def getSentence(lis):
    with open('data/list.txt') as f:
        idx = [int(i.strip()) for i in f.readlines()]
    assert(len(idx) == max_sentence)
    return [lis[i] for i in idx]

if __name__ == "__main__":
    with open('data/truth.txt') as f:
        refs = f.readlines()
    with open('data/generated.txt') as f:
        hypos = f.readlines()
    # greedy, average, extrema = test_embedding_metrics(getSentence(refs), getSentence(hypos))
    raw = test_embedding_metrics(getSentence(refs), getSentence(hypos))
    # with open('result/embedding/greedy.txt', 'w') as f:
    #     for i in greedy:
    #         f.write(str(i))
    #         f.write('\n')
    # with open('result/embedding/average.txt', 'w') as f:
    #     for i in average:
    #         f.write(str(i))
    #         f.write('\n')
    # with open('result/embedding/extrema.txt', 'w') as f:
    #     for i in extrema:
    #         f.write(str(i))
    #         f.write('\n')
    with open('result/embedding/raw.txt', 'w') as f:
        for i in raw:
            f.write(str(i))
            f.write('\n')
    print("Done!")