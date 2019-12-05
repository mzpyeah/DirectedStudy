from bleu.bleu import Bleu
from rouge.rouge import Rouge
from scipy.spatial.distance import cosine
from scipy.stats.stats import pearsonr
from bert_serving.client import BertClient
import json

max_sentence = 500
ner_weight = 0.3
qt_weight = 0.6
re_weight = 0.1
delta = 0.3
savename = "result/new/bert_best.txt"
stop_words = {"did", "have", "ourselves", "hers", "between", "yourself",
              "but", "again", "there", "about", "once", "during", "out", "very",
              "having", "with", "they", "own", "an", "be", "some", "for", "do", "its",
              "yours", "such", "into", "of", "most", "itself", "other", "off", "is", "s",
              "am", "or", "as", "from", "him", "each", "the", "themselves", "until", "below",
              "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were",
              "her", "more", "himself", "this", "down", "should", "our", "their", "while",
              "above", "both", "up", "to", "ours", "had", "she", "all", "no", "at", "any",
              "before", "them", "same", "and", "been", "have", "in", "will", "on", "does",
              "yourselves", "then", "that", "because", "over", "so", "can", "not", "now", "under",
              "he", "you", "herself", "has", "just", "too", "only", "myself", "those", "i", "after",
              "few", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "further",
              "was", "here", "than"}
question_words_global = {'What', 'Which', 'Why', 'Who', 'Whom', 'Whose', 'Where', 'When', 'How'}
question_words_global.update([w.lower() for w in question_words_global])

# remove question words & stop words & NER words
def remove_stopwords_and_NER_line(question):
    question = question.strip().split()
    question_words = question_words_global
    temp_words = []
    for word in question_words:
        for i, w in enumerate(question):
            if w == word:
                temp_words.append(w)
                # If the question type is 'what' or 'which' the following word is generally associated with
                # with the answer type. Thus it is important that it is considered a part of the question.
                if i + 1 < len(question) and (w.lower() == "what" or w.lower() == "which"):
                    temp_words.append(question[i+1])
    ner_words = [item for item in question if item not in temp_words] # remove question words
    temp_words = []
    for i in ner_words:
        if i[0].isupper() == False: # remove NER words
            if i not in stop_words:
                temp_words.append(i) # remove stop words
    return " ".join(temp_words)

# remove question words and get NER words
def NER_line(question):
    q_types = question_words_global
    question_words = question.strip().split()
    if question_words[0].lower() in q_types:
        question_words = question_words[1:]
    temp_words = []
    for i in question_words:
        if i[0].isupper():
            temp_words.append(i)
    return " ".join(temp_words)

# get stop words
def get_stopwords(question):
    question_words = question.strip().split()
    temp_words = []
    for i in question_words:
        if i.lower() in stop_words:
            temp_words.append(i.lower())
    return " ".join(temp_words)

# get question words
def questiontype(question):
    types = question_words_global
    question = question.strip().split()
    temp_words = []
    for word in types:
        for i, w in enumerate(question):
            if w == word:
                temp_words.append(w)
                if i+1 < len(question) and (w.lower() == "what" or w.lower() == "which"):
                    temp_words.append(question[i+1])
    return " ".join(temp_words)

# main Q-Bleu function
def compute_answerability_scores(fluent, imp, ner, sw, q, re_weight=re_weight, ner_weight=ner_weight, qt_weight=qt_weight, delta=delta):
    answerability = re_weight * imp + \
                    ner_weight * ner + \
                    qt_weight * q + \
                    (1 - re_weight - ner_weight - qt_weight) * sw
    return delta * answerability + (1 - delta) * fluent

class Embedding():
    def __init__(self):
        self.bc = BertClient(ip = "52.200.146.41")
    def getEmbedding(self, sentences):
        return self.bc.encode(sentences)

def getHuman():
    with open("result/human_result.txt") as f:
        lis = f.readlines()
        return [float(i.strip().split()[1]) for i in lis]

def score(ref, hypo):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
    ]
    embed = Embedding()
    ref = ref.strip()
    hypo = hypo.strip()
    if len(ref) == 0: ref = "<unk>"
    if len(hypo) == 0: hypo = "<unk>"
    ref_em = embed.getEmbedding([ref])
    hypo_em = embed.getEmbedding([hypo])
    final_scores = {}
    ref = {0: [ref]}
    hypo = {0: [hypo]}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else: final_scores[method] = score
    final_scores["bert"] = 1 - cosine(ref_em, hypo_em)
    return final_scores

def regular(data):
    mini, maxi = min(data), max(data)
    if maxi - mini != 0:
        data_ = [(i - mini) / (maxi - mini) for i in data]
    else:
        data_ = data
    return data_

def test_qbleu(refs, hypos):
    ngram_metric = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "ROUGE_L"]
    answerability_res = []
    ct = 0
    dic = {}
    # for ref, hypo in zip(refs, hypos):
    #     if ct % 100 == 0: print(ct)
    #     ct += 1
    #     ref_imp, hypo_imp = remove_stopwords_and_NER_line(ref), remove_stopwords_and_NER_line(hypo)
    #     ref_ner, hypo_ner = NER_line(ref), NER_line(hypo)
    #     ref_sw, hypo_sw = get_stopwords(ref), get_stopwords(hypo)
    #     ref_q, hypo_q = questiontype(ref), questiontype(hypo)
    #     score_fluent = score(ref, hypo)
    #     score_imp = score(ref_imp, hypo_imp)
    #     score_ner = score(ref_ner, hypo_ner)
    #     score_sw = score(ref_sw, hypo_sw)
    #     score_q = score(ref_q, hypo_q)
    #     dic[ct - 1] = [score_fluent, score_imp, score_ner, score_sw, score_q]
    # with open('tmp_data.json', 'w') as f:
    #     json.dump(dic, f)
    dic = json.load(open('tmp_data.json'))
    for i in range(5):
        bert = regular([dic[str(k)][i]["bert"] for k in range(max_sentence)])
        for k in range(max_sentence):
            dic[str(k)][i]["bert"] = bert[k]
    # scope = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # maxi = 0
    # max_rec = []
    # for delta in scope:
    #     for re_weight in scope:
    #         for ner_weight in scope:
    #             if ner_weight + re_weight > 1: break
    #             for qt_weight in scope:
    #                 if ner_weight + re_weight + qt_weight > 1: break
    #                 res = []
    #                 print("Now: delta=%f, re=%f, ner=%f, qt=%f" % (delta, re_weight, ner_weight, qt_weight))
    for i in range(max_sentence):
        idx = str(i)
        score_ans = {}
        # for ngram in ngram_metric:
        #     score_ans[ngram] = compute_answerability_scores(dic[idx][0][ngram], 
        #                                                     dic[idx][1][ngram], 
        #                                                     dic[idx][2][ngram], 
        #                                                     dic[idx][3][ngram], 
        #                                                     dic[idx][4][ngram])
        #     score_ans[ngram + "_ner_bert"] = compute_answerability_scores(dic[idx][0][ngram], 
        #                                                     dic[idx][1]["Bleu_1"], 
        #                                                     dic[idx][2]["bert"], 
        #                                                     dic[idx][3]["Bleu_1"], 
        #                                                     dic[idx][4]["Bleu_1"])
        score_ans["bert"] = compute_answerability_scores(dic[idx][0]["bert"], 
                                                        dic[idx][1]["bert"], 
                                                        dic[idx][2]["bert"], 
                                                        dic[idx][3]["bert"], 
                                                        dic[idx][4]["bert"])
                                                        # re_weight, ner_weight, qt_weight, delta)
        # res.append(score_ans["bert"])
        # r = pearsonr(res, getHuman())[0]
        # if r >= maxi:
        #     maxi = r
        #     max_rec = [re_weight, ner_weight, qt_weight, delta]
    # print(maxi, max_rec)
        answerability_res.append(score_ans)
    return answerability_res

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
    # test_qbleu(getSentence(refs), getSentence(hypos))
    qbleus = test_qbleu(getSentence(refs), getSentence(hypos))
    f_Q_bert = open(savename, "w")
    for qbleu in qbleus:
        f_Q_bert.write(str(qbleu['bert']))
        f_Q_bert.write('\n')
    f_Q_bert.close()
    print("Done!")