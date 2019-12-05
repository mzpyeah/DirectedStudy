from bleu.bleu import Bleu
from rouge.rouge import Rouge

max_sentence = 500
ner_weight = 0.6
qt_weight = 0.2
re_weight = 0.1
delta = 0.7
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
def compute_answerability_scores(fluent, imp, ner, sw, q):
    answerability = re_weight * imp + \
                    ner_weight * ner + \
                    qt_weight * q + \
                    (1 - re_weight - ner_weight - qt_weight) * sw
    return delta * answerability + (1 - delta) * fluent

def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
    ]
    final_scores = {}
    ref = {0: [ref.strip()]}
    hypo = {0: [hypo.strip()]}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

def test_qbleu(refs, hypos):
    ngram_metric = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "ROUGE_L"]
    fluent_res, answerability_res = [], []
    ct = 0
    for ref, hypo in zip(refs, hypos):
        if ct % 100 == 0: print(ct)
        ct += 1
        ref_imp, hypo_imp = remove_stopwords_and_NER_line(ref), remove_stopwords_and_NER_line(hypo)
        ref_ner, hypo_ner = NER_line(ref), NER_line(hypo)
        ref_sw, hypo_sw = get_stopwords(ref), get_stopwords(hypo)
        ref_q, hypo_q = questiontype(ref), questiontype(hypo)
        score_fluent = score(ref, hypo)
        score_imp = score(ref_imp, hypo_imp)
        score_ner = score(ref_ner, hypo_ner)
        score_sw = score(ref_sw, hypo_sw)
        score_q = score(ref_q, hypo_q)
        score_ans = {}
        for ngram in ngram_metric:
            score_ans[ngram] = compute_answerability_scores(score_fluent[ngram], score_imp['Bleu_1'], score_ner['Bleu_1'], score_sw['Bleu_1'], score_q['Bleu_1'])
        fluent_res.append(score_fluent)
        answerability_res.append(score_ans)
    return fluent_res, answerability_res

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
    ngrams, qbleus = test_qbleu(getSentence(refs), getSentence(hypos))
    f_B1 = open("result/ngram/bleu1.txt", "w")
    f_B2 = open("result/ngram/bleu2.txt", "w")
    f_B3 = open("result/ngram/bleu3.txt", "w")
    f_B4 = open("result/ngram/bleu4.txt", "w")
    f_Q_B3 = open("result/q-ngram/Qbleu3.txt", "w")
    f_Q_B4 = open("result/q-ngram/Qbleu4.txt", "w")
    for ngram, qbleu in zip(ngrams, qbleus):
        f_B1.write(str(ngram['Bleu_1']))
        f_B1.write('\n')
        f_B2.write(str(ngram['Bleu_2']))
        f_B2.write('\n')
        f_B3.write(str(ngram['Bleu_3']))
        f_B3.write('\n')
        f_B4.write(str(ngram['Bleu_4']))
        f_B4.write('\n')
        f_Q_B3.write(str(qbleu['Bleu_3']))
        f_Q_B3.write('\n')
        f_Q_B4.write(str(qbleu['Bleu_4']))
        f_Q_B4.write('\n')
    f_B1.close()
    f_B2.close()
    f_B3.close()
    f_B4.close()
    f_Q_B3.close()
    f_Q_B4.close()
    print("Done!")