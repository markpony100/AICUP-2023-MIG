from sentence_transformers import SentenceTransformer, util
import wikipedia
from collections import Counter
import opencc
import re
from utils.CKIP import CKIP
import logging

CONVERTER_T2S = opencc.OpenCC("t2s.json")
CONVERTER_S2T = opencc.OpenCC("s2t.json")


def do_st_corrections(text: str) -> str:
    simplified = CONVERTER_T2S.convert(text)

    return CONVERTER_S2T.convert(simplified)


class WikiSearch:

    def __init__(self, wikiData) -> None:
        wikipedia.set_lang("zh")
        self.wikiData = wikiData
        self.ckip = CKIP()
        self.dataBase = list(wikiData.id)
        self.text = list(wikiData.text)
        self.embedder = SentenceTransformer(
            'hfl/chinese-roberta-wwm-ext-large').to('cuda')

    def make_dict(self, x):
        result = {}
        sentences = re.split(r"\n(?=[0-9])", x)
        for sent in sentences:
            splitted = sent.split("\t")
            if len(splitted) < 2:
                # Avoid empty articles
                return result
            result[splitted[0]] = splitted[1]
        return result

    def sort_dic(self, answerBag):
        return sorted(answerBag.items(), key=lambda x: x[1], reverse=True)

    def wikinet_search(self, question):
        answerBag = Counter()
        for word in question:
            if len(word) > 45:
                word = word[:45]
            ref = wikipedia.search(word, results=5, suggestion=False)
            # if mapper.title_to_id(word) != None:
            for page in ref:
                page = do_st_corrections(page).replace(' (', '_(')
                if page in self.dataBase:
                    answerBag[page] += 1
        return answerBag

    def compare_similarity(self, text_id, claim_array):
        count = 0
        text = self.wikiData.text[self.dataBase.index(text_id)]
        for i in claim_array:
            if i in str(text):
                count += 1
        return count

    def embedd_similarity(self, claim, sentences):
        answer = []
        ask_emb = self.embedder.encode(claim)
        sent_emb = self.embedder.encode(sentences)
        score = util.semantic_search(ask_emb, sent_emb, top_k=20)
        score.sort(reverse=True,)
        for i in score[0]:
            answer.append(sentences[i['corpus_id']])
        return answer

    def clean_dict(self, answer_dict, changeList):
        count = 0
        for i in answer_dict.keys():
            count += answer_dict[i]
        for i in changeList:
            if answer_dict[i[0]] < (count/len(answer_dict)):
                del answer_dict[i[0]]
        return answer_dict

    def update_dict(self, answerBag, claim, claim_array):
        answer = self.embedd_similarity(claim, list(answerBag.keys()))[:5]
        for i in answerBag.keys():
            answerBag[i] += self.compare_similarity(i, claim_array)
            try:
                answer.index(i)
                if answer.index(i) < 5:
                    answerBag[i] += 5 - answer.index(i)

            except:
                continue
        return answerBag

    def wikiSearch(self, question, claim):
        claim_array = []
        for i in question:
            claim_array += self.ckip.nerExtract(i)
            claim_array += self.ckip.posExtract(i)
        claim_array = set(claim_array)
        answerBag = Counter()
        answerBag.update(self.wikinet_search(question))
        answerBag = self.update_dict(answerBag, claim, claim_array)
        sorted_answerBag = self.sort_dic(answerBag)
        answerBag = self.clean_dict(answerBag, sorted_answerBag[6:])
        logging.info(sorted_answerBag)
        sec_layer = []
        for i in sorted_answerBag:
            page = self.wikiData.lines[self.dataBase.index(i[0])]
            sec_layer += [i[0]+':' +
                          sent for sent in self.make_dict(str(page)).values()]
        sec_layer = self.embedd_similarity(claim, sec_layer)
        logging.info(sec_layer)
        secAnswerBag = self.wikinet_search(sec_layer)
        secAnswerBag = self.update_dict(secAnswerBag, claim, claim_array)
        sortedSecAnswerBag = self.sort_dic(secAnswerBag)
        secAnswerBag = self.clean_dict(secAnswerBag, sortedSecAnswerBag[6:])
        for i in secAnswerBag.keys():
            if i in answerBag:
                answerBag[i] += secAnswerBag[i]
            else:
                answerBag[i] = secAnswerBag[i]
        del secAnswerBag
        del sortedSecAnswerBag
        sorted_answerBag = self.sort_dic(answerBag)
        return [item[0] for item in sorted_answerBag][:10]
