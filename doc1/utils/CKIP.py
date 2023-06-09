from transformers import pipeline
from transformers import (
    BertTokenizerFast,
    AutoModelForTokenClassification,
)


class CKIP:
    def __init__(self) -> None:
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    # AutoModelForTokenClassification.from_pretrained('ckiplab/albert-base-chinese-ner').half().cuda()
        nerModel = AutoModelForTokenClassification.from_pretrained(
            'ckiplab/albert-base-chinese-ner')
    # nerDriver = pipeline('ner', model=nerModel, tokenizer=tokenizer, device=0)
        self.nerDriver = pipeline('ner', model=nerModel, tokenizer=tokenizer)
        posModel = AutoModelForTokenClassification.from_pretrained(
            'ckiplab/albert-base-chinese-pos')
        self.posDriver = pipeline('ner', model=posModel, tokenizer=tokenizer)
        wsModel = AutoModelForTokenClassification.from_pretrained(
            'ckiplab/albert-base-chinese-ws')
        self.wsDriver = pipeline('ner', model=wsModel, tokenizer=tokenizer)

    def nerExtract(self, question):
        questionNer = self.nerDriver(question)
        searchQueue = []
        name = ''
        for entity in questionNer:
            if entity['entity'][2:] not in ['CARDINAL', 'DATE', 'EVENT', 'MONEY', 'ORDINAL', 'PERCENT', 'QUANTITY', 'TIME']:
                if entity['entity'][0] == 'B':
                    name = ''
                    name += entity['word']
                elif entity['entity'][0] == 'I':
                    if len(name) == 0 and len(searchQueue) > 0:
                        name = searchQueue[-1][0]
                    name += entity['word']
                elif entity['entity'][0] == 'E':
                    name += entity['word']
                    searchQueue.append(name)
                    name = ' '
        return searchQueue

    def posExtract(self, question):
        sentence_ws = self.wsDriver(question)
        question_ws = []
        singleWord = ''
        for singleWS in sentence_ws:
            if singleWS['entity'] == 'B':
                if singleWord != '':
                    question_ws.append(singleWord)
                    singleWord = ''
            singleWord += singleWS['word']
        question_ws.append(singleWord)
        sentence_pos = self.posDriver(question_ws)
        question_pos = [wordpos[0]['entity'] for wordpos in sentence_pos]
        # print(question_pos)
        assert len(question_ws) == len(question_pos)
        res = []
        for word_ws, word_pos in zip(question_ws, question_pos):
            if word_pos in ['Na', 'Nb', 'Nc', 'Nv']:
                res.append(word_ws)
        return set(res)
