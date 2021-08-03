ll = [2,4,6,8,10,12]
hh = [128,256,512,768]

from transformers import AutoTokenizer, BertModel
import os

for l in ll:
  for h in hh:
    token = AutoTokenizer.from_pretrained('google/bert_uncased_L-{}_H-{}_A-{}'.format(l, h, int(h/64)))
    model = BertModel.from_pretrained('google/bert_uncased_L-{}_H-{}_A-{}'.format(l, h, int(h/64)))
    token.save_pretrained('google/bert_uncased_L-{}_H-{}_A-{}'.format(l, h, int(h/64)))
    model.save_pretrained('google/bert_uncased_L-{}_H-{}_A-{}'.format(l, h, int(h/64)))
