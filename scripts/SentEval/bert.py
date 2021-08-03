# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
from logzero import logger
import logzero

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# SentEval prepare and batcher
def prepare(params, samples):
    return

def batcher(params, batch):
    embeddings = []
    for sent in batch:
        ids = tokenizer.encode(' '.join(sent), add_special_tokens=True)
        ids = torch.tensor(ids).unsqueeze(0)
        tmp = encoder(ids)[1].squeeze(0).detach().cpu().numpy()
        embeddings.append(tmp)
    embeddings = np.vstack(embeddings)
    print(embeddings)
    return embeddings

model = sys.argv[1].strip()

tokenizer = AutoTokenizer.from_pretrained(model)
encoder = AutoModel.from_pretrained(model)

# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

if __name__ == "__main__":
    logzero.logfile('sent.log')
    logger.info('cuda : {}'.format(torch.cuda.is_available()))
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    #transfer_tasks = ['WordContent']
    transfer_tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)
    logger.info(sys.argv[1])
    logger.info(results)

