# probing_sfqa
Codes for reproducing the results shown in "Probing simple factoid question answering based on linguistic knowledge", which will be appeared in Journal of Natural Language Processing at 2021 Dec.

All codes and data will be prepared by just running `setup.sh`.

## dependencies
python>3.6, virtualenv.

## how to use
### to prepare BertQA systems and datasets
```
$ sh setup.sh
```
### run BertQA
```
$ sh run_bertqa.sh
```
It assumed that you already download 24 BERT models, three datasets, and BuboQA modules by `setup.sh`.

### run PLSPM analysis
Check `plspm` for the R script and raw data for our experiments.

### evaluate the result of BertQA with reachability accuracy
Check `scripts/reach_acc.py`. It assumed that you already download FB2M data by `setup.sh`.

## datasets
Free917, FreebaseQA, SimpleQuestions, and WebQSP will be prepared by `setup.sh`.

Filtered datasets for simple question answering, F917, FBQ, SQ, and WQ, can be found in `simple-qa-analysis/datasets`.

## reference
- BuboQA
```
@inproceedings{mohammed-etal-2018-strong,
    title = "Strong Baselines for Simple Question Answering over Knowledge Graphs with and without Neural Networks",
    author = "Mohammed, Salman  and
      Shi, Peng  and
      Lin, Jimmy",
    booktitle = "Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)",
    month = jun,
    year = "2018",
    address = "New Orleans, Louisiana",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N18-2047",
    doi = "10.18653/v1/N18-2047",
    pages = "291--296"
}
```
- BertQA
```
@article{DBLP:journals/corr/abs-2001-11985,
  author    = {Denis Lukovnikov and
               Asja Fischer and
               Jens Lehmann},
  title     = {Pretrained Transformers for Simple Question Answering over Knowledge
               Graphs},
  journal   = {CoRR},
  volume    = {abs/2001.11985},
  year      = {2020},
  url       = {https://arxiv.org/abs/2001.11985},
  archivePrefix = {arXiv},
  eprint    = {2001.11985},
  timestamp = {Mon, 03 Feb 2020 11:21:05 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2001-11985.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## acknowledgment
These works are based on results obtained from projects JPNP20006 and JPNP15009, commissioned by the New Energy and Industrial Technology Development Organization (NEDO), and also with the support of RIKEN–AIST Joint Research Fund (Feasibility study). 

