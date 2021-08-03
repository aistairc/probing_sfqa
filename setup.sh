# We assume that you have python3>=3.5

# set up virtualenv
virtualenv --python=python3 probing
. probing/bin/activate
pip install -U pip
pip install -r BertQA/req.txt

# download Google BERT models
python scripts/download_google_model.py

# set up dataset
git clone https://github.com/aistairc/simple-qa-analysis.git

# set up SentEval
git clone https://github.com/facebookresearch/SentEval.git
cp scripts/SentEval/* SentEval/examples/

# set up BuboQA
git clone https://github.com/castorini/BuboQA.git
cp -R -r simple-qa-analysis/diff/BuboQA/ ./BuboQA/
cd BuboQA
python -c "import nltk;nltk.download('treebank');nltk.download('stopwords')"
sh setup.sh
cd ..

deactivate

