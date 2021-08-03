import numpy as np
import os, csv
from argparse import ArgumentParser
from transformers import BertTokenizer, BertForTokenClassification, BertForSequenceClassification, AutoConfig, AutoTokenizer, get_cosine_schedule_with_warmup, AdamW
import torch
from torch.utils.data import Dataset, DataLoader
import torchtext
from logzero import logger

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, logger.info(s a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def flat_accuracy(preds, labels):
    pred_flat = preds.flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def train_rp(args, train_iter, dev_iter, model):

    if args.finetune:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"] #['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=len(train_iter) * args.batch_size * 0.05, num_training_steps=len(train_iter)* args.batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True, delta=1e-5)
    save_path = os.path.join(args.save_path, args.model)
    os.makedirs(save_path, exist_ok=True)
    snapshot_path = os.path.join(save_path, args.type + '_best_model.pt')

    for _ in range(args.epochs):
        # TRAIN loop
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_iter):
            # add batch to gpu
            b_input_ids, b_labels, _ = batch
            b_input_ids = b_input_ids.to(device)
            b_labels = b_labels.to(device)
            outputs = model(b_input_ids, labels=b_labels)
            loss = outputs.loss
            scores = outputs.logits

            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        logger.info("Train loss: {}".format(tr_loss/nb_tr_steps))
        # VALIDATION on validation set
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels = [], []
        for batch in dev_iter:
            b_input_ids, b_labels, _ = batch
            b_input_ids = b_input_ids.to(device)
            b_labels = b_labels.to(device)

            with torch.no_grad():
                outputs = model(b_input_ids, labels=b_labels)
                tmp_eval_loss = outputs.loss
                logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            predictions.extend([list(np.argmax(logits, axis=1))])
            true_labels.append(label_ids)

            tmp_eval_accuracy = flat_accuracy(np.argmax(logits, axis=1), label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss/nb_eval_steps
        logger.info("Validation loss: {}".format(eval_loss))
        logger.info("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

        early_stopping(eval_loss, model)

        if early_stopping.early_stop:
            torch.save(model, snapshot_path)
            logger.info("Early stop point")
            break

    if not early_stopping.early_stop:
        torch.save(model, snapshot_path)

    return model


def train_ed(args, train_iter, dev_iter, model):

    if args.finetune:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"] #['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=len(train_iter) * args.batch_size * 0.05, num_training_steps=len(train_iter)* args.batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True, delta=1e-5)
    save_path = os.path.join(args.save_path, args.model)
    os.makedirs(save_path, exist_ok=True)
    snapshot_path = os.path.join(save_path, args.type + '_best_model.pt')


    for _ in range(args.epochs):
        # TRAIN loop
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_iter):
            b_input_ids, b_labels, _ = batch
            b_input_ids = b_input_ids.to(device)
            b_labels = b_labels.to(device)
            outputs = model(b_input_ids, labels=b_labels)
            loss = outputs.loss
            scores = outputs.logits
            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
            # update parameters
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        logger.info("Train loss: {}".format(tr_loss/nb_tr_steps))
        # VALIDATION on validation set
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels = [], []
        for batch in dev_iter:
            b_input_ids, b_labels, _ = batch
            b_input_ids = b_input_ids.to(device)
            b_labels = b_labels.to(device)

            with torch.no_grad():
                outputs = model(b_input_ids, labels=b_labels)
                tmp_eval_loss = outputs.loss
                logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)

            tmp_eval_accuracy = flat_accuracy(np.argmax(logits, axis=2), label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss/nb_eval_steps
        logger.info("Validation loss: {}".format(eval_loss))
        logger.info("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

        early_stopping(eval_loss, model)

        if early_stopping.early_stop:
            torch.save(model, snapshot_path)
            logger.info("Early stopping")
            break

    if not early_stopping.early_stop:
        torch.save(model, snapshot_path)

    return model

def convert(fileName, idFile, outputFile):
    fin = open(fileName)
    fid = open(idFile)
    fout = open(outputFile, "w")

    for line, line_id in zip(fin.readlines(), fid.readlines()):
        query_list = []
        query_text = []
        line = line.strip().split('\t')
        sent = line[0].strip().split()
        pred = line[1].strip().split()
        for token, label in zip(sent, pred):
            if label == 'I':
                query_text.append(token)
            if label == 'O':
                query_text = list(filter(lambda x: x != '<pad>', query_text))
                if len(query_text) != 0:
                    query_list.append(" ".join(list(filter(lambda x:x!='<pad>', query_text))))
                    query_text = []
        query_text = list(filter(lambda x: x != '<pad>', query_text))
        if len(query_text) != 0:
            query_list.append(" ".join(list(filter(lambda x:x!='<pad>', query_text))))
            query_text = []
        if len(query_list) == 0:
            query_list.append(" ".join(list(filter(lambda x:x!='<pad>',sent))))
        fout.write(" %%%% ".join([line_id.strip()]+query_list)+"\n")

def predict_rp(args, model, test_iter, index2tag, data_name="test", load=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path = os.path.join(args.save_path, args.model)
    if load:
        snapshot_path = os.path.join(save_path, args.type + '_best_model.pt')
        model = torch.load(snapshot_path)

    model = model.to(device)

    fname = "{}.txt".format(data_name)
    results_file = open(os.path.join(save_path, fname), 'w')

    model.eval()

    n_correct = 0
    n_retrieved = 0

    fid = open(os.path.join(args.data_dir,"lineids_{}.txt".format(data_name)))
    sent_id = [x.strip() for x in fid.readlines()]
    idx = 0

    for step, batch in enumerate(test_iter):
        # add batch to gpu
        b_input_ids, b_labels, _ = batch
        b_input_ids = b_input_ids.to(device)
        b_labels = b_labels.to(device)
        # forward pass
        outputs = model(b_input_ids, labels=b_labels)
        loss = outputs.loss
        scores = outputs.logits

        cor = torch.sum(torch.max(scores, 1)[1].view(b_labels.size()).data == b_labels.data, dim=1)
        cor = cor.cpu().data.numpy()
        n_correct += cor.sum()

        top_k_scores, top_k_indices = torch.topk(scores, k=args.hits, dim=1, sorted=True)
        top_k_scores_array = top_k_scores.cpu().data.numpy()
        top_k_indices_array = top_k_indices.cpu().data.numpy()
        top_k_relatons_array = index2tag[top_k_indices_array]

        gold_array = index2tag[b_labels.cpu().data.numpy()]

        for i, (relations_row, scores_row) in enumerate(zip(top_k_relatons_array, top_k_scores_array)):
            idx += 1
            relation = gold_array[i]
            for j, (rel, score) in enumerate(zip(relations_row, scores_row)):
                if (rel == relation):
                    label = 1
                    n_retrieved += 1
                else:
                    label = 0
                results_file.write(
                    "{} %%%% {} %%%% {} %%%% {}\n".format(sent_id[idx-1], rel, label, score))

    P = 1. * n_correct / idx #len(test_iter)
    logger.info(' %%%% '.join([args.type, args.data_dir, args.save_path, args.model]))
    logger.info("{} Precision: {:10.6f}%".format(data_name, 100. * P))
    logger.info("no. retrieved: {} out of {}".format(n_retrieved, idx)) #len(test_iter)))
    retrieval_rate = 100. * n_retrieved / idx #len(test_iter)
    logger.info("{} Retrieval Rate {:10.6f}".format(data_name, retrieval_rate))

    return 1

def predict_ed(args, model, test_iter, index2tag, data_name="test", load=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if load:
        save_path = os.path.join(args.save_path, args.model)
        snapshot_path = os.path.join(save_path, args.type + '_best_model.pt')
        model = torch.load(snapshot_path)

    model = model.to(device)

    gold_list = []
    pred_list = []

    fname = "{}.txt".format(data_name)
    temp_file = 'tmp'+fname
    results_file = open(temp_file, 'w')

    n_correct = 0
    model.eval()

    for step, batch in enumerate(test_iter):
        # add batch to gpu
        b_input_ids, b_labels, question_array = batch
        b_input_ids = b_input_ids.to(device)
        b_labels = b_labels.to(device)
        # forward pass
        outputs = model(b_input_ids, labels=b_labels)
        loss = outputs.loss
        scores = outputs.logits

        n_correct += torch.sum(torch.sum(torch.max(scores, 2)[1].view(b_labels.size()).data == b_labels.data, dim=1) \
                          == b_labels.size()[0]).item()
        index_tag = torch.max(scores, 2)[1].view(b_labels.size()).cpu().data.numpy()

        for i in range(len(index_tag)):
            for j in range(len(index_tag[i])):
                if index_tag[i][j] > len(index2tag):
                    index_tag[i][j] = 0

        tag_array = index2tag[index_tag]
        pred_list.append(index_tag)


        gold_array = index2tag[b_labels.cpu().data.numpy()]
        gold_list.append(b_labels.cpu().data.numpy())

        for question, label, gold in zip(question_array, tag_array, gold_array):
            results_file.write("{}\t{}\t{}\n".format(" ".join(question), " ".join(label), " ".join(gold)))

    from evaluation import evaluation
    P, R, F = evaluation(gold_list, pred_list, index2tag, type=False)
    logger.info(' %%%% '.join([args.type, args.data_dir, args.save_path, args.model]))
    logger.info("{} Precision: {:10.6f}% Recall: {:10.6f}% F1 Score: {:10.6f}%".format(data_name, 100. * P, 100. * R, 100. * F))

    results_file.flush()
    results_file.close()
    convert(temp_file, os.path.join(args.data_dir, "lineids_{}.txt".format(data_name)), os.path.join(args.save_path, args.model, "query.{}".format(data_name)))

    return 1

class BuboDataset(Dataset):
    def __init__(self, dataset):
        self.text = []
        self.label = []
        self.raw = []
        for datum in dataset:
            self.text.append(datum.text)
            self.label.append(datum.ed)
            self.raw.append(datum.raw)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx], self.label[idx], self.raw[idx]


def generate_batch(batch):
    text, ed, raw = zip(*batch)
    text = torch.nn.utils.rnn.pad_sequence(text, batch_first=True)
    ed = torch.nn.utils.rnn.pad_sequence(ed, batch_first=True)

    return torch.tensor(text, dtype=torch.long), torch.tensor(ed, dtype=torch.long), raw

def prepare_data(args):
    TEXT = torchtext.data.Field()
    ED = torchtext.data.Field()

    if args.type == 'ed':
        train, dev, test = torchtext.data.TabularDataset.splits(path=args.data_dir, train='train.txt', validation='valid.txt', test='test.txt', format='tsv', fields=[('id', None), ('sub', None), ('entity', None), ('relation', None), ('obj', None), ('text', TEXT), ('ed', ED)], csv_reader_params={'quoting':csv.QUOTE_NONE})
    else:
        train, dev, test = torchtext.data.TabularDataset.splits(path=args.data_dir, train='train.txt', validation='valid.txt', test='test.txt', format='tsv', fields=[('id', None), ('sub', None), ('entity', None), ('ed', ED), ('obj', None), ('text', TEXT), ('span', None)], csv_reader_params={'quoting':csv.QUOTE_NONE})
        
    ED.build_vocab(train, dev)
    
    for t in train:
        t.raw = t.text
        t.text = torch.Tensor(tokenizer.convert_tokens_to_ids(t.text))
        t.ed = torch.Tensor([ED.vocab.stoi[e] for e in t.ed])

    for t in dev:
        t.raw = t.text
        t.text = torch.Tensor(tokenizer.convert_tokens_to_ids(t.text))
        t.ed = torch.Tensor([ED.vocab.stoi[e] for e in t.ed])

    for t in test:
        t.raw = t.text
        t.text = torch.Tensor(tokenizer.convert_tokens_to_ids(t.text))
        t.ed = torch.Tensor([ED.vocab.stoi[e] for e in t.ed])

    train = BuboDataset(train)
    dev = BuboDataset(dev)
    test = BuboDataset(test)

    index2tag = np.array(ED.vocab.itos)

    return DataLoader(train, batch_size=args.batch_size, collate_fn=generate_batch, shuffle=True), DataLoader(dev, batch_size=args.batch_size, collate_fn=generate_batch, shuffle=True), DataLoader(test, batch_size=args.batch_size, collate_fn=generate_batch), index2tag#, index2word


def main(args):
    train_iter, dev_iter, test_iter, index2tag = prepare_data(args)
    set_seed(args)

    if args.type == 'ed':
        logger.info(" $$$$ TAG INFORMATION $$$$ ")
        logger.info(index2tag)

        model = BertForTokenClassification.from_pretrained('../google/{}'.format(args.model), num_labels=len(index2tag))
        if not args.onlyeval:
            model = train_ed(args, train_iter, dev_iter, model)
        else:
            snapshot_path = os.path.join(args.save_path, args.model, args.type + '_best_model.pt')
            model = torch.load(snapshot_path)
        predict_ed(args, model, dev_iter, index2tag, data_name='valid')
        predict_ed(args, model, test_iter, index2tag, data_name='test')

    elif args.type == 'rp':
        model = BertForSequenceClassification.from_pretrained('../google/{}'.format(args.model), num_labels=len(index2tag))
        snapshot_path = os.path.join(args.save_path, args.model, args.type + '_best_model.pt')
        if args.onlyeval: #os.path.exists(snapshot_path):
            model = torch.load(snapshot_path)
        else:
            model = train_rp(args, train_iter, dev_iter, model)
        predict_rp(args, model, test_iter, index2tag) #, load=True)

def set_seed(args):
    np.random.seed(args.seed)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500) #30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=5e-5) #1e-4)
    parser.add_argument('--patience', type=int, default=5) # 10)
    parser.add_argument('--hits', type=int, default=5) # 10)
    parser.add_argument('--type', type=str, default='ed')
    parser.add_argument('--save_path', type=str, default='saved_checkpoints')
    parser.add_argument('--weight_decay',type=float, default=0)
    parser.add_argument('--data_dir', type=str, default='../data/processed_simplequestions_dataset')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--model', type=str)
    parser.add_argument('--onlyeval', action='store_true')
    args = parser.parse_args()

    main(args)
