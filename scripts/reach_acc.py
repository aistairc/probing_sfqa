import pickle, os
from collections import defaultdict
from argparse import ArgumentParser

# copy & paste from BuboQA/scripts/util.py
def www2fb(in_str):
    if in_str.startswith("www.freebase.com"):
        in_str = 'fb:%s' % (in_str.split('www.freebase.com/')[-1].replace('/', '.'))
    return in_str

# load FB2M; make gold index
def load_fb2m(args):
    obj_sbj = defaultdict(set)
    obj_rel = defaultdict(set)
    fb2m = defaultdict(dict)

    for line in open(args.fb2m_path):
        tokens = line.strip().split('\t')
        if len(tokens) != 3:
            print('error : {}'.format(line))
        sbj = www2fb(tokens[0])
        rel = www2fb(tokens[1])
        objs = tokens[2]

        if rel not in fb2m[sbj]:
            fb2m[sbj][rel] = []
        for obj in tokens[2].split():
            fb2m[sbj][rel].append(www2fb(obj))
        
        if len(objs.split()) == 1:
            obj_sbj[www2fb(objs)].add(sbj)
            obj_rel[www2fb(objs)].add(rel)
        else:
            for obj in objs.split():
                obj = www2fb(obj)
                obj_sbj[obj].add(sbj)
                obj_rel[obj].add(rel)
        
    pickle.dump(obj_sbj, open(args.pickle_path+'fb2m_obj_sbj.pkl', 'wb'))
    pickle.dump(obj_rel, open(args.pickle_path+'fb2m_obj_rel.pkl', 'wb'))
    pickle.dump(fb2m, open(args.pickle_path+'fb2m.pkl', 'wb'))

    return obj_sbj, obj_rel, fb2m

def load_gold(args, obj_sbj, obj_rel):
    # load gold data
    path_gold = args.dataset_path+args.dataset_name+'/test.txt'

    gold_obj = {}
    gold_ent = 0.0
    gold_rel = 0.0
    cnt = 0

    for line in open(path_gold):
        cnt += 1
        tokens = line.strip().split('\t')
        sbj = tokens[1]
        rel = tokens[3]
        obj = tokens[4]
        
        if sbj in obj_sbj[obj]:
            gold_ent += 1
        if rel in obj_rel[obj]:
            gold_rel += 1
        gold_obj[tokens[0]] = obj

    print('gold recall for ent : {}'.format(gold_ent / cnt))
    print('gold recall for rel : {}'.format(gold_rel / cnt))

    return gold_obj

def eval_el(args, obj_sbj, gold_obj):
    path = '../BuboQA/entity_linking/'
    path = path + '{}/{}/lstm/'.format(args.dataset_name, args.model_name)
    txt_file = 'test-h100.txt'

    path_txt = path + txt_file
    if not os.path.exists(path_txt): return 0

    ent_cor = 0.0

    for line in open(path_txt):
        tokens = line.strip().split(' %%%% ')
        if len(tokens) == 1:
            continue
        id = tokens[0]
        for token in tokens[1:51]:
            sbj = token.strip().split('\t')[0]
            if sbj in obj_sbj[gold_obj[id]]:
                ent_cor += 1
                break

    print('{}\t{}\ted'.format(args.dataset_name, args.model_name))
    print(ent_cor / len(gold_obj))

    return 1        

def eval_rp(args, obj_rel, gold_obj):
    path = '../BertQA/'
    path = path + '{}/{}/lstm/'.format(args.dataset_name, args.model_name)
    txt_file = 'test.rp'

    path_txt = path + txt_file
    if not os.path.exists(path_txt): return 0

    rel_cor = 0.0
    pred_res = defaultdict(list)

    for line in open(path_txt):
        tokens = line.strip().split(' %%%% ')
        id = tokens[0]
        if len(pred_res[id]) < 5:
            pred_res[id].append(tokens[1].strip())

    for id in pred_res:
        for rel in pred_res[id]:
            if rel in obj_rel[gold_obj[id]]:
                rel_cor += 1
                break

    print('{}\t{}\trp'.format(args.dataset_name, args.model_name))
    print(rel_cor / len(gold_obj))
        
    return 1    

def eval_ei(args, fb2m, gold_obj):
    path = '../BuboQA/evidence_integration/'
    path = path + '{}/{}/lstm-cnn/'.format(args.dataset_name, args.model_name)
    txt_file = 'test.txt'

    path_txt = path + txt_file
    if not os.path.exists(path_txt): return 0

    path_cor = 0.0

    for line in open(path_txt):
        tokens = line.strip().split(' %%%% ')
        if len(tokens) == 1:
            continue
        id = tokens[0]
        sbj = tokens[1].split()[0]
        rel = tokens[1].split()[1]
        obj = gold_obj[id]

        if obj in fb2m.get(sbj, dict()).get(rel, []):
            path_cor += 1

    print('{}\t{}\trp'.format(args.dataset_name, args.model_name))
    print(path_cor / len(gold_obj))

    return 1

def main(args):
    if os.path.exists(args.pickle_path+'fb2m_obj_sbj.pkl'):
        obj_sbj = pickle.load(open(args.pickle_path+'fb2m_obj_sbj.pkl', 'rb'))
        obj_rel = pickle.load(open(args.pickle_path+'fb2m_obj_rel.pkl', 'rb'))
        fb2m = pickle.load(open(args.pickle_path+'fb2m.pkl','rb'))
    else:
        obj_sbj, obj_rel, fb2m = load_fb2m(args)

    gold_obj = load_gold(args, obj_sbj, obj_rel)
    eval_el(args, obj_sbj, gold_obj)
    eval_rp(args, obj_rel, gold_obj)
    eval_ei(args, fb2m, gold_obj)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='FBQ')
    parser.add_argument('--model_name', type=str, default='bert_uncased_L-12_H-768_A-12')
    parser.add_argument('--pickle_path', type=str, default='./')
    parser.add_argument('--fb2m_path', type=str, default='../BuboQA/data/SimpleQuestions_v2/freebase-subsets/freebase-FB2M.txt')
    parser.add_argument('--dataset_path', type=str, default='../simple-qa-analysis/datasets/')
    args = parser.parse_args()
    main(args)
