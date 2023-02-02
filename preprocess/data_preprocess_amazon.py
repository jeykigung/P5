# %%
from collections import defaultdict
import os
import torch
import random
import numpy as np
import pandas as pd
import json
import pickle
import gzip
from tqdm import tqdm

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    
def ReadLineFromFile(path):
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)
        
'''
Set seeds
'''
seed = 2022
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# %%
short_data_name = 'sports'
os.mkdir(short_data_name)

# %%
if short_data_name == 'beauty':
    full_data_name = 'Beauty'
elif short_data_name == 'toys':
    full_data_name = 'Toys_and_Games'
elif short_data_name == 'sports':
    full_data_name = 'Sports_and_Outdoors'
else:
    raise NotImplementedError

# %% [markdown]
# ### For Sequential Recommendation

# %%
# return (user item timestamp) sort in get_interaction
def Amazon(dataset_name, rating_score):
    '''
    reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
    asin - ID of the product, e.g. 0000013714
    reviewerName - name of the reviewer
    helpful - helpfulness rating of the review, e.g. 2/3
    --"helpful": [2, 3],
    reviewText - text of the review
    --"reviewText": "I bought this for my husband who plays the piano. ..."
    overall - rating of the product
    --"overall": 5.0,
    summary - summary of the review
    --"summary": "Heavenly Highway Hymns",
    unixReviewTime - time of the review (unix time)
    --"unixReviewTime": 1252800000,
    reviewTime - time of the review (raw)
    --"reviewTime": "09 13, 2009"
    '''
    datas = []
    # older Amazon
    data_file = './raw_data/reviews_' + dataset_name + '.json.gz'
    # latest Amazon
    # data_file = '/home/hui_wang/data/new_Amazon/' + dataset_name + '.json.gz'
    for inter in parse(data_file):
        if float(inter['overall']) <= rating_score: # 小于一定分数去掉
            continue
        user = inter['reviewerID']
        item = inter['asin']
        time = inter['unixReviewTime']
        datas.append((user, item, int(time)))
    return datas

def Amazon_meta(dataset_name, data_maps):
    '''
    asin - ID of the product, e.g. 0000031852
    --"asin": "0000031852",
    title - name of the product
    --"title": "Girls Ballet Tutu Zebra Hot Pink",
    description
    price - price in US dollars (at time of crawl)
    --"price": 3.17,
    imUrl - url of the product image (str)
    --"imUrl": "http://ecx.images-amazon.com/images/I/51fAmVkTbyL._SY300_.jpg",
    related - related products (also bought, also viewed, bought together, buy after viewing)
    --"related":{
        "also_bought": ["B00JHONN1S"],
        "also_viewed": ["B002BZX8Z6"],
        "bought_together": ["B002BZX8Z6"]
    },
    salesRank - sales rank information
    --"salesRank": {"Toys & Games": 211836}
    brand - brand name
    --"brand": "Coxlures",
    categories - list of categories the product belongs to
    --"categories": [["Sports & Outdoors", "Other Sports", "Dance"]]
    '''
    datas = {}
    meta_file = './raw_data/meta_' + dataset_name + '.json.gz'
    item_asins = list(data_maps['item2id'].keys())
    for info in parse(meta_file):
        if info['asin'] not in item_asins:
            continue
        datas[info['asin']] = info
    return datas

def add_comma(num):
    # 1000000 -> 1,000,000
    str_num = str(num)
    res_num = ''
    for i in range(len(str_num)):
        res_num += str_num[i]
        if (len(str_num)-i-1) % 3 == 0:
            res_num += ','
    return res_num[:-1]

# categories 和 brand is all attribute
def get_attribute_Amazon(meta_infos, datamaps, attribute_core):

    attributes = defaultdict(int)
    for iid, info in tqdm(meta_infos.items()):
        for cates in info['categories']:
            for cate in cates[1:]: # 把主类删除 没有用
                attributes[cate] +=1
        try:
            attributes[info['brand']] += 1
        except:
            pass

    print(f'before delete, attribute num:{len(attributes)}')
    new_meta = {}
    for iid, info in tqdm(meta_infos.items()):
        new_meta[iid] = []

        try:
            if attributes[info['brand']] >= attribute_core:
                new_meta[iid].append(info['brand'])
        except:
            pass
        for cates in info['categories']:
            for cate in cates[1:]:
                if attributes[cate] >= attribute_core:
                    new_meta[iid].append(cate)
    # 做映射
    attribute2id = {}
    id2attribute = {}
    attributeid2num = defaultdict(int)
    attribute_id = 1
    items2attributes = {}
    attribute_lens = []

    for iid, attributes in new_meta.items():
        item_id = datamaps['item2id'][iid]
        items2attributes[item_id] = []
        for attribute in attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            attributeid2num[attribute2id[attribute]] += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))
    print(f'before delete, attribute num:{len(attribute2id)}')
    print(f'attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}')
    # 更新datamap
    datamaps['attribute2id'] = attribute2id
    datamaps['id2attribute'] = id2attribute
    datamaps['attributeid2num'] = attributeid2num
    return len(attribute2id), np.mean(attribute_lens), datamaps, items2attributes


def get_interaction(datas):
    user_seq = {}
    for data in datas:
        user, item, time = data
        if user in user_seq:
            user_seq[user].append((item, time))
        else:
            user_seq[user] = []
            user_seq[user].append((item, time))

    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1])  # 对各个数据集得单独排序
        items = []
        for t in item_time:
            items.append(t[0])
        user_seq[user] = items
    return user_seq

# K-core user_core item_core
def check_Kcore(user_items, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        for item in items:
            user_count[user] += 1
            item_count[item] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True # 已经保证Kcore

# 循环过滤 K-core
def filter_Kcore(user_items, user_core, item_core): # user 接所有items
    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    while not isKcore:
        for user, num in user_count.items():
            if user_count[user] < user_core: # 直接把user 删除
                user_items.pop(user)
            else:
                for item in user_items[user]:
                    if item_count[item] < item_core:
                        user_items[user].remove(item)
        user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    return user_items

def id_map(user_items): # user_items dict
    user2id = {} # raw 2 uid
    item2id = {} # raw 2 iid
    id2user = {} # uid 2 raw
    id2item = {} # iid 2 raw
    user_id = 0
    item_id = 0
    final_data = {}
    random_user_list = list(user_items.keys())
    random.shuffle(random_user_list)
    
    user_set = set()
    item_set = set()
    for user in random_user_list:
        user_set.add(user)
        items = user_items[user]
        item_set.update(items)
        
    random_user_mapping = [str(i) for i in range(len(user_set))]
    random_item_mapping = [str(i) for i in range(len(item_set))]
    random.shuffle(random_user_mapping)
    random.shuffle(random_item_mapping)
    
    for user in random_user_list:
        items = user_items[user]
        if user not in user2id:
            user2id[user] = str(random_user_mapping[user_id])
            id2user[str(random_user_mapping[user_id])] = user
            user_id += 1
        iids = [] # item id lists
        for item in items:
            if item not in item2id:
                item2id[item] = str(random_item_mapping[item_id])
                id2item[str(random_item_mapping[item_id])] = item
                item_id += 1
            iids.append(item2id[item])
        uid = user2id[user]
        final_data[uid] = iids
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item
    }
    return final_data, user_id, item_id, data_maps

# %%
def main(data_name, acronym, data_type='Amazon'):
    assert data_type in {'Amazon', 'Yelp'}
    rating_score = 0.0  # rating score smaller than this score would be deleted
    # user 5-core item 5-core
    user_core = 5
    item_core = 5
    attribute_core = 0

    if data_type == 'Yelp':
        date_max = '2019-12-31 00:00:00'
        date_min = '2019-01-01 00:00:00'
        datas = Yelp(date_min, date_max, rating_score)
    else:
        datas = Amazon(data_name+'_5', rating_score=rating_score)

    user_items = get_interaction(datas)
    print(f'{data_name} Raw data has been processed! Lower than {rating_score} are deleted!')
    # raw_id user: [item1, item2, item3...]
    user_items = filter_Kcore(user_items, user_core=user_core, item_core=item_core)
    print(f'User {user_core}-core complete! Item {item_core}-core complete!')

    user_items, user_num, item_num, data_maps = id_map(user_items)
    user_count, item_count, _ = check_Kcore(user_items, user_core=user_core, item_core=item_core)
    user_count_list = list(user_count.values())
    user_avg, user_min, user_max = np.mean(user_count_list), np.min(user_count_list), np.max(user_count_list)
    item_count_list = list(item_count.values())
    item_avg, item_min, item_max = np.mean(item_count_list), np.min(item_count_list), np.max(item_count_list)
    interact_num = np.sum([x for x in user_count_list])
    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    show_info = f'Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                f'Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                f'Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%'
    print(show_info)


    print('Begin extracting meta infos...')

    if data_type == 'Amazon':
        meta_infos = Amazon_meta(data_name, data_maps)
        attribute_num, avg_attribute, datamaps, item2attributes = get_attribute_Amazon(meta_infos, data_maps, attribute_core)
    else:
        meta_infos = Yelp_meta(data_maps)
        attribute_num, avg_attribute, datamaps, item2attributes = get_attribute_Yelp(meta_infos, data_maps, attribute_core)

    print(f'{data_name} & {add_comma(user_num)}& {add_comma(item_num)} & {user_avg:.1f}'
          f'& {item_avg:.1f}& {add_comma(interact_num)}& {sparsity:.2f}\%&{add_comma(attribute_num)}&'
          f'{avg_attribute:.1f} \\')

    # -------------- Save Data ---------------
    data_file = './{}/'.format(acronym) + 'sequential_data.txt'
    item2attributes_file = './{}/'.format(acronym) + 'item2attributes.json'
    datamaps_file = './{}/'.format(acronym) + 'datamaps.json'

    with open(data_file, 'w') as out:
        for user, items in user_items.items():
            out.write(user + ' ' + ' '.join(items) + '\n')
    json_str = json.dumps(item2attributes)
    with open(item2attributes_file, 'w') as out:
        out.write(json_str)
        
    json_str = json.dumps(datamaps)
    with open(datamaps_file, 'w') as out:
        out.write(json_str)

# %%
main(full_data_name, short_data_name, data_type='Amazon')

# %%
def sample_test_data(data_name, test_num=99, sample_type='random'):
    """
    sample_type:
        random:  sample `test_num` negative items randomly.
        pop: sample `test_num` negative items according to item popularity.
    """

    data_file = f'sequential_data.txt'
    test_file = f'negative_samples.txt'

    item_count = defaultdict(int)
    user_items = defaultdict()

    lines = open('./{}/'.format(data_name) + data_file).readlines()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_items[user] = items
        for item in items:
            item_count[item] += 1

    all_item = list(item_count.keys())
    count = list(item_count.values())
    sum_value = np.sum([x for x in count])
    probability = [value / sum_value for value in count]

    user_neg_items = defaultdict()

    for user, user_seq in user_items.items():
        test_samples = []
        while len(test_samples) < test_num:
            if sample_type == 'random':
                sample_ids = np.random.choice(all_item, test_num, replace=False)
            else: # sample_type == 'pop':
                sample_ids = np.random.choice(all_item, test_num, replace=False, p=probability)
            sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in test_samples]
            test_samples.extend(sample_ids)
        test_samples = test_samples[:test_num]
        user_neg_items[user] = test_samples

    with open('./{}/'.format(data_name) + test_file, 'w') as out:
        for user, samples in user_neg_items.items():
            out.write(user+' '+' '.join(samples)+'\n')

# %%
sample_test_data(short_data_name)

# %% [markdown]
# ### 

# %% [markdown]
# ### Create Splits for Review

# %%
datamaps = load_json("./{}/datamaps.json".format(short_data_name))
print(datamaps.keys())

# %%
review_data = []
for review in parse("./raw_data/reviews_{}_5.json.gz".format(full_data_name)):
    review_data.append(review)
print(len(review_data))
print(review_data[0])

raw_explanations = load_pickle('./raw_data/reviews_{}.pickle'.format(full_data_name))
print(len(raw_explanations))
print(raw_explanations[0])

print(len(datamaps['user2id']))
print(len(datamaps['item2id']))
sparsity = 100.0 * len(review_data) / (len(datamaps['user2id']) * len(datamaps['item2id'])) 
print('sparsity: ', sparsity)

# %%
valid_review_indices = []
for i in tqdm(range(len(review_data))):
    if review_data[i]['reviewerID'] in datamaps['user2id'] and review_data[i]['asin'] in datamaps['item2id']:
        valid_review_indices.append(i)
print(len(valid_review_indices))

# %%
combined_review_data = []
no_sentence = 0
for i in range(len(review_data)):
    rev_ = review_data[i]
    exp_ = raw_explanations[i]
    assert rev_['reviewerID'] == exp_['user']
    assert rev_['asin'] == exp_['item']
    if 'sentence' in exp_:
        list_len = len(exp_['sentence'])
        selected_idx = random.randint(0, list_len-1)
        rev_['explanation'] = exp_['sentence'][selected_idx][2]
        rev_['feature'] = exp_['sentence'][selected_idx][0]   # add a random, or list all possible sentences
    else:
        no_sentence += 1
    combined_review_data.append(rev_)

# %%
combined_review_data[16]

# %% [markdown]
# ### Metadata for Users & Items

# %%
user_id2name = {}
for i in range(len(combined_review_data)):
    user_id = datamaps['user2id'][combined_review_data[i]['reviewerID']]
    if 'reviewerName' in combined_review_data[i]:
        user_id2name[user_id] = combined_review_data[i]['reviewerName']
    else:
        user_id2name[user_id] = combined_review_data[i]['reviewerID']

# %%
save_pickle(user_id2name, '{}/user_id2name.pkl'.format(short_data_name))

# %% [markdown]
# ### Create Train/Val/Test Splits

# %%
population = len(review_data)
print(population)
data = range(population)

user_mention_dict = {}
item_mention_dict = {}
for i in data:
    review_datum = review_data[i]
    user_ = review_datum['reviewerID']
    item_ = review_datum['asin']
    if user_ not in user_mention_dict:
        user_mention_dict[user_] = [i]
    else:
        user_mention_dict[user_].append(i)
    if item_ not in item_mention_dict:
        item_mention_dict[item_] = [i]
    else:
        item_mention_dict[item_].append(i)

# %%
train_indices = []
for u in tqdm(user_mention_dict.keys()):
    index_cand = user_mention_dict[u]
    random_choice = random.randint(0, len(index_cand)-1)
    if index_cand[random_choice] not in train_indices:
        train_indices.append(index_cand[random_choice])
for it in tqdm(item_mention_dict.keys()):
    index_cand = item_mention_dict[it]
    random_choice = random.randint(0, len(index_cand)-1)
    if index_cand[random_choice] not in train_indices:
        train_indices.append(index_cand[random_choice])
print(len(train_indices))

# %%
remaining_indices = list(set(range(population)).difference(set(train_indices))) # in population but not in train_indices
        
print(len(remaining_indices))

# %%
sub_indices = random.sample(range(len(remaining_indices)), round(population * 0.8) - len(train_indices))
print(len(sub_indices))

final_train_indices = list(set(np.array(remaining_indices)[sub_indices]).union(set(train_indices)))
print(len(final_train_indices))

# %%
val_test_indices = list(set(range(population)).difference(set(final_train_indices)))
print(len(val_test_indices))

# %%
sub_sub_indices = random.sample(range(len(val_test_indices)), round(population * 0.1))

val_indices = list(np.array(val_test_indices)[sub_sub_indices])
test_indices = list(set(val_test_indices).difference(set(val_indices)))
print(len(val_indices))
print(len(test_indices))

all_indices = final_train_indices + val_indices + test_indices
print(len(set(all_indices)))

# %%
train_review_data = []
for i in final_train_indices:
    train_review_data.append(combined_review_data[i])
    
val_review_data = []
for j in val_indices:
    val_review_data.append(combined_review_data[j])
    
test_review_data = []
for k in test_indices:
    test_review_data.append(combined_review_data[k])

# %%
outputs = {'train': train_review_data,
           'val': val_review_data,
           'test': test_review_data,
           'train_indices': final_train_indices,
           'val_indices': val_indices,
           'test_indices': test_indices
}

# %%
save_pickle(outputs, './{}/review_splits.pkl'.format(short_data_name))

# %%
train_review_data[80]

# %%
train_exp_data = []
for i in final_train_indices:
    if 'explanation' in combined_review_data[i]:
        train_exp_data.append(combined_review_data[i])
        
val_exp_data = []
for j in val_indices:
    if 'explanation' in combined_review_data[j]:
        val_exp_data.append(combined_review_data[j])

test_exp_data = []
for k in test_indices:
    if 'explanation' in combined_review_data[k]:
        test_exp_data.append(combined_review_data[k])

# %%
outputs = {'train': train_exp_data,
           'val': val_exp_data,
           'test': test_exp_data,
}

# %%
save_pickle(outputs, './{}/exp_splits.pkl'.format(short_data_name))

# %%
train_exp_data[6]

# %% [markdown]
# ### Re-balancing Training Split

# %%
data_splits = load_pickle('./{}/review_splits.pkl'.format(short_data_name))
train_review_data = data_splits['train']

# %% [markdown]
# #### Augmentation of Minority Ratings

# %%
counts = {float(r): 0 for r in range(1,6)}
for record in train_review_data:
    counts[record['overall']] += 1
T = sum(counts.values())
print(T)
for k,c in counts.items():
    counts[k] = float(counts[k])/T
print(counts)

# %%
import matplotlib.pyplot as plt
X = [float(r) for r in range(1,6)]
plt.bar(X, [counts[x] for x in X])
plt.show()

# %%
torch.arange(1, 0, -0.1)

# %%
torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))

# %% [markdown]
# #### 1. How should each class augment

# %%
augment_prob = {r: max(1.0 / (len(counts)-1) - c,0.) for r,c in counts.items()}
print(augment_prob)
sum_P = sum(augment_prob.values())
print(sum_P)
all_ratings = [float(r) for r in range(1,6)]
augment_multiplier = {r: augment_prob[r] / (sum_P * counts[r]) for r in all_ratings}
print(augment_multiplier)

# %% [markdown]
# #### 2. Augment with variation

# %%
import numpy as np
dist_to_prob = [0.5,0.1,0.03,0.01,0.003]
rating_perturbation_prob = {}
for r in all_ratings:
    P = np.array([dist_to_prob[int(abs(r - r_2))] for r_2 in all_ratings])
    P /= np.sum(P)
    rating_perturbation_prob[r] = P
print(rating_perturbation_prob)
augmented_counts = {r: 0 for r in all_ratings}
augmented_records = []
for record in train_review_data:
    record_rating = record['overall']
    M = augment_multiplier[record_rating]
    remainder = M % 1
    M = int(M)+1 if np.random.random() < remainder else int(M)
    sample_amount = np.random.multinomial(M, rating_perturbation_prob[record_rating])
    for i,n_sample in enumerate(sample_amount):
        for j in range(n_sample):
            new_record = {k:v for k,v in record.items()}
            new_record['overall'] = float(i+1)
            augmented_records.append(new_record)
        augmented_counts[float(i+1)] += n_sample
print(augmented_counts)
print(len(augmented_records))

# %%
data_splits['train'] = data_splits['train'] + augmented_records
print(len(data_splits['train']))

# %%
save_pickle(data_splits, './{}/rating_splits_augmented.pkl'.format(short_data_name)) # for rating

# %% [markdown]
# #### 3. New counts

# %%
train_review_data = data_splits['train']

# %%
counts = {float(r): 0 for r in range(1,6)}
for record in train_review_data:
    counts[record['overall']] += 1
T = sum(counts.values())
print(T)
for k,c in counts.items():
    counts[k] = float(counts[k])/T
print(counts)

# %%
import matplotlib.pyplot as plt
X = [float(r) for r in range(1,6)]
plt.bar(X, [counts[x] for x in X])
plt.show()

# %%



