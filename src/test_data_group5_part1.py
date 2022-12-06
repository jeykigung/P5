from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import gzip
import random
from multiprocessing import Pool
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
import os
from torch.utils.data.distributed import DistributedSampler
from copy import deepcopy

from transformers import T5Tokenizer, T5TokenizerFast
from tokenization import P5Tokenizer, P5TokenizerFast

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

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

    
# This test dataloader version is created for the first four prompts in Task Family 5 (direct recommendation)
class P5_Amazon_Dataset(Dataset):
    def __init__(self, all_tasks, task_list, tokenizer, args, sample_numbers, mode='train', split='toys', rating_augment=False, sample_type='random'): 
        self.all_tasks = all_tasks
        self.task_list = task_list
        self.tokenizer = tokenizer
        self.args = args
        self.sample_numbers = sample_numbers
        self.split = split
        self.rating_augment = rating_augment
        self.sample_type = sample_type
        
        print('Data sources: ', split.split(','))
        self.mode = mode
        assert self.mode == 'test'
        if self.mode == 'test':
            self.review_data = load_pickle(os.path.join('data', split, 'review_splits.pkl'))['test']
            self.exp_data = load_pickle(os.path.join('data', split, 'exp_splits.pkl'))['test']
            if self.rating_augment:
                self.rating_data = load_pickle(os.path.join('data', split, 'rating_splits_augmented.pkl'))['test']
            else:
                self.rating_data = self.review_data
        else:
            raise NotImplementedError
            
        self.sequential_data = ReadLineFromFile(os.path.join('data', split, 'sequential_data.txt'))
        item_count = defaultdict(int)
        user_items = defaultdict()

        for line in self.sequential_data:
            user, items = line.strip().split(' ', 1)
            items = items.split(' ')
            items = [int(item) for item in items]
            user_items[user] = items
            for item in items:
                item_count[item] += 1
                
        self.all_item = list(item_count.keys())
        count = list(item_count.values())
        sum_value = np.sum([x for x in count])
        self.probability = [value / sum_value for value in count]
        self.user_items = user_items
        
        if self.mode == 'test':
            self.negative_samples = ReadLineFromFile(os.path.join('data', split, 'negative_samples.txt'))
            
        datamaps = load_json(os.path.join('data', split, 'datamaps.json'))
        self.user2id = datamaps['user2id']
        self.item2id = datamaps['item2id']
        self.user_list = list(datamaps['user2id'].keys())
        self.item_list = list(datamaps['item2id'].keys())
        self.id2item = datamaps['id2item']
        
        self.user_id2name = load_pickle(os.path.join('data', split, 'user_id2name.pkl'))
        
        self.meta_data = []
        for meta in parse(os.path.join('data', split, 'meta.json.gz')):
            self.meta_data.append(meta)
        self.meta_dict = {}
        for i, meta_item in enumerate(self.meta_data):
            self.meta_dict[meta_item['asin']] = i
            
        print('compute_datum_info')
        self.total_length = 0
        self.datum_info = []
        self.compute_datum_info()
        
    def compute_datum_info(self):
        curr = 0
        assert 'traditional' in self.task_list.keys()
        key = 'traditional'
        assert 0 < int(self.task_list[key].split('-')[1]) <= 4
        self.total_length += len(self.user2id) * 100
        for i in range(self.total_length - curr):
            self.datum_info.append((i + curr, key, i // 100, i % 100))
        curr = self.total_length

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        
        out_dict = {}
        out_dict['args'] = self.args
        
        loss_weight = 1.0
        
        datum_info_idx = self.datum_info[idx]
        assert datum_info_idx[0] == idx
        if len(datum_info_idx) == 4:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2]
            candidate_item_idx = datum_info_idx[3]
        else:
            raise NotImplementedError
            
        if task_name == 'traditional':
            sequential_datum = self.sequential_data[datum_idx]
            sequence = sequential_datum.split()
            user_id = sequence[0]
            user_desc = self.user_id2name[user_id]
            assert self.mode == 'test'
            if candidate_item_idx == 0:
                target_item = sequence[-1]
            
            task_candidates = self.task_list[task_name]
            task_template = self.all_tasks['traditional'][task_candidates]
            assert task_template['task'] == 'traditional'
            
            if task_template['id'] == '5-1':
                if candidate_item_idx == 0:
                    source_text = task_template['source'].format(user_id, target_item)
                    target_text = task_template['target'].format('yes')
                else:
                    assert user_id == self.negative_samples[int(user_id)-1].split(' ', 1)[0]
                    candidate_samples = self.negative_samples[int(user_id)-1].split(' ', 1)[1].split(' ')
                    source_text = task_template['source'].format(user_id, candidate_samples[candidate_item_idx-1])
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-2':
                if candidate_item_idx == 0:
                    source_text = task_template['source'].format(target_item, user_desc)
                    target_text = task_template['target'].format('yes')
                else:
                    assert user_id == self.negative_samples[int(user_id)-1].split(' ', 1)[0]
                    candidate_samples = self.negative_samples[int(user_id)-1].split(' ', 1)[1].split(' ')
                    source_text = task_template['source'].format(candidate_samples[candidate_item_idx-1], user_desc)
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-3':
                if candidate_item_idx == 0:
                    if 'title' in self.meta_data[self.meta_dict[self.id2item[target_item]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[target_item]]]['title']
                    else:
                        title = 'unknown title'
                    source_text = task_template['source'].format(user_desc, title)
                    target_text = task_template['target'].format('yes')
                else:
                    assert user_id == self.negative_samples[int(user_id)-1].split(' ', 1)[0]
                    candidate_samples = self.negative_samples[int(user_id)-1].split(' ', 1)[1].split(' ')
                    if 'title' in self.meta_data[self.meta_dict[self.id2item[candidate_samples[candidate_item_idx-1]]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[candidate_samples[candidate_item_idx-1]]]]['title']
                    else:
                        title = 'unknown title'
                    source_text = task_template['source'].format(user_desc, title)
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-4':
                if candidate_item_idx == 0:
                    if 'title' in self.meta_data[self.meta_dict[self.id2item[target_item]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[target_item]]]['title']
                    else:
                        title = 'unknown title'
                    source_text = task_template['source'].format(user_id, title)
                    target_text = task_template['target'].format('yes')
                else:
                    assert user_id == self.negative_samples[int(user_id)-1].split(' ', 1)[0]
                    candidate_samples = self.negative_samples[int(user_id)-1].split(' ', 1)[1].split(' ')
                    if 'title' in self.meta_data[self.meta_dict[self.id2item[candidate_samples[candidate_item_idx-1]]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[candidate_samples[candidate_item_idx-1]]]]['title']
                    else:
                        title = 'unknown title'
                    source_text = task_template['source'].format(user_id, title)
                    target_text = task_template['target'].format('no')
            else:
                raise NotImplementedError
            
        input_ids = self.tokenizer.encode(
                source_text, padding=True, truncation=True, max_length=self.args.max_text_length)
        tokenized_text = self.tokenizer.tokenize(source_text)
        whole_word_ids = self.calculate_whole_word_ids(tokenized_text, input_ids)
        assert len(whole_word_ids) == len(input_ids)
        
        target_ids = self.tokenizer.encode(
                target_text, padding=True, truncation=True, max_length=self.args.gen_max_length)

        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        out_dict['whole_word_ids'] = torch.LongTensor(whole_word_ids)
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['target_length'] = len(target_ids)

        out_dict['source_text'] = source_text
        out_dict['tokenized_text'] = tokenized_text
        out_dict['target_text'] = target_text

        out_dict['task'] = task_template['task']

        out_dict['loss_weight'] = loss_weight

        return out_dict
    
    def calculate_whole_word_ids(self, tokenized_text, input_ids):
        whole_word_ids = []
        curr = 0
        for i in range(len(tokenized_text)):
            if tokenized_text[i].startswith('▁'):
                curr += 1
                whole_word_ids.append(curr)
            else:
                whole_word_ids.append(curr)
        last_item = whole_word_ids[len(input_ids) - 2]
        return whole_word_ids[:len(input_ids) - 1] + [0] ## the added [0] is for </s>
    
    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        args = self.args

        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        whole_word_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        loss_weights = torch.ones(B, dtype=torch.float)

        tasks = []
        source_text = []
        tokenized_text = []
        target_text = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            whole_word_ids[i, :entry['input_length']] = entry['whole_word_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']

            if 'task' in entry:
                tasks.append(entry['task'])

            if 'source_text' in entry:
                source_text.append(entry['source_text'])
                
            if 'tokenized_text' in entry:
                tokenized_text.append(entry['tokenized_text'])
                
            if 'target_text' in entry:
                target_text.append(entry['target_text'])

            if 'loss_weight' in entry:
                ## length-aware loss normalization
                if entry['target_length'] > 0:
                    loss_weights[i] = entry['loss_weight'] / entry['target_length']
                else:
                    loss_weights[i] = entry['loss_weight']

        assert 't5' in args.backbone
        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100
        batch_entry['task'] = tasks

        batch_entry['source_text'] = source_text
        batch_entry['target_text'] = target_text

        batch_entry['input_ids'] = input_ids
        batch_entry['whole_word_ids'] = whole_word_ids
        batch_entry['target_ids'] = target_ids

        batch_entry['loss_weights'] = loss_weights

        return batch_entry
    
    
class P5_Yelp_Dataset(Dataset):
    def __init__(self, all_tasks, task_list, tokenizer, args, sample_numbers, mode='train', split='yelp', rating_augment=False, sample_type='random'): 
        self.all_tasks = all_tasks
        self.task_list = task_list
        self.tokenizer = tokenizer
        self.args = args
        self.sample_numbers = sample_numbers
        self.split = split
        self.rating_augment = rating_augment
        self.sample_type = sample_type
        
        print('Data sources: ', split.split(','))
        self.mode = mode
        assert self.mode == 'test'
        if self.mode == 'test':
            self.review_data = load_pickle(os.path.join('data', split, 'review_splits.pkl'))['test']
            self.exp_data = load_pickle(os.path.join('data', split, 'exp_splits.pkl'))['test']
            if self.rating_augment:
                self.rating_data = load_pickle(os.path.join('data', split, 'rating_splits_augmented.pkl'))['test']
            else:
                self.rating_data = self.review_data
        else:
            raise NotImplementedError
            
        self.sequential_data = ReadLineFromFile(os.path.join('data', split, 'sequential_data.txt'))
        item_count = defaultdict(int)
        user_items = defaultdict()

        for line in self.sequential_data:
            user, items = line.strip().split(' ', 1)
            items = items.split(' ')
            items = [int(item) for item in items]
            user_items[user] = items
            for item in items:
                item_count[item] += 1
                
        self.all_item = list(item_count.keys())
        count = list(item_count.values())
        sum_value = np.sum([x for x in count])
        self.probability = [value / sum_value for value in count]
        self.user_items = user_items
        
        if self.mode == 'test':
            self.negative_samples = ReadLineFromFile(os.path.join('data', split, 'negative_samples.txt'))
            
        datamaps = load_json(os.path.join('data', split, 'datamaps.json'))
        self.user2id = datamaps['user2id']
        self.item2id = datamaps['item2id']
        self.user_list = list(datamaps['user2id'].keys())
        self.item_list = list(datamaps['item2id'].keys())
        self.id2item = datamaps['id2item']
        
        self.user_id2name = load_pickle(os.path.join('data', split, 'user_id2name.pkl'))
            
        self.meta_data = load_pickle(os.path.join('data', split, 'meta_data.pkl'))
        self.user_data = load_pickle(os.path.join('data', split, 'user_data.pkl'))
        self.meta_dict = {}
        for i, meta_item in enumerate(self.meta_data):
            self.meta_dict[meta_item['business_id']] = i
        self.user_meta_dict = {}
        for j, user_meta_item in enumerate(self.user_data):
            self.user_meta_dict[user_meta_item['user_id']] = j
            
        print('compute_datum_info')
        self.total_length = 0
        self.datum_info = []
        self.compute_datum_info()
        
    def compute_datum_info(self):
        curr = 0
        assert 'traditional' in self.task_list.keys()
        key = 'traditional'
        assert 0 < int(self.task_list[key].split('-')[1]) <= 4
        self.total_length += len(self.user2id) * 100
        for i in range(self.total_length - curr):
            self.datum_info.append((i + curr, key, i // 100, i % 100))
        curr = self.total_length
            
    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        
        out_dict = {}
        out_dict['args'] = self.args
        
        loss_weight = 1.0
        
        datum_info_idx = self.datum_info[idx]
        assert datum_info_idx[0] == idx
        if len(datum_info_idx) == 4:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2]
            candidate_item_idx = datum_info_idx[3]
        else:
            raise NotImplementedError
            
        if task_name == 'traditional':
            sequential_datum = self.sequential_data[datum_idx]
            sequence = sequential_datum.split()
            user_id = sequence[0]
            user_desc = self.user_id2name[user_id]
            assert self.mode == 'test'
            if candidate_item_idx == 0:
                target_item = sequence[-1]
            
            task_candidates = self.task_list[task_name]
            task_template = self.all_tasks['traditional'][task_candidates]
            assert task_template['task'] == 'traditional'
            
            if task_template['id'] == '5-1':
                if candidate_item_idx == 0:
                    source_text = task_template['source'].format(user_id, target_item)
                    target_text = task_template['target'].format('yes')
                else:
                    assert user_id == self.negative_samples[int(user_id)-1].split(' ', 1)[0]
                    candidate_samples = self.negative_samples[int(user_id)-1].split(' ', 1)[1].split(' ')
                    source_text = task_template['source'].format(user_id, candidate_samples[candidate_item_idx-1])
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-2':
                if candidate_item_idx == 0:
                    source_text = task_template['source'].format(target_item, user_desc)
                    target_text = task_template['target'].format('yes')
                else:
                    assert user_id == self.negative_samples[int(user_id)-1].split(' ', 1)[0]
                    candidate_samples = self.negative_samples[int(user_id)-1].split(' ', 1)[1].split(' ')
                    source_text = task_template['source'].format(candidate_samples[candidate_item_idx-1], user_desc)
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-3':
                if candidate_item_idx == 0:
                    if 'name' in self.meta_data[self.meta_dict[self.id2item[target_item]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[target_item]]]['name']
                    else:
                        title = 'unknown name'
                    source_text = task_template['source'].format(user_desc, title)
                    target_text = task_template['target'].format('yes')
                else:
                    assert user_id == self.negative_samples[int(user_id)-1].split(' ', 1)[0]
                    candidate_samples = self.negative_samples[int(user_id)-1].split(' ', 1)[1].split(' ')
                    if 'name' in self.meta_data[self.meta_dict[self.id2item[candidate_samples[candidate_item_idx-1]]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[candidate_samples[candidate_item_idx-1]]]]['name']
                    else:
                        title = 'unknown name'
                    source_text = task_template['source'].format(user_desc, title)
                    target_text = task_template['target'].format('no')
            elif task_template['id'] == '5-4':
                if candidate_item_idx == 0:
                    if 'name' in self.meta_data[self.meta_dict[self.id2item[target_item]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[target_item]]]['name']
                    else:
                        title = 'unknown name'
                    source_text = task_template['source'].format(user_id, title)
                    target_text = task_template['target'].format('yes')
                else:
                    assert user_id == self.negative_samples[int(user_id)-1].split(' ', 1)[0]
                    candidate_samples = self.negative_samples[int(user_id)-1].split(' ', 1)[1].split(' ')
                    if 'name' in self.meta_data[self.meta_dict[self.id2item[candidate_samples[candidate_item_idx-1]]]]:
                        title = self.meta_data[self.meta_dict[self.id2item[candidate_samples[candidate_item_idx-1]]]]['name']
                    else:
                        title = 'unknown name'
                    source_text = task_template['source'].format(user_id, title)
                    target_text = task_template['target'].format('no')
            else:
                raise NotImplementedError
            
        input_ids = self.tokenizer.encode(
                source_text, padding=True, truncation=True, max_length=self.args.max_text_length)
        tokenized_text = self.tokenizer.tokenize(source_text)
        whole_word_ids = self.calculate_whole_word_ids(tokenized_text, input_ids)
        assert len(whole_word_ids) == len(input_ids)
        
        target_ids = self.tokenizer.encode(
                target_text, padding=True, truncation=True, max_length=self.args.gen_max_length)

        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        out_dict['whole_word_ids'] = torch.LongTensor(whole_word_ids)
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['target_length'] = len(target_ids)

        out_dict['source_text'] = source_text
        out_dict['tokenized_text'] = tokenized_text
        out_dict['target_text'] = target_text

        out_dict['task'] = task_template['task']

        out_dict['loss_weight'] = loss_weight

        return out_dict
    
    def calculate_whole_word_ids(self, tokenized_text, input_ids):
        whole_word_ids = []
        curr = 0
        for i in range(len(tokenized_text)):
            if tokenized_text[i].startswith('▁'):
                curr += 1
                whole_word_ids.append(curr)
            else:
                whole_word_ids.append(curr)
        last_item = whole_word_ids[len(input_ids) - 2]
        return whole_word_ids[:len(input_ids) - 1] + [0] # [0] for </s>
    
    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        args = self.args

        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        whole_word_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        loss_weights = torch.ones(B, dtype=torch.float)

        tasks = []
        source_text = []
        tokenized_text = []
        target_text = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            whole_word_ids[i, :entry['input_length']] = entry['whole_word_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']

            if 'task' in entry:
                tasks.append(entry['task'])

            if 'source_text' in entry:
                source_text.append(entry['source_text'])
                
            if 'tokenized_text' in entry:
                tokenized_text.append(entry['tokenized_text'])
                
            if 'target_text' in entry:
                target_text.append(entry['target_text'])

            if 'loss_weight' in entry:
                # length-aware loss normalization
                if entry['target_length'] > 0:
                    loss_weights[i] = entry['loss_weight'] / entry['target_length']
                else:
                    loss_weights[i] = entry['loss_weight']

        assert 't5' in args.backbone
        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100
        batch_entry['task'] = tasks

        batch_entry['source_text'] = source_text
        batch_entry['target_text'] = target_text

        batch_entry['input_ids'] = input_ids
        batch_entry['whole_word_ids'] = whole_word_ids
        batch_entry['target_ids'] = target_ids

        batch_entry['loss_weights'] = loss_weights

        return batch_entry
    

def get_loader(args, task_list, sample_numbers, split='toys', mode='train', 
               batch_size=16, workers=4, distributed=False):

    if 't5' in args.backbone:
        tokenizer = P5Tokenizer.from_pretrained(
            args.backbone, 
            max_length=args.max_text_length, 
            do_lower_case=args.do_lower_case)

    if split == 'yelp':
        from all_yelp_templates import all_tasks as task_templates
        
        dataset = P5_Yelp_Dataset(
            task_templates,
            task_list,
            tokenizer,
            args,
            sample_numbers,
            mode=mode,
            split=split,
            rating_augment=False
        )
    else:
        from all_amazon_templates import all_tasks as task_templates

        dataset = P5_Amazon_Dataset(
            task_templates,
            task_list,
            tokenizer,
            args,
            sample_numbers,
            mode=mode,
            split=split,
            rating_augment=False
        )

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)
        
    return loader
