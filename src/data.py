# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
import json
import numpy as np
from tqdm import tqdm

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 opt,
                 question_prefix='question:',
                 title_prefix='title:',
                 passage_prefix='context:',
                 is_eval=False):
        self.data = data
        self.n_contexts = opt.n_contexts
        self.ctx_anno = opt.ctx_anno ## "has_answer, mytho"
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.is_eval = is_eval
        # self.sort_data()

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target
        elif 'answers' in example:
            return random.choice(example['answers'])
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question']
        target = self.get_target(example)

        if 'ctxs' in example and self.n_contexts is not None:
            f1 = self.passage_prefix + " {}"
            f2 = self.title_prefix + " {} " + self.passage_prefix + " {}"

            contexts = example['ctxs'][:self.n_contexts]
            passages = []
            has_answers = []
            sentence_spans_ = [] ## _ means multiple!
            has_answers_sent_ = []
            input_ids_ = []

            for c in contexts:
                passages.append(f2.format(c['title'], c['text']))
                if self.is_eval:
                    has_answer = c['has_answer']
                else:
                    has_answer = c[self.ctx_anno]
                
                has_answers.append(has_answer)
                
                sentence_spans_.append(c.get('sentence_spans', []))

                has_answers_sent = c.get('has_answers_sent', [])
                if not has_answer:
                    has_answers_sent_.append([0] * len(has_answers_sent))
                else:
                    has_answers_sent_.append(has_answers_sent)
                    
                input_ids_.append(c.get('input_ids'))

            # passages = [f.format(c['title'], c['text']) for c in contexts]
            # TODO(egrave): do we want to keep this?
            if len(contexts) == 0:
                contexts = [question]
        else:
            passages = None

        return {
            'index' : index,
            'question' : question,
            'target' : target,
            'passages' : passages,
            'has_answers' : has_answers,
            'sentence_spans_': sentence_spans_,
            'has_answers_sent_': has_answers_sent_,
            'input_ids_': input_ids_,
        }

    def sort_data(self):
        if self.n_contexts is None or not 'score' in self.data[0]['ctxs'][0]:
            return
        for ex in self.data:
            ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

    def get_example(self, index):
        return self.data[index]

    # def convert_examples_to_features(self, tokenizer, question, examples):
    #     question = self.question_prefix + " " + question
    #     f2 = self.title_prefix + " {} " + self.passage_prefix + " {}"
    #     contexts = [f2.format(example['title'], example['text']) for example in examples]        
    #     text_passages = [question + " " + t for t in contexts]
    #     input_ids, attention_masks = encode_passages([text_passages], tokenizer, 200)
        
    #     return input_ids, attention_masks


def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)

    return passage_ids, passage_masks.bool()

def process_batch_input_ids(batch_input_ids, max_length):
    ## Padding for those with different lengths
    ## batch_input_ids = list of list of input_ids, e.g., (2, 20, length)
    passage_ids, passage_masks = [], []
    for q_input_ids in batch_input_ids:
        q_passage_ids, q_passage_masks = [], []
        for p_input_ids in q_input_ids:
            len_input_ids = len(p_input_ids)
            if len_input_ids > max_length - 1:
                p_input_ids = p_input_ids[:max_length - 1]
                len_input_ids = max_length - 1

            p_input_ids = p_input_ids + [1] + [0] * (max_length - len_input_ids - 1)
            p_attn_mask = [1] * len_input_ids + [1] + [0] * (max_length - len_input_ids - 1)
            
            q_passage_ids.append(p_input_ids)
            q_passage_masks.append(p_attn_mask)
        passage_ids.append(q_passage_ids)
        passage_masks.append(q_passage_masks)
    passage_ids = torch.cat([torch.LongTensor(passage_ids)], dim=0)
    passage_masks = torch.cat([torch.BoolTensor(passage_masks)], dim=0)

    return passage_ids, passage_masks

class Collator(object):
    def __init__(self, text_maxlength, tokenizer, sep_q_p, soft_prompts, extra_question, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength
        self.sep_q_p = sep_q_p
        self.soft_prompts = soft_prompts
        self.extra_question = extra_question

        if self.soft_prompts is not None: ## default: None or int values
            temp_prompts = []
            for i in range(self.soft_prompts):
                temp_prompts.append(f'<extra_id_{i}>')
            self.soft_prompts = ' '.join(temp_prompts)
            print(f">> Set soft_prompts to {self.soft_prompts}")

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        has_answers = torch.tensor([ex['has_answers'] for ex in batch])
        sentence_spans_ = [ex['sentence_spans_'] for ex in batch]
        has_answers_sent_ = [ex['has_answers_sent_'] for ex in batch]
        input_ids_ = [ex['input_ids_'] for ex in batch]

        target = self.tokenizer.batch_encode_plus(
            target,
            padding=True,
            return_tensors='pt'
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_question(example, soft_prompts=None, extra_question=None): ## soft_prompts deprecated
            if example['passages'] is None:
                return [example['question']]
            
            if extra_question == 'embs':
                return [example['question']] + [example['question'] + " " + t for t in example['passages']]
            else:
                return [example['question'] + " " + t for t in example['passages']]

        if input_ids_[0][0] is not None:
            passage_ids, passage_masks = process_batch_input_ids(input_ids_, self.text_maxlength)

        else:
            ## sep_q_p is deprecated .. (2023-05-25)
            if self.sep_q_p:
                text_passages = [[example['question']] + [t for t in example['passages']] for example in batch]
            else:
                text_passages = [append_question(example, self.soft_prompts, self.extra_question) for example in batch]

            passage_ids, passage_masks = encode_passages(text_passages,
                                                        self.tokenizer,
                                                        self.text_maxlength)

        question = [example['question'] for example in batch]
        q_tokens = self.tokenizer(question, add_special_tokens=False)['input_ids']

        return (index, target_ids, passage_ids, passage_masks, q_tokens, has_answers, sentence_spans_, has_answers_sent_)

def load_data(data_path=None, global_rank=-1, world_size=-1, n_qas=None):
    assert data_path
    # if data_path.endswith('.jsonl'):
    #     data = open(data_path, 'r')
    # elif data_path.endswith('.json'):
    #     with open(data_path, 'r') as fin:
    #         data = json.load(fin)
    with open(data_path, 'r') as fin:
        data = json.load(fin)    
    if n_qas is not None:
        data = data[:n_qas]

    examples = []
    for k, example in enumerate(tqdm(data, desc="> Loading data")):
        if global_rank > -1 and not k%world_size==global_rank:
            continue
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        if example.get('id') is None:
            example['id'] = k

        examples.append(example)
    ## egrave: is this needed?
    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()

    return examples

class RetrieverCollator(object):
    def __init__(self, tokenizer, passage_maxlength=200, question_maxlength=40):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength

    def __call__(self, batch):
        index = torch.tensor([ex['index'] for ex in batch])

        question = [ex['question'] for ex in batch]
        question = self.tokenizer.batch_encode_plus(
            question,
            pad_to_max_length=True,
            return_tensors="pt",
            max_length=self.question_maxlength,
            truncation=True
        )
        question_ids = question['input_ids']
        question_mask = question['attention_mask'].bool()

        if batch[0]['scores'] is None or batch[0]['passages'] is None:
            return index, question_ids, question_mask, None, None, None

        scores = [ex['scores'] for ex in batch]
        scores = torch.stack(scores, dim=0)

        passages = [ex['passages'] for ex in batch]
        passage_ids, passage_masks = encode_passages(
            passages,
            self.tokenizer,
            self.passage_maxlength
        )

        return (index, question_ids, question_mask, passage_ids, passage_masks, scores)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 title_prefix='title:',
                 passage_prefix='context:'):
        self.data = data
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        text = self.title_prefix + " " + example[2] + " " + \
            self.passage_prefix + " " + example[1]
        return example[0], text

class TextCollator(object):
    def __init__(self, tokenizer, maxlength=200):
        self.tokenizer = tokenizer
        self.maxlength = maxlength

    def __call__(self, batch):
        index = [x[0] for x in batch]
        encoded_batch = self.tokenizer.batch_encode_plus(
            [x[1] for x in batch],
            pad_to_max_length=True,
            return_tensors="pt",
            max_length=self.maxlength,
            truncation=True
        )
        text_ids = encoded_batch['input_ids']
        text_mask = encoded_batch['attention_mask'].bool()

        return index, text_ids, text_mask
