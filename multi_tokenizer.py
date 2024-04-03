import os
import torch
import pickle
import json
import argparse
import numpy as npl
import multiprocessing

from nltk.tokenize import sent_tokenize
from difflib import SequenceMatcher

from copy import deepcopy
from tqdm import tqdm
from transformers import T5Tokenizer
from functools import partial

from src.evaluation import has_answer

TEXT_MAXLENGTH = 192
N_CONTEXTS = 20

tok = T5Tokenizer.from_pretrained('t5-base')
def get_sentence_position(qas):

    qas['ctxs'] = qas['ctxs'][:N_CONTEXTS]
    qas = get_sentence_positions_using_nltk(qas, tok)

    # tok(f"question: {q} title: {ctx['title']} context: {ctx['text']}")['input_ids']
    # tok = T5Tokenizer.from_pretrained('T5-base')
    q = qas['question']
    ctxs = qas['ctxs']
    # if qas.get('target') is not None:
    #     answers = [qas['target']]
    # else:
    answers = qas['answers']

    for ctx in ctxs:
        title = ctx['title']

        if ctx.get('has_answer') is None:
            context_text = f"title: {ctx['title']} context: {ctx['text']}"
            ctx_has_answer = int(has_answer(answers, context_text))
            ctx['has_answer'] = ctx_has_answer

        # context_input_ids = tok(context, return_tensors='pt')['input_ids'][0, :-1]
        q_t = tok(f"question: {q} title: {title} context: ", return_tensors='pt')['input_ids'][0][:-1]
        append_len_q_t = len(q_t)
        append_len_q = len(tok(f"question: {q} title: ")['input_ids'][:-1])

        # if len(context_input_ids) + append_len_q_t > TEXT_MAXLENGTH:
        #     context_input_ids = context_input_ids[:TEXT_MAXLENGTH - append_len_q_t]

        # period_positions = torch.where(context_input_ids == 5)[0] + 1 ## +1 to get the position of the period
        period_positions = ctx['period_positions_sent']

        s_idx = 0
        sentence_spans = [(append_len_q, append_len_q_t - 2)] ## -2 to remove space and a title
        has_answers_sent = [int(has_answer(answers, title))]
        for p in period_positions:
            p = p [1]
            # p = p.item()
            sentence_spans.append((s_idx + append_len_q_t, p + append_len_q_t))
            # has_answers_sent.append(int(has_answer(answers, tok.decode(context_input_ids[s_idx:p]))))
            s_idx = p
            
        # if s_idx < len(context_input_ids):
        #     sentence_spans.append((s_idx + append_len_q_t, len(context_input_ids) + append_len_q_t))
        #     has_answers_sent.append(int(has_answer(answers, tok.decode(context_input_ids[s_idx:len(context_input_ids)]))))

        '''
        sentences = sent_tokenize(context)
        tokens = [q_t]
        s_idx = append_len_q_t
        for sentence in sentences:
            sent_tokens = tok(sentence, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
            tokens.append(sent_tokens)
            e_idx = s_idx + len(sent_tokens)
            sentence_spans.append((s_idx, e_idx))
            has_answers_sent.append(int(has_answer(answers, sentence)))
            s_idx = e_idx
        tokens = torch.cat(tokens, dim=0).tolist()
        ctx['input_ids'] = tokens
        '''

        ctx['sentence_spans'] = sentence_spans
        ctx['has_answers_sent'] = has_answers_sent + ctx['has_answers_sent'] ## title + context
        del ctx['period_positions_sent']
        # ctx['has_answers_sent'] = has_answers_sent

    return qas



def get_sentence_positions_using_nltk(qas, tok):
    q = qas['question']
    ctxs = qas['ctxs']
    # if qas.get('target') is not None:
    #     answers = [qas['target']]
    # else:
    answers = qas['answers']

    for ctx in ctxs:
    # for ctx in [ctxs[1]]:
        title = ctx['title']
        context = ctx['text']

        context_input_ids = tok(context, return_tensors='pt', add_special_tokens=False)['input_ids'][0]

        sentences = sent_tokenize(context)
        sent_tokens = tok(sentences, add_special_tokens=False)['input_ids']
        period_positions_sent = []
        last_token_sent_in_last_token_doc = False
        last_token_doc_in_last_token_sent = False
        cur_len = 0
        for sent_tokens_ in sent_tokens:
            if cur_len >= len(context_input_ids):
                break
            diff = 0
            skip_this_sent = False
            sent_len = len(sent_tokens_)
            
            first_token_doc = context_input_ids[cur_len].item()
            first_token_doc_decode = tok.decode(first_token_doc)
            first_token_sent = sent_tokens_[0]
            first_token_sent_decode = tok.decode(first_token_sent)
            same_first_token = first_token_doc_decode == first_token_sent_decode

            if last_token_doc_in_last_token_sent: ### "."    doc은 .에서 시작 sent는 "(2)에서 시작. 
                if first_token_sent_decode == '':
                    diff = 1
                elif first_token_sent_decode in first_token_doc_decode:
                    diff = 0
                else:
                    diff = -1
                last_token_doc_in_last_token_sent = False
            elif last_token_sent_in_last_token_doc: ## 일단 1 뺴고 시작
                diff = 1  ## 일단 1 빼고 시작
                while True:
                    if diff >= sent_len:
                        skip_this_sent = True
                        break
                    first_token_sent = sent_tokens_[diff]
                    first_token_sent_decode = tok.decode(first_token_sent)
                    if (first_token_sent_decode != '' and first_token_sent_decode in first_token_doc_decode) or (first_token_doc_decode in first_token_sent_decode and first_token_doc_decode != ''):
                        match = SequenceMatcher(None, sent_tokens_, context_input_ids[cur_len:cur_len+len(sent_tokens_)].tolist()).find_longest_match()
                        if match.size > 0:
                            sent_exp_len = match.a
                            doc_exp_len = match.b
                            diff = sent_exp_len - doc_exp_len
                        break
                    
                    # if first_token_doc_decode == '':
                    #     pass
                    # elif first_token_sent_decode in first_token_doc_decode or first_token_doc_decode in first_token_sent_decode:
                    #     break
                    diff = diff + 1
                last_token_sent_in_last_token_doc = False
            elif not same_first_token: ## sentence로 나누니까 token이 달라짐. ". (19)"와 "(19)" 의 토큰이 다름.
                if sent_len == 1:
                    skip_this_sent = True
                elif first_token_doc_decode in first_token_sent_decode: ## sent: (19 doc: ( --> n 칸 더 가자
                    if tok("_" + tok.decode(sent_tokens_))['input_ids'][2] == first_token_doc:
                        match = SequenceMatcher(None, sent_tokens_, context_input_ids[cur_len:cur_len+len(sent_tokens_)].tolist()).find_longest_match()
                        sent_exp_len = match.a
                        doc_exp_len = match.b
                        diff = sent_exp_len - doc_exp_len
                    else:
                        diff = - (torch.where(context_input_ids[cur_len:] == sent_tokens_[1])[0][0].item() - 1)
                else:
                    if first_token_sent_decode != '' and first_token_sent_decode in first_token_doc_decode:
                        diff = -1
                        while True:
                            first_token_sent = sent_tokens_[diff+1]
                            first_token_sent_decode = tok.decode(first_token_sent)
                            if diff >= sent_len:
                                skip_this_sent = True
                                break
                            if first_token_sent_decode == first_token_doc_decode:
                                diff = diff + 1
                                break
                            if first_token_sent_decode in first_token_doc_decode:
                                pass
                            else:
                                break
                            first_token_doc_decode = first_token_doc_decode.replace(first_token_sent_decode, '', 1)
                            diff = diff + 1
                    else:
                        match = SequenceMatcher(None, sent_tokens_, context_input_ids[cur_len:cur_len+len(sent_tokens_)].tolist()).find_longest_match()
                        if match.size > 0:
                            sent_exp_len = match.a
                            doc_exp_len = match.b
                            diff = sent_exp_len - doc_exp_len
                        else:
                            skip_this_sent = True

            sent_len = sent_len - diff

            if skip_this_sent:
                continue
            cur_last_token_doc = context_input_ids[cur_len + sent_len - 1].item()
            cur_last_token_sent = sent_tokens_[-1]
            if cur_last_token_doc != cur_last_token_sent:
                if tok.decode(cur_last_token_sent) in tok.decode(cur_last_token_doc):
                    last_token_sent_in_last_token_doc = True
                else:
                    ## Sent가 포함할 수 도 있음.
                    if tok.decode(cur_last_token_doc) in tok.decode(cur_last_token_sent):
                        last_token_doc_in_last_token_sent = True
                    elif cur_last_token_doc == 535:
                        last_token_sent_in_last_token_doc = True
                        pass # This is very exceptional. But fine.
                    else:
                        token_indices = torch.where(context_input_ids[cur_len:cur_len+sent_len] == cur_last_token_sent)[0]
                        if token_indices.size()[0] > 0:
                            sent_len = token_indices[-1].item() + 1
                        else:                            
                            break

            period_positions_sent.append((cur_len, cur_len + sent_len))
            cur_len = cur_len + sent_len
        ctx['period_positions_sent'] = period_positions_sent
        ctx['has_answers_sent'] = [int(has_answer(answers, tok.decode(context_input_ids[s:e]))) for s, e in period_positions_sent]
    return qas

def main(args):

    num_workers = args.num_workers
    data_in = json.load(open(args.data_path))
    print(f"Loaded {len(data_in)} data from {args.data_path}")

    # from IPython import embed; embed()
    pool = multiprocessing.Pool(processes=num_workers)
    func = partial(get_sentence_position)
    # pool.map(func, data_in)
    values = []
    for return_values in tqdm(pool.imap(func, data_in), total=len(data_in), desc="> Processing"):
        values.append(return_values)
    pool.close()
    pool.join()
    with open(args.data_path.replace(".json", f"_sent_{N_CONTEXTS}_{TEXT_MAXLENGTH}_nltk_temp.json"), 'w') as f:
        json.dump(values, f, indent=4)
    # from IPython import embed; embed()

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, \
                        default="", \
                        help='Path of data in json (e.g., open_domain_data/nq/train.json')
    parser.add_argument('--num_workers', type=int, required=False, \
                        default=32)

    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())