# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import pickle
import torch
import argparse
import transformers
import numpy as np
import json
import csv
from pathlib import Path
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from collections import defaultdict

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import src.slurm
import src.util
import src.data
import src.evaluation
import src.model
from src.options import Options
from src.ResultTable import ResultTable
from copy import deepcopy

from sklearn import metrics

def evaluate(model, dataset, dataloader, tokenizer, opt):
    simple_tokenizer = src.evaluation.SimpleTokenizer()
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    if opt.write_crossattention_scores:
        assert opt.per_gpu_batch_size == 1, "Cross attention scores can only be written when batch size is 1"
        model.reset_score_storage()
        return_dict_in_generate = True
        pred_recall_dict_ca = {cut_off:[] for cut_off in opt.cut_offs}
    else:
        return_dict_in_generate = False
    total = 0
    exactmatch, recall, exactmatch_G2 = [], [], []
    recall_dict = {cut_off:[] for cut_off in opt.cut_offs}
    pred_recall_dict = {cut_off:[] for cut_off in opt.cut_offs}
    sent_has_answer_preds, sent_has_answer_labels = [], []
    num_passages_in_decoder = []

    if opt.write_results:
        # write_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        fw = open(f'outputs/{opt.n_contexts}_group{opt.eval_group}_{opt.select_gold_context}.txt', 'wt')
        print(f'Writing results to outputs/{opt.n_contexts}_group{opt.eval_group}_{opt.select_gold_context}.txt', 'wt')
        # fw = open(write_path / f'{opt.n_contexts}_group{opt.eval_group}_{opt.select_gold_context}.txt', 'a')
        # print(f'Writing results to {write_path / f"{opt.n_contexts}_group{opt.eval_group}_{opt.select_gold_context}.txt"}')


    if opt.eval_group == 'G':
        answer_contain_i_list = answer_contain_i_list_10
    elif opt.eval_group == 'G-2':
        answer_contain_i_list = answer_contain_i_list_1
    elif opt.eval_group == 'G-1':
        answer_contain_i_list = answer_contain_i_list_10 - answer_contain_i_list_1
    else:
        print("Eval group is set to all")
        answer_contain_i_list = set(np.arange(len(dataset.data)).tolist())

    ## Target group denotes group #1 unless the max voting option is True.
    group_len, len_target_group, did2dids = [], [], defaultdict(list)

    
    with torch.no_grad():
        for b_i, batch in enumerate(tqdm(dataloader)):
            (idx, labels, context_ids, context_mask, q_tokens, has_answers, sentence_spans_, batch_has_answers_sent) = batch
            if opt.select_gold_context:
                example = dataset.data[idx[0]]
                answers = example['answers']
                ctx_id_list = []
                for ctx_id, ctx in enumerate(example['ctxs'][:opt.n_contexts]):
                    if src.evaluation.has_answer(answers, ctx['title'] + ' ' + ctx['text'], simple_tokenizer):
                        ctx_id_list.append(ctx_id)
                        break

                # new_context_ids = context_ids[0, ctx_id_list]
                # context_ids = new_context_ids.view(1, len(ctx_id_list), -1)
                # new_context_mask = context_mask[0, ctx_id_list]
                # context_mask = new_context_mask.view(1, len(ctx_id_list), -1)

                new_rank_1 = context_ids[0, ctx_id]
                context_ids = new_rank_1.view(1, 1, -1)
                new_rank_1_mask = context_mask[0, ctx_id]
                context_mask = new_rank_1_mask.view(1, 1, -1)
                
            ####### Check if the answer is in the n contexts ########
            # example = dataset.data[idx[0]]
            # flag = False
            # answers = example['answers']
            # for ctx_id, ctx in enumerate(example['ctxs'][:opt.n_contexts]):
            #     if src.evaluation.has_answer(answers, ctx['title'] + ' ' + ctx['text'], simple_tokenizer):
            #         flag = True
            #         break
            # if not flag:
            #     print(f'{i}-th qas has no answer in the {opt.n_contexts} contexts')
            #     continue
            # answer_contain_i_list.append(i)

            # new_rank_1 = context_ids[0, ctx_id]
            # context_ids = new_rank_1.view(1, 1, -1)
            # new_rank_1_mask = context_mask[0, ctx_id]
            # context_mask = new_rank_1_mask.view(1, 1, -1)
            outputs = model.generate(input_ids=context_ids.cuda(),
                                     attention_mask=context_mask.cuda(),
                                     max_length=50,
                                     sentence_spans=sentence_spans_,
                                     return_dict_in_generate=return_dict_in_generate,
                                     output_attentions=True,)

            if return_dict_in_generate and opt.write_crossattention_scores:
                outputs, probs, sentence_preds, crossattention_scores = outputs
            else:
                outputs, probs, sentence_preds = outputs

            if sentence_preds is not None:
                for e_i, has_answers_sent_e in enumerate(batch_has_answers_sent):
                    for c_i, has_answers_sent_ctx in enumerate(has_answers_sent_e):
                        for s_i, label in enumerate(has_answers_sent_ctx):
                            sent_has_answer_preds.append(sentence_preds[e_i, c_i, s_i].item())
                            sent_has_answer_labels.append(label)

                        # if torch.sum(sentence_preds[e_i, c_i, s_i+1:]) != 0:
                        #     print('\n\n\n########################### wrong #############################\n\n')

            answer_array = has_answers.numpy()
            scores_array = probs.cpu().numpy()
            sorted_indices = np.argsort(scores_array, axis=1)[:, ::-1]
            pred_answer_array = np.take_along_axis(answer_array, sorted_indices, axis=1)
            if return_dict_in_generate and opt.write_crossattention_scores:
                ca_scores_array = crossattention_scores.cpu().numpy()
                ca_sorted_indices = np.argsort(ca_scores_array, axis=1)[:, ::-1]
                pred_answer_array_ca = np.take_along_axis(answer_array, ca_sorted_indices, axis=1)

            for cut_off in opt.cut_offs:
                recall_dict[cut_off].extend(answer_array[:, :cut_off].sum(1).astype('bool').tolist())
                pred_recall_dict[cut_off].extend(pred_answer_array[:, :cut_off].sum(1).astype('bool').tolist())
                if return_dict_in_generate and opt.write_crossattention_scores:
                    pred_recall_dict_ca[cut_off].extend(pred_answer_array_ca[:, :cut_off].sum(1).astype('bool').tolist())

            if opt.ce_mask_threshold > 0.0:
                num_passages_in_decoder.extend((probs > opt.ce_mask_threshold).sum(1).tolist())

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                example = dataset.data[idx[k]]
                if 'answers' in example:
                    score = src.evaluation.ems(ans, example['answers'])
                    exactmatch.append(score)
                    # if b_i in answer_contain_i_list_1:
                    #     exactmatch_G2.append(score)
                answers = example['answers']
                has_answer = check_has_answer(answers, example['ctxs'][:opt.n_contexts], simple_tokenizer)
                recall.append(has_answer)
                q_i = b_i * opt.per_gpu_batch_size + k
                if opt.write_results:
                    fw.write(f"{q_i}\t{ans}\t{1 if score else 0}\t{has_answer}\t{sorted_indices[k].tolist()}\t{pred_answer_array[k, :20].tolist()}\n")

                if opt.write_crossattention_scores:
                    for j in range(context_ids.size(1)):
                        example['ctxs'][j]['score'] = crossattention_scores[k, j].item()

                total += 1

            if (b_i + 1) % opt.eval_print_freq == 0:
                log = f'Process rank:{opt.global_rank}, {b_i+1} / {len(dataloader)}'
                if len(exactmatch) == 0:
                    log += '| no answer to compute scores'
                else:
                    log += f' | average = {np.mean(exactmatch):.3f}'
                    # log += f' | average (G-2) = {np.mean(exactmatch_G2):.3f}'

                logger.warning(log)

    # logger.warning(f'Average group length, target = {np.mean(group_len):.2f}, {np.mean(len_target_group):.2f}')
    logger.warning(f'Process rank:{opt.global_rank}, total {total} | EM = {100 * np.mean(exactmatch):.2f}')
    if opt.write_results:
        fw.close()

    if opt.is_distributed:
        torch.distributed.barrier()
    score, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    recall_dict = {f'Recall{k}':100 * src.util.weighted_average(np.mean(v), total, opt)[0] for k, v in recall_dict.items()}
    preds = {f'Recall{k}':100 * src.util.weighted_average(np.mean(v), total, opt)[0] for k, v in pred_recall_dict.items()}
    if return_dict_in_generate and opt.write_crossattention_scores:
        pred_recall_dict_ca = {f'CA_Recall{k}':100 * src.util.weighted_average(np.mean(v), total, opt)[0] for k, v in pred_recall_dict_ca.items()}
        preds.update(pred_recall_dict_ca)
    preds['EM'] = 100 * np.mean(score)

    if opt.ce_mask_threshold > 0.0 or opt.reduced_n > 0:
        num_passages_in_decoder, _ = src.util.weighted_average(np.mean(num_passages_in_decoder), total, opt)
        preds['Avg. #passages'] = np.round(num_passages_in_decoder, 2)

    if sent_has_answer_labels and sent_has_answer_preds:
        sent_has_answer_labels = np.array(sent_has_answer_labels)
        sent_has_answer_preds = np.array(sent_has_answer_preds)
        fpr, tpr, _ = metrics.roc_curve(sent_has_answer_labels, sent_has_answer_preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        acc = metrics.accuracy_score(sent_has_answer_labels, sent_has_answer_preds)
        auc, _ = src.util.weighted_average(auc, total, opt)
        acc, _ = src.util.weighted_average(acc, total, opt)
        preds['Sent. AUC'] = auc
        preds['Sent. Acc.'] = acc

        ## Select only positive sentences to evaluate
        true_indices_labels = np.where(sent_has_answer_labels == 1)[0]
        sent_has_answer_labels_1 = sent_has_answer_labels[true_indices_labels]
        sent_has_answer_preds_1 = sent_has_answer_preds[true_indices_labels]
        acc_1 = metrics.accuracy_score(sent_has_answer_labels_1, sent_has_answer_preds_1)
        acc_1, _ = src.util.weighted_average(acc_1, total, opt)
        preds['Sent. Acc. (pos)'] = acc_1

    return score, total, recall_dict, preds

def check_has_answer(answers, examples, simple_tokenizer):
    flag = False
    for ctx_id, ctx in enumerate(examples):
        if src.evaluation.has_answer(answers, ctx['title'] + ' ' + ctx['text'], simple_tokenizer):
            flag = True
            break
    return flag

def most_frequent(ans_list):
    return max(set(ans_list), key = ans_list.count)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of the config file', default=None) ## e.g., checkpoints/nq_reader_qbos/checkpoint/best_dev
    args, remaining_args = parser.parse_known_args()
    
    options = Options()
    options.add_reader_options()
    options.add_eval_options()

    opt_path = Path('/'.join(args.model_path.split('/')[:2])) / "options.json"

    if opt_path:
        with open(opt_path, 'r') as f:
            opt_dict = json.load(f)
        loaded_opt = argparse.Namespace(**opt_dict)
    else:
        print("> No options file found")
        exit(1)

    options.parser.set_defaults(**vars(loaded_opt))

    # Parse command line arguments
    # Any command line argument will overwrite the one from JSON file
    opt = options.parse(remaining_args)
    opt.model_path = args.model_path

    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)
    
    dir_path = Path(opt.checkpoint_dir)/opt.name
    directory_exists = dir_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    dir_path.mkdir(parents=True, exist_ok=True)
    if opt.write_results:
        (dir_path / 'test_results').mkdir(parents=True, exist_ok=True)
    logger = src.util.init_logger(opt.is_main, opt.is_distributed, Path(opt.checkpoint_dir) / opt.name / 'run.log')
    
    if not directory_exists and opt.is_main:
        options.print_options(opt)
    
    tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base', return_dict=False)

    collator_function = src.data.Collator(opt.text_maxlength,
                                           tokenizer,
                                           sep_q_p=opt.sep_q_p,
                                           soft_prompts=opt.soft_prompts,
                                           extra_question=opt.extra_question)
    eval_examples = src.data.load_data(
        opt.eval_data, 
        global_rank=opt.global_rank, #use the global rank and world size attibutes to split the eval set on multiple gpus
        world_size=opt.world_size
    )
    eval_dataset = src.data.Dataset(eval_examples, opt, is_eval=True)    

    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        batch_size=opt.per_gpu_batch_size,
        num_workers=1,
        collate_fn=collator_function
    )
    
    # t5 = transformers.T5ForConditionalGeneration.from_pretrained('t5-base')
    # model = src.model.FiDT5(t5.config, opt)
    # model.load_t5(t5.state_dict())
    
    model_class = src.model.FiDT5
    model = model_class.from_pretrained(opt.model_path, opt)
    model = model.to(opt.device)
    
    logger.info("Start eval")
    exactmatch, total, recall_dict, preds = evaluate(model, eval_dataset, eval_dataloader, tokenizer, opt)

    logger.info(f'EM {100*exactmatch:.2f}, Total number of example {total}')

    # recall_dict = {f'Recall{k}':100 * np.mean(v) for k, v in recall_dict.items()}
    # preds = {f'Recall{k}':100 * np.mean(v) for k, v in preds.items()}
    evaluation_table = ResultTable(table_name='Eval Result', header=list(preds.keys()))
    evaluation_table.add_row('orig.', recall_dict)
    evaluation_table.add_row('pred.', preds)

    logger.info(evaluation_table.to_string())

    if opt.write_results and opt.is_main:
        glob_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        write_path = Path(opt.checkpoint_dir) / opt.name / 'final_output.txt'
        src.util.write_output(glob_path, write_path) 
    if opt.write_crossattention_scores:
        src.util.save_distributed_dataset(eval_dataset.data, opt)