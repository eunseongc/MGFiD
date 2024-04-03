# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import json
import time
import sys
import torch
import transformers
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from src.options import Options
from tqdm import tqdm

import src.slurm
import src.util
import src.evaluation
import src.data
import src.model
from src.ResultTable import ResultTable

from scipy.sparse import csc_matrix
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

import wandb
os.environ['WANDB_API_KEY'] = "01e2c25c291441f9d7a138a24249bd2b5e39eb8f"

from sklearn import metrics

def train(model, optimizer, scheduler, checkpoint_step, train_dataset, eval_dataset, opt, collator, best_dev_em, checkpoint_path):
    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir)/opt.name)
        except:
            tb_logger = None

    torch.manual_seed(opt.global_rank + opt.seed) #different seed for different sampling depending on global_rank
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=opt.num_workers,
        collate_fn=collator
    )
    step = 0
    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    pbar = tqdm(total=opt.total_steps, desc=f"> training", dynamic_ncols=True)

    while step < opt.total_steps + 1:
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            step += 1
            pbar.update(1)
            if step <= checkpoint_step:
                continue
            (idx, labels, context_ids, context_mask, q_tokens, has_answers, sentence_spans_, has_answers_sent_) = batch

            outputs = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=labels.cuda(),
                q_tokens=q_tokens,
                has_answers=has_answers,
                sentence_spans=sentence_spans_,
                has_answers_sent=has_answers_sent_,
                output_attentions=True,
                step=step,
            )
            train_loss = outputs[0]
            
            pred_loss = None
            if opt.pointwise is not None:
                if opt.pointwise == 'v1':
                    # v1
                    col_indices = torch.where((labels!=-100))
                    cols = labels[col_indices]
                    logits = outputs.enc_pred_embs[col_indices] @ model.shared.weight.T
                    loss_fct = CrossEntropyLoss()
                    pred_loss = loss_fct(logits, cols.to(logits.device))
                # v2
                elif opt.pointwise == 'v2':
                    col_indices = torch.where((labels!=-100) & (labels!=1))
                    cols = labels[col_indices]
                    vals = torch.ones_like(cols).flatten()
                    rows_temp = torch.arange(labels.shape[0]).repeat_interleave(labels.shape[1]).view(labels.shape[0], -1) 
                    rows = rows_temp[col_indices]
                    label_matrix = csc_matrix((vals, (rows, cols)), shape=(labels.shape[0], model.shared.num_embeddings))
                    
                    logits = (outputs.enc_pred_embs @ model.shared.weight.T).max(1)[0]
                    logits = -(F.softmax(logits, dim=-1)+1e-10).log()
                    label_matrix = torch.Tensor(label_matrix.toarray()).to(logits.device)
                    pred_loss = (label_matrix * logits).sum(dim=1).mean()                    
                train_loss = train_loss + opt.lamb * pred_loss
            
            if torch.isnan(train_loss):
                print(step)
                from IPython import embed; embed()

            train_loss = train_loss / opt.accumulation_steps
            if step % opt.print_freq == 0:
                print(train_loss)

            train_loss.backward()
            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)

            if pred_loss is not None:  
                wandb_log = {"step": step, "train_loss": train_loss.item(), "pred_loss": pred_loss.item(),
                         "lr": scheduler.get_last_lr()[0]}
            else:
                wandb_log = {"step": step, "train_loss": train_loss.item(),
                             "lr": scheduler.get_last_lr()[0]}

            curr_loss += train_loss.item()

            if step >= opt.eval_from and step % opt.eval_freq == 0:
                logger.info(f'> Start eval ..')
                dev_em, recalls, n_samples = evaluate(model, eval_dataset, tokenizer, collator, opt)
                wandb_log['EM (dev)'] = 100*dev_em
                model.train()
                if opt.is_main:
                    if dev_em > best_dev_em:
                        best_dev_em = dev_em
                        src.util.save(model, optimizer, scheduler, step, best_dev_em,
                                  opt, checkpoint_path, 'best_dev')
                    evaluation_table = ResultTable(table_name=f'Valid Result ({n_samples})', header=list(recalls[1].keys()))
                    evaluation_table.add_row('orig.', recalls[0])
                    evaluation_table.add_row('pred.', recalls[1])

                    log = f"{step} / {opt.total_steps} |"
                    log += f"train loss: {curr_loss/opt.eval_freq:.3f} |"
                    # log += f"evaluation (EM): {100*dev_em:.2f} |"
                    # log += f"Recall@1: {100 * np.mean(recalls[0]):.2f} |Recall@5: {100 * np.mean(recalls[1]):.2f} |"
                    # log += f"Pred Recall@1: {100 * np.mean(recalls[2]):.2f} |Pred Recall@5: {100 * np.mean(recalls[3]):.2f} |"
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log)
                    logger.info(evaluation_table.to_string())
                    if tb_logger is not None:
                        tb_logger.add_scalar("Evaluation", dev_em, step)
                        tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), step)
                    curr_loss = 0.

            # wandb.log(wandb_log)
            if opt.is_main and step % opt.eval_freq == 0:
                src.util.save(model, optimizer, scheduler, step, best_dev_em,
                          opt, checkpoint_path, f"step-{step}")
            if step > opt.total_steps:
                pbar.close()
                break

def evaluate(model, dataset, tokenizer, collator, opt):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=10,
        collate_fn=collator
    )
    model.eval()
    total = 0
    exactmatch = []
    num_passages_in_decoder = []
    sent_has_answer_preds, sent_has_answer_labels, sent_auc_acc = [], [], None
    model = model.module if hasattr(model, "module") else model
    recall_dict = {cut_off:[] for cut_off in opt.cut_offs}
    pred_recall_dict = {cut_off:[] for cut_off in opt.cut_offs}
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, dynamic_ncols=True)):
            (idx, labels, context_ids, context_mask, q_tokens, has_answers, sentence_spans_, batch_has_answers_sent) = batch
            outputs, probs, sentence_preds = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=20,
                sentence_spans=sentence_spans_,
                output_attentions=True,
            )

            if opt.ce_mask_threshold > 0.0:
                num_passages_in_decoder.extend((probs > opt.ce_mask_threshold).sum(1).tolist())
            elif opt.reduced_n > 0:
                num_passages_in_decoder.extend((probs > 0).sum(1).tolist())

            answer_array = has_answers.numpy()
            scores_array = probs.cpu().numpy()

            sorted_indices = np.argsort(scores_array, axis=1)[:, ::-1]
            pred_answer_array = np.take_along_axis(answer_array, sorted_indices, axis=1)
            for cut_off in opt.cut_offs:
                recall_dict[cut_off].extend(answer_array[:, :cut_off].sum(1).astype('bool').tolist())
                pred_recall_dict[cut_off].extend(pred_answer_array[:, :cut_off].sum(1).astype('bool').tolist())

            if sentence_preds is not None:
                for e_i, has_answers_sent_e in enumerate(batch_has_answers_sent):
                    for c_i, has_answers_sent_ctx in enumerate(has_answers_sent_e):
                        for s_i, label in enumerate(has_answers_sent_ctx):
                            sent_has_answer_preds.append(sentence_preds[e_i, c_i, s_i].item())
                            sent_has_answer_labels.append(label)

            for k, o in enumerate(outputs):
                if model.encoder.use_decode_num_sent:
                    o = o[3:]
                ans = tokenizer.decode(o, skip_special_tokens=True)
                gold = dataset.get_example(idx[k])['answers']
                score = src.evaluation.ems(ans, gold)
                total += 1
                exactmatch.append(score)
    n_samples = len(recall_dict[opt.cut_offs[0]])

    recall_dict = {f'Recall{k}':100 * src.util.weighted_average(np.mean(v), total, opt)[0] for k, v in recall_dict.items()}
    preds = {f'Recall{k}':100 * src.util.weighted_average(np.mean(v), total, opt)[0] for k, v in pred_recall_dict.items()}

    exactmatch, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    preds['EM'] = 100*exactmatch
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

    recalls = [recall_dict, preds]

    return exactmatch, recalls, n_samples

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    options.add_eval_options()
    opt = options.parse()

    checkpoint_path = Path(opt.checkpoint_dir)/opt.name
    opt_path = checkpoint_path / "options.json"

    if opt.local_rank == 0 or opt.local_rank == -1:
        checkpoint_exists = checkpoint_path.exists()
        if 'temp' in str(checkpoint_path):
            checkpoint_exists = False
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        if checkpoint_exists and opt.model_path is None:
            exit("> Remove the folder unless you are training the model continually")
        print(f"Save the options in {opt_path}")
        with open(opt_path, 'w') as outf:
            json.dump(vars(opt), outf, indent=4)
    else:
        checkpoint_exists = False
        if opt.model_path is not None:
            checkpoint_exists = True

    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()
    if opt.is_distributed:
        torch.distributed.barrier()

    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )

    model_name = opt.model_name
    if opt.model_class == 'FiDT5':
        model_class = src.model.FiDT5
    elif opt.model_class == 'FiD_encoder':
        model_class = src.model.FiD_encoder

    #load data
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    collator = src.data.Collator(opt.text_maxlength,
                                 tokenizer,
                                 sep_q_p=opt.sep_q_p,
                                 soft_prompts=opt.soft_prompts,
                                 extra_question=opt.extra_question,
                                 answer_maxlength=opt.answer_maxlength)

    # use golbal rank and world size to split the eval set on multiple gpus
    train_examples = src.data.load_data(
        opt.train_data,
        global_rank=opt.global_rank, 
        world_size=opt.world_size,
        n_qas = opt.n_qas)
    train_dataset = src.data.Dataset(train_examples, opt)

    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = src.data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size)
    eval_dataset = src.data.Dataset(eval_examples, opt, is_eval=True)
    
    # Define the FiD model
    if not checkpoint_exists and opt.model_path is None:
        t5 = model_class.__base__.from_pretrained(model_name, opt)
        model = model_class(t5.config, opt) 
        # t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        # model = src.model.FiDT5(t5.config, opt)
        if opt.enc_weight_path is not None: ## T5를 바꿔버리는 거니까 된다된다..!
            logger.info(f"Loading encoder weights from {opt.enc_weight_path}")
            t5.load_state_dict(torch.load(opt.enc_weight_path), strict=False)
        model.load_t5(t5.state_dict())
        model = model.to(opt.local_rank)
        optimizer, scheduler = src.util.set_optim(opt, model)
        step, best_dev_em = 0, 0.0

    elif opt.model_path is None:
        load_path = checkpoint_path / 'checkpoint' / 'latest'
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, load_path, opt, reset_params=False)
        logger.info(f"Model loaded from {load_path}")

    else:
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, opt.model_path, opt, reset_params=True)
        logger.info(f"Model loaded from {opt.model_path}")

    model.set_checkpoint(opt.use_checkpoint)
    if opt.model_class == 'FiD_encoder' and opt.enc_weight_path is not None:
        model.load_state_dict(torch.load(opt.enc_weight_path), strict=False)
        
    if opt.freeze_model is not None:
        print(f"> Option freeze_model is {opt.freeze_model}")
        if opt.freeze_model in ['encoder', 'all']:
            print(f">> freezing params in the encoder")
            for param in model.encoder.parameters():
                param.requires_grad = False
            if opt.model_class == 'FiD_encoder':
                for param in model.pooler.parameters():
                    param.requires_grad = False
                for param in model.classifier.parameters():
                    param.requires_grad = False
            
        if opt.freeze_model in ['decoder', 'all']:
            print(f"> Freezing params in the decoder")
            for param in model.decoder.parameters():
                param.requires_grad = False
        ## HLATR도?
        # print(f"> Melting the param of the embedding")
        # for param in model.shared.parameters():
        #     param.requires_grad = True

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False)
    logger.info(opt)    
    logger.info("> Start training")
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_em,
        checkpoint_path
    )
