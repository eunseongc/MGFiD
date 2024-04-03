# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def add_optim_options(self):
        self.parser.add_argument('--warmup_steps', type=int, default=0)
        self.parser.add_argument('--total_steps', type=int, default=64000)
        self.parser.add_argument('--scheduler_steps', type=int, default=None, 
                        help='total number of step for the scheduler, if None then scheduler_total_step = total_step')
        self.parser.add_argument('--accumulation_steps', type=int, default=1)
        self.parser.add_argument('--use_acc', action='store_true', help='')
        self.parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        self.parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
        self.parser.add_argument('--optim', type=str, default='adam')
        self.parser.add_argument('--scheduler', type=str, default='fixed')
        self.parser.add_argument('--weight_decay', type=float, default=0.1)
        self.parser.add_argument('--fixed_lr', action='store_true')
        self.parser.add_argument('--wandb_tag', type=str, default=None)
        self.parser.add_argument('--n_neg_samples', type=int, default=2)

    def add_eval_options(self):
        self.parser.add_argument('--eval_group', type=str, default=None, help='choose a question group to evaluate')
        self.parser.add_argument('--select_gold_context', action='store_true', help='select gold passage and use it only')
        self.parser.add_argument('--use_sep_embs', action='store_true', help='use passage seperately')
        self.parser.add_argument('--graph_scorer', type=str, default='dpr', help='scorer used to build corpus graph')
        self.parser.add_argument('--use_graph_grouping', action='store_true', help='use corpus graph to group passages')
        self.parser.add_argument('--use_extra_graph_context', action='store_true', help='use extra context using corpus graph')
        self.parser.add_argument('--write_results', action='store_true', help='save results')
        self.parser.add_argument('--write_crossattention_scores', action='store_true', 
                        help='save dataset with cross-attention scores')
        self.parser.add_argument('--cut_offs', type=list, default=[1, 2, 3, 5, 10, 20], help='cutoffs for recall')


    def add_reader_options(self):
        self.parser.add_argument('--train_data', type=str, default=None, help='path of train data')
        self.parser.add_argument('--eval_data', type=str, default=None, help='path of eval data')
        self.parser.add_argument('--cls_train_data', type=str, default=None, help='path of eval data')
        self.parser.add_argument('--cls_eval_data', type=str, default=None, help='path of eval data')
        self.parser.add_argument('--model_name', type=str, default='t5-base')
        self.parser.add_argument('--model_class', type=str, default='FiDT5', help='model class, e.g., FiDT5, FiD_encoder')
        self.parser.add_argument('--use_checkpoint', action='store_true', help='use checkpoint in the encoder')
        self.parser.add_argument('--text_maxlength', type=int, default=192, 
                        help='maximum number of tokens in text segments (question+passage)')
        self.parser.add_argument('--answer_maxlength', type=int, default=-1, 
                        help='maximum number of tokens used to train the model, no truncation if -1')
        self.parser.add_argument('--no_title', action='store_true', 
                        help='article titles not included in passages')
        self.parser.add_argument('--n_contexts', type=int, default=20)
        self.parser.add_argument('--ctx_anno', type=str, default='has_answer', help="e.g., has_answer, mytho")
        self.parser.add_argument('--n_qas', type=int, default=None)
        self.parser.add_argument('--n_extra_contexts', type=int, default=10)
        self.parser.add_argument('--n_max_groups', type=int, default=1)
        self.parser.add_argument('--use_major_group', action='store_true', default=False)
        self.parser.add_argument('--use_max_voting', action='store_true', default=False)
        self.parser.add_argument('--use_max_first_prob', action='store_true', default=False)
        self.parser.add_argument('--use_recursive_graph', action='store_true', default=False)
        self.parser.add_argument('--use_half_extra', action='store_true', default=False)
        self.parser.add_argument('--use_group_dpr', action='store_true', default=False)
        self.parser.add_argument('--ret_path', type=str, default='pretrained_models/nq_retriever')
        self.parser.add_argument('--enc_weight_path', type=str, default=None)
        self.parser.add_argument('--sep_q_p', type=str, default=None, help="e.g., tokens, embs, embs_mean")
        self.parser.add_argument('--tokens_k', type=str, default=None, help="e.g., f1, m2, l3, f4")  
        self.parser.add_argument('--soft_prompts', type=int, default=None, help="e.g., 1, 2, 3, 4")
        self.parser.add_argument('--freeze_model', type=str, default=None, help="whether to freeze models, e.g., encoder, decoder, hlatr, all")
        self.parser.add_argument('--pointwise', type=str, default=None, help="whether to use pointwise loss in embeddings")
        self.parser.add_argument('--lamb', type=float, default=0.01, help="lambda for pred loss")
        self.parser.add_argument('--extra_question', type=str, default=False, help="embs, ids for the question")
        self.parser.add_argument('--replace_bos_token', action='store_true', default=False)
        self.parser.add_argument('--use_conv_dropout', action='store_true', default=False)
        self.parser.add_argument('--use_conv_relu', action='store_true', default=False)
        self.parser.add_argument('--use_local_interaction', action='store_true', default=False)
        self.parser.add_argument('--use_late_decoding', action='store_true', default=False)
        self.parser.add_argument('--use_probs_bias', action='store_true', default=False)
        self.parser.add_argument('--ce_loss_type', type=str, default=None, help="type for ce loss, e.g., pair, point, list")
        self.parser.add_argument('--ce_loss_weight', type=float, default=0.0, help="ce_loss_weight")
        self.parser.add_argument('--ce_topk', type=int, default=None, help="Calculate ce loss for only ce_topk contexts")
        self.parser.add_argument('--sce_loss_type', type=str, default=None, help="type for ce loss, e.g., pair, point, list")
        self.parser.add_argument('--sce_loss_weight', type=float, default=0.0, help="sce_loss_weight")
        self.parser.add_argument('--pooler_activation', type=str, default='tanh', help="type of pooler activation")
        self.parser.add_argument('--use_sent_classifier', action='store_true', default=False, help="whether to use sentence classifier")
        self.parser.add_argument('--cls_norm_type', type=str, default=None, help="norm for classifier")
        self.parser.add_argument('--ce_mask_threshold', type=float, default=0.0, help="ce_mask_threshold")
        self.parser.add_argument('--sce_mask_threshold', type=float, default=0.0, help="sce_mask_threshold")
        self.parser.add_argument('--reduced_n', type=int, default=0, help="Reduced number of passages for decoder must be less than n_contexts")
        self.parser.add_argument('--passage_pooling', type=str, default=None, help="passage_pooling")
        self.parser.add_argument('--sentence_pooling', type=str, default=None, help="sentence_pooling")
        self.parser.add_argument('--class_weight', type=str, default=None, help="class_weight")
        self.parser.add_argument('--sce_loss_reduction', type=str, default='mean', help="loss_reduction") ## If sum, it averages over all sentences in a batch.
        self.parser.add_argument('--hlatr_loss_weight', type=float, default=0.0, help="hlatr_loss_weight")
        self.parser.add_argument('--use_hlatr', action='store_true', default=False)
        self.parser.add_argument('--use_sent_summary', action='store_true', default=False)
        self.parser.add_argument('--sent_summary_tf', action='store_true', default=False)
        self.parser.add_argument('--sent_summary_pool', type=str, default=None, help="Pooling method for sentence summary")
        self.parser.add_argument('--summary_bos_option', type=str, default=None, help="how to use summary token, must use sent_summary")
        self.parser.add_argument('--sent_highlight', type=str, default=None, help="high lighting pos. pred. sentences")
        self.parser.add_argument('--use_decode_num_sent', action='store_true', default=False)
        self.parser.add_argument('--save_output', action='store_true', default=False)
        self.parser.add_argument('--use_rank_embedding', action='store_true', default=False)
        self.parser.add_argument('--sce_loss_fct', type=str, default="CE", help="sce_loss_fct")
        self.parser.add_argument('--focal_alpha', type=float, default=0.25, help="focal_alpha")
        


    def add_retriever_options(self):
        self.parser.add_argument('--train_data', type=str, default=None, help='path of train data')
        self.parser.add_argument('--eval_data', type=str, default=None, help='path of eval data')
        self.parser.add_argument('--indexing_dimension', type=int, default=768)
        self.parser.add_argument('--no_projection', action='store_true', 
                        help='No addition Linear layer and layernorm, only works if indexing size equals 768')
        self.parser.add_argument('--question_maxlength', type=int, default=40, 
                        help='maximum number of tokens in questions')
        self.parser.add_argument('--passage_maxlength', type=int, default=200, 
                        help='maximum number of tokens in passages')
        self.parser.add_argument('--no_question_mask', action='store_true')
        self.parser.add_argument('--no_passage_mask', action='store_true')
        self.parser.add_argument('--extract_cls', action='store_true')
        self.parser.add_argument('--no_title', action='store_true', 
                        help='article titles not included in passages')
        self.parser.add_argument('--n_contexts', type=int, default=20)


    def initialize_parser(self):
        # basic parameters
        self.parser.add_argument('--name', type=str, default='test', help='name of the experiment')
        self.parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='models are saved here')
        self.parser.add_argument('--model_path', type=str, default=None, help='path for retraining')

        # dataset parameters
        self.parser.add_argument("--per_gpu_batch_size", default=1, type=int, 
                        help="Batch size per GPU/CPU for training.")
        self.parser.add_argument('--maxload', type=int, default=-1)

        self.parser.add_argument("--local-rank", type=int, default=-1,
                        help="For distributed training: local_rank")
        self.parser.add_argument("--main_port", type=int, default=-1,
                        help="Main port (for multi-node SLURM jobs)")
        self.parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")
        self.parser.add_argument('--num_workers', type=int, default=10, help="")
        # training parameters
        self.parser.add_argument('--print_freq', type=int, default=2000,
                        help='print training loss every <print_freq> steps')
        self.parser.add_argument('--eval_freq', type=int, default=500,
                        help='evaluate model every <eval_freq> steps during training')
        self.parser.add_argument('--eval_from', type=int, default=0)
        self.parser.add_argument('--save_freq', type=int, default=5000,
                        help='save model every <save_freq> steps during training')
        self.parser.add_argument('--eval_print_freq', type=int, default=1000,
                        help='print intermdiate results of evaluation every <eval_print_freq> steps')


    def print_options(self, opt):
        message = '\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default_value = self.parser.get_default(k)
            if v != default_value:
                comment = f'\t(default: {default_value})'
            message += f'{str(k):>30}: {str(v):<40}{comment}\n'

        expr_dir = Path(opt.checkpoint_dir)/ opt.name
        model_dir = expr_dir / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(expr_dir/'opt.log', 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

        logger.info(message)

    def parse(self, args=None):
        if args is None:
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)
            
        return opt


def get_options(use_reader=False,
                use_retriever=False,
                use_optim=False,
                use_eval=False):
    options = Options()
    if use_reader:
        options.add_reader_options()
    if use_retriever:
        options.add_retriever_options()
    if use_optim:
        options.add_optim_options()
    if use_eval:
        options.add_eval_options()
    return options.parse()