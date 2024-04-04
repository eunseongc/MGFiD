# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import types
import torch
import torch.sparse as sparse
import transformers
import torch.nn.functional as F
import numpy as np

import torch.nn as nn


from kornia.losses import focal
from .t5_model import T5Pooler, HLATR_reranker, CustomT5ForConditionalGeneration


class FiDT5(CustomT5ForConditionalGeneration):
    def __init__(self, config, opt=None):
        super().__init__(config, opt)
        self.opt = opt

        self.wrap_encoder(opt)
        self.sep_q_p = opt.sep_q_p
        self.tokenizer = transformers.T5Tokenizer.from_pretrained(opt.model_name)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        
        if kwargs.get('step') is not None:
            del kwargs['step']

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        return outputs 

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, max_length, return_dict_in_generate=False, **kwargs):
        self.encoder.n_passages = input_ids.size(1)
        encoder_outputs, _, _, probs, sentence_probs, sent_summary_bos = self.encoder(input_ids=input_ids.view(input_ids.size(0), -1),
                                                                             attention_mask=attention_mask.view(attention_mask.size(0), -1),
                                                                             **kwargs)
        kwargs['sent_summary_bos'] = sent_summary_bos
        kwargs['return_dict_in_generate'] = return_dict_in_generate
        
        outputs = super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            encoder_outputs=encoder_outputs,
            max_length=max_length,
            **kwargs
        )
        if return_dict_in_generate:
            cross_attentions = outputs.cross_attentions
            outputs = outputs.sequences
            crossattention_scores = self.get_crossattention_scores_my(outputs, cross_attentions, attention_mask, all_gen_tokens=False)
            if kwargs['output_attentions']:
                return outputs, probs, sentence_probs, crossattention_scores
        
        return outputs, probs, sentence_probs

    def wrap_encoder(self, opt, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, opt=opt, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict, strict=False)
        self.wrap_encoder(self.opt)

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None

    def get_crossattention_scores_my(self, sequences, cross_attentions, attention_mask, all_gen_tokens=False):
        """
        sequences: torch.tensor (bsz, #gen tokens)
        cross_attentions: list(#gen tokens) of list(#layers) of (bsz, n_heads, 1, n_passages * text_maxlength)
        attention_mask: torch.tensor (bsz, n_passages, text_maxlength)
        """
        
        # Assuming that the cross_attentions are arranged as a list of [gen tokens][layers], where each element is
        # a tensor of shape (bsz, n_heads, 1, n_passages * text_maxlength)
        
        cross_attentions_per_gen_token = [torch.stack(gen_tok_cross_attention) for gen_tok_cross_attention in cross_attentions]
        cross_attention_scores_all = torch.stack(cross_attentions_per_gen_token)
        
        bsz, n_passages,text_maxlength = attention_mask.size()
        n_gen_tokens, n_layers, bsz, n_heads, n_seq, _ = cross_attention_scores_all.size()
        n_heads = cross_attention_scores_all.size(3)
        n_layers = cross_attention_scores_all.size(1)
        ## cross_attention_scores_all: (#gen_tokens, #layers, bsz (which is 1 always), #heads, 1, n_passages * text_maxlength)
        ## if considering all gen tokens, then sum over gen tokens
        if all_gen_tokens:
            cross_attention_scores_all = cross_attention_scores_all.sum(dim=0)
        else:
            ## Else, consider only the first gen token
            cross_attention_scores_all = cross_attention_scores_all[0]
        scores = cross_attention_scores_all.view(bsz, n_heads, n_layers, n_passages, -1)
        scores = scores.masked_fill(~attention_mask[:, None, None], 0.)
        scores = scores.sum(dim=[1, 2, 4])
        ntokens = attention_mask.sum(dim=[2]) * n_layers * n_heads
        scores = scores/ntokens

        return scores

    def get_crossattention_scores(self, context_mask):
        """
        Cross-attention scores are aggregated to obtain a single scalar per
        passage. This scalar can be seen as a similarity score between the
        question and the input passage. It is obtained by averaging the
        cross-attention scores obtained on the first decoded token over heads,
        layers, and tokens of the input passage.

        More details in Distilling Knowledge from Reader to Retriever:
        https://arxiv.org/abs/2012.04584.
        """
        scores = []
        n_passages = context_mask.size(1)
        for mod in self.decoder.block:
            scores.append(mod.layer[1].EncDecAttention.score_storage)
        scores = torch.cat(scores, dim=2)
        bsz, n_heads, n_layers, _ = scores.size()
        # batch_size, n_head, n_layers, n_passages, text_maxlength
        scores = scores.view(bsz, n_heads, n_layers, n_passages, -1)
        scores = scores.masked_fill(~context_mask[:, None, None], 0.)
        scores = scores.sum(dim=[1, 2, 4])
        ntokens = context_mask.sum(dim=[2]) * n_layers * n_heads
        scores = scores/ntokens
        return scores

class EncoderWrapper(nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """
    def __init__(self, encoder, opt=None, use_checkpoint=False):
        super().__init__()

        self.main_input_name = 'input_ids'
        self.encoder = encoder
        self.config = encoder.config
        self.d_model = self.config.d_model
        ## Parameters from options
        self.sep_q_p = opt.sep_q_p
        self.extra_question = opt.extra_question
        self.use_local_interaction = opt.use_local_interaction
        self.tokens_k = opt.tokens_k

        self.opt = opt
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        #### FOR CROSS_ENCODER ####
        ##===================================##
        self.ce_loss_weight = self.opt.ce_loss_weight
        self.sce_loss_weight = self.opt.sce_loss_weight
        self.ce_topk = self.opt.ce_topk
        self.ce_mask_threshold = self.opt.ce_mask_threshold
        self.sce_mask_threshold = self.opt.sce_mask_threshold
        self.reduced_n = self.opt.reduced_n
        self.ce_loss_type = self.opt.ce_loss_type
        self.sce_loss_type = self.opt.sce_loss_type
        self.use_sent_classifier = self.opt.use_sent_classifier

        if self.ce_loss_weight > 0 or self.sce_loss_weight > 0:
            self.pooler = T5Pooler(self.config, opt)
            self.dropout = nn.Dropout(self.config.dropout_rate)

        if self.ce_loss_weight > 0:    
            self.classifier = nn.Linear(self.config.hidden_size, 2)
            self.sent_classifier = self.classifier
        
        if self.use_sent_classifier:
            self.sent_classifier = nn.Linear(self.config.hidden_size, 2)

        self.class_weight = None
        if opt.class_weight is None or eval(opt.class_weight) is None:
            pass
        else:
            self.class_weight = torch.tensor(eval(opt.class_weight)).float()

        self.use_rank_embedding = opt.use_rank_embedding
        if self.use_rank_embedding:
            self.rank_embedding = nn.Embedding(100, self.d_model)
        self.use_sent_summary = opt.use_sent_summary
        self.sent_summary_pool = opt.sent_summary_pool
        self.sent_summary_tf = opt.sent_summary_tf
        ### 하나만 걸려라.
        self.sent_highlight = opt.sent_highlight
        if self.sent_highlight is not None:
            ## Highlight only sentences which are selected
            ## Therefore, define 1 vector to add to sentence embeddings
            self.sent_highlight_emb = nn.Embedding(2, self.d_model)
            self.sent_highlight_emb.weight.data.normal_(mean=0.0, std=self.config.initializer_factor * 1.0)
            self.sent_highlight_emb.weight.data[0] = 0
        self.use_decode_num_sent = opt.use_decode_num_sent
        self.sce_loss_reduction = opt.sce_loss_reduction
        if self.opt.sce_loss_fct == 'CE':
            self.sce_loss_fct = nn.CrossEntropyLoss(reduction=self.sce_loss_reduction)
        elif self.opt.sce_loss_fct == 'focal':
            self.sce_loss_fct = focal.FocalLoss(alpha=opt.focal_alpha, reduction=self.sce_loss_reduction)

        ##===================================##

        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    ## True encoder 
    def forward(self, input_ids=None, attention_mask=None, q_tokens=None, has_answers=None, has_answers_sent=None, sentence_spans=None, **kwargs):
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz*self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz*self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        n_passages = self.n_passages

        outputs.last_hidden_state = outputs.last_hidden_state.contiguous().view(bsz, n_passages*passage_length, -1)
        attention_mask = attention_mask.contiguous().view(bsz, n_passages*passage_length)
        hidden_states = outputs.last_hidden_state
        # hidden_states = self.relu(hidden_states) ## Adopting ReLU to avoid negative values

        cross_encoder_loss, sent_loss = 0, 0
        probs = torch.zeros((bsz, n_passages), device=hidden_states.device)
        sentence_logits, sentence_preds = None, None

        # > Cross-encoder
        ## =============================== ##
        ############################ For sentence classifier
        if self.sce_loss_weight > 0:
            if self.ce_topk is not None:
                from IPython import embed; embed(0)
                hidden_states = hidden_states.view(bsz, n_passages, -1, self.config.d_model)
            else:
                pooled_output = self.pooler(hidden_states.contiguous().view(bsz*n_passages, -1, self.config.d_model),
                                            attention_mask.contiguous().view(bsz*n_passages, -1),
                                            sentence_spans=sentence_spans,
                                            passage_pooling=self.opt.passage_pooling,
                                            sentence_pooling=self.opt.sentence_pooling)
            pooled_output, sentence_embeddings, sentence_mask = pooled_output
            if sentence_mask is None:
                ## IF THERE IS NO SENTENCE IN THE BATCH BECAUSE OF MAX_SEQ_LENGTH, THEN SKIP
                pass
            else:
                sentence_embeddings = self.dropout(sentence_embeddings)
                sentence_logits = self.sent_classifier(sentence_embeddings)
                sentence_preds = torch.argmax(sentence_logits, dim=-1).view(bsz, n_passages, -1) ## This only covers pointwise
                if has_answers_sent is not None:
                    if self.sce_loss_type == 'point':
                        sentence_logits = sentence_logits.view(bsz, n_passages, -1, 2)
                        if self.class_weight is not None:
                            sce_loss_fct = nn.CrossEntropyLoss(reduction=self.sce_loss_reduction, weight=self.class_weight.to(sentence_logits.device))
                        else:
                            sce_loss_fct = self.sce_loss_fct
                        sent_cnt = 0
                        passage_cnt = 0
                        for b, b_label in enumerate(has_answers_sent):
                            for i, sent_label in enumerate(b_label):
                                if sent_label == []:
                                    continue
                                num_sent = len(sent_label)
                                sent_label = torch.tensor(sent_label)
                                sent_loss_i = sce_loss_fct(sentence_logits[b][i][:num_sent], sent_label.to(sentence_logits.device))
                                sent_loss += sent_loss_i
                                passage_cnt += 1
                                sent_cnt += num_sent

                        if self.sce_loss_reduction == 'sum' and sent_cnt > 0:
                            sent_loss = sent_loss / num_sent ## This actually need to be divided by num_sent * passage_cnt
                        elif self.sce_loss_reduction == 'mean' and passage_cnt > 0:
                            sent_loss = sent_loss / passage_cnt
                    else:
                        sentence_logits = sentence_logits[:, 1].view(bsz*n_passages, -1) + sentence_mask
                        sentence_probs = nn.functional.softmax(sentence_logits, dim=1)
                        sentence_label = torch.zeros_like((sentence_mask))
                        for b, b_row in enumerate(has_answers_sent):
                            for i, row in enumerate(b_row):
                                sentence_label[b*n_passages+i, :len(row)] = torch.tensor(row, device=sentence_label.device)
                        sentence_label_sum = sentence_label.sum(dim=1)
                        num_sent_has_answer = sentence_label.sum(dim=1).nonzero().shape[0]

                        sent_loss = -(sentence_label * (sentence_probs+1e-32).log()).sum(dim=1)
                        if num_sent_has_answer == 0:
                            num_sent_has_answer = 1
                        sent_loss = (sent_loss / (sentence_label_sum+1e-32)).sum() / (num_sent_has_answer+1)
                    sent_loss = self.sce_loss_weight * sent_loss
        ####################### for passages classifier
        if self.ce_loss_weight > 0:
            if not self.sce_loss_weight > 0:
                pooled_output = self.pooler(hidden_states.contiguous().view(bsz*n_passages, -1, self.config.d_model),
                                            attention_mask.contiguous().view(bsz*n_passages, -1),
                                            passage_pooling=self.opt.passage_pooling)

            pooled_output = self.dropout(pooled_output)
            if self.ce_loss_type == 'point':
                logits = self.classifier(pooled_output)
                probs = nn.functional.softmax(logits, dim=1)[:, 1].view(bsz, n_passages)
            else:
                logits = self.classifier(pooled_output)[:, 1]
                logits = logits.view(bsz, n_passages)
                probs = nn.functional.softmax(logits, dim=1)

            if has_answers is not None: ## which means it is training now
                if self.ce_loss_type == 'point':
                    loss_fct = nn.CrossEntropyLoss(reduction='mean')
                    cross_encoder_loss = loss_fct(logits, has_answers.view(-1))
                elif self.ce_loss_type == 'pair':
                    loss_fct = nn.CrossEntropyLoss(reduction='mean')
                    for b, b_row in enumerate(has_answers):
                        ## if b_row is all 0, then skip
                        if b_row.sum() == 0 or b_row.sum() == b_row.shape[0]:
                            b_logits_pair  = logits[b][[0,0]].view(1,2)
                            b_labels = torch.zeros((1, 2), device=b_row.device)
                        else:
                            pos_indices = b_row.nonzero(as_tuple=True)[0]
                            neg_indices = (b_row == 0).nonzero(as_tuple=True)[0]
                            pos_neg_pair = torch.cartesian_prod(pos_indices, neg_indices)
                            b_logits_pair  = logits[b][pos_neg_pair]
                            b_labels = torch.zeros(b_logits_pair.shape[0], dtype=torch.long, device=b_logits_pair.device)
                        cross_encoder_loss += loss_fct(b_logits_pair, b_labels)
                    num_has_answer = has_answers.sum(dim=1).nonzero().shape[0]
                    if num_has_answer == 0:
                        num_has_answer = 1
                    cross_encoder_loss = cross_encoder_loss / num_has_answer
                elif self.ce_loss_type == 'list':
                    has_answers = has_answers.to(probs.device)
                    has_answers_sum = has_answers.sum(dim=1)
                    num_has_answer = has_answers_sum.nonzero().shape[0]
                    # cross_encoder_loss = -(has_answers * (probs+1e-10).log()).sum(dim=1).mean()
                    cross_encoder_loss = -(has_answers * (probs+1e-32).log()).sum(dim=1)
                    if num_has_answer == 0:
                        num_has_answer = 1
                    cross_encoder_loss = (cross_encoder_loss / (has_answers_sum+1e-32)).sum() / num_has_answer

                cross_encoder_loss = self.ce_loss_weight * cross_encoder_loss

        
        ### =============================== For rank embedding
        if self.use_rank_embedding:
            passage_order = torch.argsort(probs)
            rank_embeddings = self.rank_embedding(passage_order)
            rank_embeddings = rank_embeddings.view(bsz, n_passages, -1, self.d_model)
            hidden_states = hidden_states.view(bsz, n_passages, -1, self.d_model)
            hidden_states = hidden_states + rank_embeddings
            hidden_states = hidden_states.view(bsz, -1, self.d_model)

        ### =============================== For Masking ###
        not_selected_indices = None
        if self.ce_mask_threshold > 0.0:
            selected = probs > self.ce_mask_threshold
            selected[(selected.sum(1) == 0).nonzero()] = True
            not_selected_indices = torch.where(~selected) ## bsz * n_passages
            attention_mask.view(bsz, n_passages, -1)[not_selected_indices] = False ## bsz * n_passages * seq_len
            if sentence_logits is not None: ## Which implies that sent_classifier is used
                sentence_logits.view(bsz, n_passages, -1, 2)[not_selected_indices[0], not_selected_indices[1], :, 0] = 1e+10
        
        if self.sce_mask_threshold > 0.0:
            sentence_not_selected = sentence_probs < self.sce_mask_threshold ## sentences which will be masked are True
            sentence_not_selected[(sentence_not_selected.sum(1) == sentence_not_selected.shape[1]).nonzero()] = False
            ## select True which will be maksed
            not_selected_sentence_indices = sentence_not_selected.nonzero(as_tuple=True)
            if not_selected_indices is not None:
                not_selected_ctx_indices = not_selected_indices[0] * 20 + not_selected_indices[1]
                temp_mask = ~torch.isin(not_selected_sentence_indices[0], not_selected_ctx_indices)

                # Filter the tensors based on the mask
                filtered_ctx = not_selected_sentence_indices[0][temp_mask]
                filtered_sent = not_selected_sentence_indices[1][temp_mask]
                not_selected_sentence_indices = (filtered_ctx, filtered_sent)

            attention_mask = attention_mask.view(bsz*n_passages, -1)
            ## iterate sentence
            new_sentence_spans = []
            for b_sentence_spans in sentence_spans:
                new_sentence_spans.extend(b_sentence_spans)

            ## remove sent_i if sent_i is bigger than number of sentences in ctx_i
            for ctx_i, sent_i in zip(*not_selected_sentence_indices):
                if sent_i >= len(new_sentence_spans[ctx_i]):
                    not_selected_sentence_indices[1][not_selected_sentence_indices[0] == ctx_i] = len(new_sentence_spans[ctx_i]) - 1

            for ctx_i, sent_i in zip(*not_selected_sentence_indices):
                ctx_sentece_spans = new_sentence_spans[ctx_i]
                if sent_i >= len(ctx_sentece_spans):
                    continue
                span = new_sentence_spans[ctx_i][sent_i]
                attention_mask[ctx_i, span[0]:span[1]] = False
            attention_mask = attention_mask.view(bsz, -1)
            
        #### Add code for sentences summary
        sent_summary_bos = None
        if self.use_sent_summary:
            if sentence_mask is not None:
                sentence_embeddings = sentence_embeddings.view(bsz, -1, self.d_model)
                sentence_logits = sentence_logits.view(bsz, -1, 2)
                max_num_sent = sentence_logits.shape[1] // n_passages
                if has_answers_sent is not None and self.sent_summary_tf: # tf is abbreviation of teacher forcing
                    b_i, sent_i = [], []
                    for b, b_label in enumerate(has_answers_sent):
                        for c_i, sent_label in enumerate(b_label):
                            for s_i, label in enumerate(sent_label):
                                if label == 1:
                                    b_i.append(b)
                                    sent_i.append(c_i * max_num_sent + s_i)
                    selected_sentence_indices = (torch.tensor(b_i, dtype=torch.int64), torch.tensor(sent_i, dtype=torch.int64))
                else:
                    selected_sentence_indices = torch.nonzero(torch.argmax(sentence_logits, dim=-1).reshape(bsz, -1), as_tuple=True)
                selected_sentence_embeddings = sentence_embeddings[selected_sentence_indices]
                sent_summary_bos = []
                for b in range(bsz):
                    indices = selected_sentence_indices[0] == b
                    if indices.sum() == 0:
                        ## if there is no sentence, then just add zero vector
                        sent_summary_bos.append(torch.zeros((self.d_model), device=sentence_embeddings.device))
                        continue
                    if self.sent_summary_pool == 'mean':
                        sent_summary_bos.append(selected_sentence_embeddings[indices].mean(dim=0))
                    elif self.sent_summary_pool == 'max':
                        sent_summary_bos.append(torch.max(selected_sentence_embeddings[indices], dim=0).values)
                    else:
                        ## Raise erorr and print "Must specify sent_summary_pool as 'mean' or 'max'"
                        raise ValueError("Must specify sent_summary_pool as 'mean' or 'max'")
                sent_summary_bos = torch.stack(sent_summary_bos)
            else:
                sent_summary_bos = torch.zeros((bsz, self.d_model), device=sentence_embeddings.device)
                
        if self.sent_highlight == 'add':
            ## Add sentence highlight vector to sentence embeddings
            ## where sentence is predicted as positive
            sentence_logits = sentence_logits.view(bsz, -1, 2)
            max_num_sent = sentence_logits.shape[1] // n_passages
            selected_sentence_indices = torch.argmax(sentence_logits, dim=-1).nonzero(as_tuple=True)
            # Initialize mask
            mask = torch.zeros((bsz, n_passages * passage_length), dtype=torch.long)

            # Efficiently create mask
            for b in range(bsz):
                sample_sent_indices = selected_sentence_indices[1][selected_sentence_indices[0] == b]
                passage_indices = (sample_sent_indices // max_num_sent).long()
                sent_indices = (sample_sent_indices % max_num_sent).long()

                for p_i, s_i in zip(passage_indices.tolist(), sent_indices.tolist()):
                    if p_i < len(sentence_spans[b]) and s_i < len(sentence_spans[b][p_i]):
                        start, end = sentence_spans[b][p_i][s_i]
                        offset = passage_length * p_i
                        mask[b, start + offset:end + offset] = 1
            # Update hidden states
            hidden_states = hidden_states + self.sent_highlight_emb(mask.to(hidden_states.device))

        ## =============================== ##
        if self.reduced_n > 0 :
            topN_indices = probs.argsort(dim=-1, descending=True)[:, :self.reduced_n]
            hidden_states = hidden_states.view(bsz, n_passages, -1, self.d_model)
            seq_len = hidden_states.shape[2]
            hidden_states = torch.gather(hidden_states, dim=1, index=topN_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, seq_len, self.d_model)).contiguous().view(bsz, -1, self.d_model)
            
            attention_mask = attention_mask.view(bsz, n_passages, seq_len)
            attention_mask = torch.gather(attention_mask, dim=1, index=topN_indices.unsqueeze(-1).expand(-1, -1, seq_len))
            attention_mask = attention_mask.view(bsz, -1)
        ## =============================== ##


        ## > multiply prob with hidden states # deprecated
        # ## =============================== ##
        # probs = n_passages * probs ## For multiplyscoreN, otherwise just multiplyscore. 코드 위치는 여기가 맞을듯..
        # hidden_states = probs.view(bsz, n_passages, 1, 1) * hidden_states.view(bsz, n_passages, -1, self.d_model)
        # ## =============================== ##
        outputs.last_hidden_state = hidden_states

        # outputs.last_hidden_state = hidden_states.contiguous().view(bsz, -1, self.d_model)
        outputs.attention_mask = attention_mask.contiguous().view(bsz, -1)
        ## (TODO) return output dictionary 처리
        return outputs, cross_encoder_loss, sent_loss, probs, sentence_preds, sent_summary_bos

class CheckpointWrapper(nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """
    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output

def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block


class FiD_classifier(transformers.BertForSequenceClassification):
    def __init__(self, config, opt=None):
        super().__init__(config)
        self.opt = opt
        
        self.score_layer1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj_act = nn.GELU()
        self.score_layer2 = nn.Linear(config.hidden_size, 1)
        self.score_layer = nn.Sequential(self.score_layer1, self.proj_act, self.score_layer2)

        nn.init.normal_(self.score_layer1.weight, std=0.02)
        nn.init.zeros_(self.score_layer1.bias)
        nn.init.normal_(self.score_layer2.weight, std=0.02)
        nn.init.zeros_(self.score_layer2.bias)

    def forward(self, **batch):
        outputs = super().forward(**batch)
        new_logits = self.score_layer(outputs.pooled_output)
        outputs.logits = new_logits
        return outputs
    
class FiD_encoder(transformers.T5EncoderModel):
    def __init__(self, config, opt=None):
        super().__init__(config)
        self.opt = opt
        self.wrap_encoder(opt)

        self.pooler = T5Pooler(config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_classifier()
        self.ce_loss = nn.CrossEntropyLoss()
        if opt.use_hlatr:
            from transformers import BertConfig
            self.HLATR_config = BertConfig()
            self.HLATR_config.hidden_size = 768
            self.HLATR_config.num_hidden_layers = 4
            self.HLATR_config.num_attention_heads = 2
            self.HLATR = HLATR_reranker(self.HLATR_config)

    def init_classifier(self):
        self.classifier.weight.data.normal_(mean=0.0, std=self.config.initializer_factor * 1.0)
        self.classifier.bias.data.zero_()
        return super().init_weights()

    def forward(self, input_ids=None, attention_mask=None, q_tokens=None, has_answers=None, generate=False, **kwargs):
        bsz, n_passages, passage_length = input_ids.shape
        input_ids = input_ids.view(bsz*n_passages, passage_length)
        attention_mask = attention_mask.view(bsz*n_passages, passage_length)
        outputs = self.encoder.encoder(input_ids, attention_mask)
        sequence_output = outputs.last_hidden_state

        pooled_output = self.pooler(sequence_output, attention_mask) ## attention_mask here is different.
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output).view(bsz, n_passages)
        if self.opt.use_hlatr:
            hlatr_input = pooled_output.view(bsz, n_passages, -1)
            hlatr_attention_mask = torch.ones((bsz, n_passages), device=hlatr_input.device)
            hlatr_attention_mask = self.get_extended_attention_mask(hlatr_attention_mask, hlatr_input.shape)
            loss, hlatr_hidden_states, logits = self.HLATR(hlatr_input, hlatr_attention_mask, has_answers)
            probs = nn.functional.softmax(logits, dim=1)
        else:
            probs = nn.functional.softmax(logits, dim=1)

        if has_answers is not None:
            labels = torch.zeros(has_answers.shape[0], dtype=torch.long, device=logits.device)
            loss = self.ce_loss(logits, labels)
            # loss = -(has_answers * (probs+1e-10).log()).sum(dim=1).mean()
            outputs = (loss, probs)
        elif generate:
            outputs = ([], probs) ## outputs (generated_ids, probs)

        return outputs 

    def generate(self, **kwargs):
        ## This function is to align with FiDT5 model to use same train_reader code.
        outputs = self.forward(generate=True, **kwargs)
        return outputs

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict, strict=False)
        self.wrap_encoder(self.opt)

    def wrap_encoder(self, opt, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, opt=opt, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint