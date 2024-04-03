import torch
import torch.nn as nn
import warnings

from typing import Optional, Tuple, Union
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.bert.modeling_bert import BertEncoder

class T5Pooler(nn.Module):
    def __init__(self, config, opt):
        super().__init__()
        self.opt = opt
        self.cls_norm_type = opt.cls_norm_type
        self.n_passages = opt.n_contexts
        self.bsz = opt.per_gpu_batch_size
        self.hidden_size = config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)

        if self.opt.pooler_activation == 'tanh':
            self.activation = nn.Tanh()
        elif self.opt.pooler_activation == 'relu':
            self.activation = nn.ReLU()
        elif self.opt.pooler_activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        ## Input for norm = (bsz*n_passage, dim)
        # if self.cls_norm_type == 'layer':
        #     self.cls_norm = T5LayerNorm(self.hidden_size, eps=config.layer_norm_epsilon)
        # elif self.cls_norm_type == 'batch' or self.cls_norm_type == 'abn':
        #     self.cls_norm = nn.BatchNorm1d(self.hidden_size)

        self.init_weights()

    def init_weights(self):
        self.dense.weight.data.normal_(mean=0.0, std=1.0)
        self.dense.bias.data.zero_()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        sentence_spans: Optional[list] = None,
        passage_pooling: Optional[str] = 'first',
        sentence_pooling: Optional[str] = 'mean',
    ) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        ### ========= This is for act --> pool ========== ###
        # hidden_states = self.dense(hidden_states)
        # hidden_states = self.activation(hidden_states)

        if sentence_spans is not None:
            ## token_tensor shape: (bsz*n_pasage, passage_length, dim)
            token_tensor = hidden_states
        else:
            ## token_tensor shape: (bsz*n_passage, dim)
            if passage_pooling == 'first':
                token_tensor = hidden_states[:, 0] ## CLS가 아닌뎅..도 불구하고 제일 좋다!
            else:
                if passage_pooling == 'last':
                    token_tensor = hidden_states[torch.arange(hidden_states.shape[0]), torch.sum(attention_mask, dim=1)-1] ## Last
                elif passage_pooling == 'mean':
                    masked_states = hidden_states * attention_mask.unsqueeze(-1).float()
                    sum_vectors = torch.sum(masked_states, dim=1)
                    count_vectors = torch.sum(attention_mask, dim=1, keepdim=True)
                    token_tensor = sum_vectors / count_vectors
                elif passage_pooling == 'max':
                    masked_states = hidden_states * attention_mask.unsqueeze(-1).float()
                    token_tensor = torch.max(masked_states, dim=1)[0]
                                    

        pooled_output = self.dense(token_tensor)
        ## pooled_output shape: (bsz*n_passage, dim)
        ## batchnorm should be applied each passage in each batch
        if self.cls_norm_type == 'layer':
            pooled_output = self.cls_norm(pooled_output)
        elif self.cls_norm_type == 'batch':
            pooled_output = pooled_output.view(self.bsz, self.n_passages, -1, self.hidden_size)
            normed_output = []
            for b in range(self.bsz):
                normed_output.append(self.cls_norm(pooled_output[b].permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous())
            if sentence_spans is not None:
                pooled_output = torch.cat(normed_output).view(self.bsz*self.n_passages, -1, self.hidden_size)
            else:
                pooled_output = torch.cat(normed_output).view(self.bsz*self.n_passages, self.hidden_size)
        elif self.cls_norm_type == 'abn':
            ## pooled_output shape: (bsz*n_passage, dim) or (bsz*n_passage, passage_length, dim)
            if sentence_spans is not None: ## (bsz*n_passage, passage_length, dim)
                pooled_output = self.cls_norm(pooled_output.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
            else: ## (bsz*n_passage, dim)
                pooled_output = self.cls_norm(pooled_output)

        pooled_output = self.activation(pooled_output)

        if sentence_spans is not None:
            pooled_output_passages = pooled_output[:, 0]

            n_passages = len(sentence_spans[0])
            max_sentence_spans = max([len(x) for b in sentence_spans for x in b])
            sentence_embeddings = []
            sentence_mask = []
            zero_vectors = torch.zeros((1, pooled_output.shape[-1]), device=pooled_output.device)
            for b, b_spans in enumerate(sentence_spans):
                for qas, qas_spans in enumerate(b_spans):
                    qas_sentence_embeddings = []
                    for _, span in enumerate(qas_spans):
                        if span[1] - span[0] == 0:
                            continue
                        if sentence_pooling == 'mean':
                            o = pooled_output[b * n_passages + qas, span[0]:span[1]].mean(0, keepdim=True)
                        elif sentence_pooling == 'max':
                            o = pooled_output[b * n_passages + qas, span[0]:span[1]].max(0, keepdim=True)[0]
                        elif sentence_pooling == 'first':
                            o = pooled_output[b * n_passages + qas, span[0]].unsqueeze(0)
                        elif sentence_pooling == 'last':
                            o = pooled_output[b * n_passages + qas, span[1]-1].unsqueeze(0)

                        qas_sentence_embeddings.append(o)
                        sentence_mask.append(0)
                    while len(qas_sentence_embeddings) < max_sentence_spans:
                        qas_sentence_embeddings.append(zero_vectors)
                        sentence_mask.append(torch.finfo(torch.float32).min)

                    if len(qas_sentence_embeddings) > 0:
                        sentence_embeddings.append(torch.cat(qas_sentence_embeddings))
                    else: ## No sentence spans because of max_seq_length
                        sentence_embeddings.append(zero_vectors)

            sentence_embeddings = torch.cat(sentence_embeddings)
            if max_sentence_spans > 0:
                sentence_mask = torch.tensor(sentence_mask, device=sentence_embeddings.device).view(-1, max_sentence_spans)
            else: ## No sentence spans because of max_seq_length
                sentence_mask = None
            pooled_output = (pooled_output_passages, sentence_embeddings, sentence_mask)

        return pooled_output

class HLATR_reranker(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = BertEncoder(config)
        self.emb_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.linear = nn.Linear(config.hidden_size, 1)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def prepare_input(self, inputs):
        input_shape = inputs.size()
        seq_length = input_shape[1]
        inputs = self.emb_layer(inputs)
        position_ids = self.position_ids[:,: seq_length]
        position_emb = self.position_embeddings(position_ids)
        embs = inputs + position_emb
        embs = self.LayerNorm(embs)
        embs = self.dropout(embs)
        return embs

    def forward(self,inputs, attention_mask=None, has_answers=None): ## has_answers is labels
        # print(inputs.keys())

        inputs = self.prepare_input(inputs)
        output = self.model(inputs, attention_mask=attention_mask)
        hidden_states = output['last_hidden_state'] # bsz * seq_len * hidden_size

        logits = self.linear(hidden_states).squeeze(-1) # bsz * seq_len * 1
        
        hlatr_loss = 0
        if has_answers is not None:
            hlatr_loss = self.cross_entropy(logits, has_answers.float())

        return hlatr_loss, hidden_states, logits


class CustomT5ForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, opt=None):
        super().__init__(config)
        self.opt = opt

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        auto_regression: Optional[bool] = False,
        replace_bos_token: Optional[bool] = False,
        q_tokens: Optional[list] = None,
        sentence_spans: Optional[list] = None,
        has_answers_sent: Optional[list] = None,
        use_late_decoding: Optional[bool] = False,
        has_answers: Optional[torch.LongTensor] = None,
        sent_summary_bos: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        cross_encoder_loss = 0

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs, cross_encoder_loss, sent_loss, probs, sentence_preds, sent_summary_bos = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                has_answers=has_answers,
                has_answers_sent=has_answers_sent,
                sentence_spans=sentence_spans,
                q_tokens=q_tokens
            )
        hidden_states = encoder_outputs.last_hidden_state
        attention_mask = encoder_outputs.attention_mask

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            if self.encoder.use_decode_num_sent:
                num_pos_sents = sentence_preds.sum([1,2])
                str_pos_num_sents = [str(n.item()) + ' <extra_id_0>' for n in num_pos_sents]
                print(str_pos_num_sents)
                tokens_pos_num_sents = self.tokenizer(str_pos_num_sents, truncation=True, add_special_tokens=False)['input_ids']
                prepend_tensors = torch.stack([torch.tensor(tokens[-2:]) for tokens in tokens_pos_num_sents]).to(labels.device)
                labels = torch.cat([prepend_tensors, labels], dim=1)
            
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        if sent_summary_bos is not None:
            if self.opt.summary_bos_option == "replace":
                if past_key_values is None:
                    decoder_inputs_embeds = self.decoder.embed_tokens(decoder_input_ids) ## (bsz, seq_len, dim)
                    ## method 1. replace bos token with sent_summary_bos
                    decoder_inputs_embeds[:, 0] = sent_summary_bos
                    decoder_input_ids = None
            elif self.opt.summary_bos_option == "add":
                if past_key_values is None:
                    decoder_inputs_embeds = self.decoder.embed_tokens(decoder_input_ids)
                    ## method 2. add sent_summary_bos to bos token
                    decoder_inputs_embeds[:, 0] = decoder_inputs_embeds[:, 0] + sent_summary_bos
                    decoder_input_ids = None
            elif self.opt.summary_bos_option == "add_every":
                if labels is not None: ## meaning that it is training
                    decoder_inputs_embeds = self.decoder.embed_tokens(decoder_input_ids) ## (bsz, seq_len, dim)
                    ## method 2. add sent_summary to all tokens
                    # decoder_inputs_embes: (bsz, seq_len, dim)
                    # sent_summary_bos: (bsz, dim)
                    decoder_inputs_embeds = decoder_inputs_embeds + sent_summary_bos.unsqueeze(1)
                    decoder_input_ids = None
                else: ## meaning that it is inference
                    decoder_inputs_embeds = self.decoder.embed_tokens(decoder_input_ids)
                    ## method 1. replace bos token with sent_summary_bos one by one each step
                    decoder_inputs_embeds[:, 0] = decoder_inputs_embeds[:, 0] + sent_summary_bos
                    decoder_input_ids = None

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        # score_list = []
        # cross_attentions = decoder_outputs.cross_attentions ## len: 12 (num_layers)
        # bsz, num_heads, _, input_seq_len = cross_attentions[0].shape
        # for cross_attention in cross_attentions:
        #     score_list.append(cross_attention[..., 0, :].view(bsz, num_heads, n_passages, -1).mean(-1).mean(1))
        # cross_attention_scores = torch.cat(score_list).view(bsz, -1, n_passages).mean(1)
        # cross_attention_scores = nn.functional.softmax(cross_attention_scores, dim=1)
        ## kldiv loss between probs and cross_attention_scores
        ## input: probs, objective: cross_attention_scores
        # kl_loss = (cross_attention_scores * (cross_attention_scores / (probs+1e-10)).log()).sum(dim=1).mean()
        # cross_encoder_loss = 0.5 * cross_encoder_loss + 0.5 * kl_loss

        ## 12 * (bsz, num_heads, gen_seq_len, input_seq_len)


        # code를 바꿈..
        # if q_attention_mask is not None:
        #     max_q_len = q_attention_mask.sum(1).max()
        #     if replace_bos_token:
        #         max_q_len = max_q_len - 1
        #     sequence_output = decoder_outputs[0][:, max_q_len:]
        # else:
        #     sequence_output = decoder_outputs[0]
        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            loss = loss + cross_encoder_loss + sent_loss
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        sentence_spans=None,
        sent_summary_bos=None,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "sentence_spans": sentence_spans,
            "sent_summary_bos": sent_summary_bos,
        }