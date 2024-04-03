    def forward(self, input_ids=None, attention_mask=None, q_tokens=None, **kwargs,):
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz*self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz*self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        n_passages = self.n_passages

        outputs.attention_mask = attention_mask
        question_indices = np.arange(0, bsz*n_passages, n_passages)
        if self.extra_question == 'embs':
            pair_indices = np.setdiff1d(np.arange(0, bsz*n_passages), question_indices)
            if not self.use_local_interaction:    
                q_last_hidden_state = outputs.last_hidden_state[question_indices]
                outputs.last_hidden_state = outputs.last_hidden_state[pair_indices]
                outputs['q_last_hidden_state'] = q_last_hidden_state
                n_passages = self.n_passages - 1
                
                q_attention_mask = attention_mask[question_indices]
                attention_mask = attention_mask[pair_indices]
                outputs.attention_mask = attention_mask
                outputs['q_attention_mask'] = q_attention_mask
        elif self.extra_question == 'ids':
            ## Questions are not encoded here.
            max_q_len = max(map(len, q_tokens))
            new_q_ids, new_q_mask = [], []
            for q_token in q_tokens:
                q_token_len = len(q_token)
                diff = max_q_len -q_token_len
                q_token = q_token + [0] * diff
                q_mask = [1] * q_token_len + [0] * diff
                new_q_ids.append(q_token)
                new_q_mask.append(q_mask)

            q_last_hidden_state = self.encoder.embed_tokens(torch.LongTensor(new_q_ids).to(input_ids.device))
            q_attention_mask =  torch.tensor(new_q_mask).to(input_ids.device)
            outputs['q_last_hidden_state'] = q_last_hidden_state
            outputs['q_attention_mask'] = q_attention_mask

        if self.tokens_k is not None:
            if self.tokens_k[0] == 0: ## 0 means 'f'irst
                passage_length = self.tokens_k[1]
                outputs.last_hidden_state = outputs.last_hidden_state[:, :passage_length, :]
                outputs.attention_mask = attention_mask[:, :passage_length]
            elif self.tokens_k[0] == 1: ## 1 means 'm'iddle
                passage_length = self.tokens_k[1]
                last_hidden_state_list, attention_mask_list = [], []
                for b_idx, q_token in enumerate(q_tokens):
                    len_q = len(q_token)
                    s_idx = b_idx * n_passages
                    e_idx = (b_idx + 1) * n_passages
                    last_hidden_state_list.append(outputs.last_hidden_state[s_idx:e_idx, len_q:len_q+passage_length])     
                    attention_mask_list.append(attention_mask[s_idx:e_idx, len_q:len_q+passage_length])              
                outputs.last_hidden_state = torch.stack(last_hidden_state_list)
                outputs.attention_mask = torch.stack(attention_mask_list)
            elif self.tokens_k[0] == 2: ## 2 means 'l'ast
                passage_length = self.tokens_k[1]
                last_hidden_state_list, attention_mask_list = [], []
                d_lens = attention_mask.sum(1)
                for d_idx, d_len in enumerate(d_lens):
                    s_idx = max(0, d_len-passage_length)
                    e_idx = max(d_len, passage_length)
                    last_hidden_state_list.append(outputs.last_hidden_state[d_idx, s_idx:e_idx])
                    attention_mask_list.append(attention_mask[d_idx, s_idx:e_idx])
                outputs.last_hidden_state = torch.stack(last_hidden_state_list)
                outputs.attention_mask = torch.stack(attention_mask_list)
            elif self.tokens_k[0] == 3: # Select the top 3 vectors from the batch
                passage_length = self.tokens_k[1]
                scores = self.proj_layer(outputs.last_hidden_state).squeeze(-1)
                top_indices = scores.topk(passage_length, 1, sorted=False).indices
                top_indices = top_indices.sort().values
                expanded_indices = top_indices.unsqueeze(-1).repeat(1, 1, outputs.last_hidden_state.size(-1))
                outputs.last_hidden_state = torch.gather(outputs.last_hidden_state, dim=1, index=expanded_indices)
                
            elif self.tokens_k[0] == 4: ## Max pooling
                passage_length = (outputs.last_hidden_state.size(1) - self.tokens_k[1]) // self.tokens_k[2] + 1
                
                # Reshape input tensor using unfold
                unfolded = outputs.last_hidden_state.unfold(1, self.tokens_k[1], self.tokens_k[2])
                unfolded_attention_mask = attention_mask.unfold(1, self.tokens_k[1], self.tokens_k[2])
                # Perform max pooling along the unfolded dimension
                pooled, _ = unfolded.max(dim=-1)
                pooled_attention_mask, _ = unfolded_attention_mask.max(dim=-1)

                outputs.last_hidden_state = pooled
                outputs.attention_mask = pooled_attention_mask
            elif self.tokens_k[0] == 5: ## Mean pooling
                passage_length = (outputs.last_hidden_state.size(1) - self.tokens_k[1]) // self.tokens_k[2] + 1
                
                # Reshape input tensor using unfold
                unfolded = outputs.last_hidden_state.unfold(1, self.tokens_k[1], self.tokens_k[2])
                unfolded_attention_mask = attention_mask.unfold(1, self.tokens_k[1], self.tokens_k[2])
                # Perform max pooling along the unfolded dimension
                pooled = unfolded.mean(dim=-1)
                pooled_attention_mask, _ = unfolded_attention_mask.max(dim=-1)

                outputs.last_hidden_state = pooled
                outputs.attention_mask = pooled_attention_mask
            elif self.tokens_k[0] == 6: ## Conv.
                passage_length = (outputs.last_hidden_state.size(1) - self.tokens_k[1]) // self.tokens_k[2] + 1
                # unfolded = outputs.last_hidden_state.unfold(1, self.tokens_k[1], self.tokens_k[2]).contiguous().view(bsz*n_passages, passage_length,-1)
                # pooled = (unfolded @ self.weight_sparse) + self.bias
                # pooled = self.conv_layer(unfolded)
                # conv_outputs = []
                # for i in range(self.d_model):
                #     conv_outputs.append(self.convs[i](outputs.last_hidden_state[:, :, i].unsqueeze(-2)).squeeze(1))
                
                # pooled = torch.stack(conv_outputs, dim=-1)
                if self.opt.use_conv_dropout:
                    outputs.last_hidden_state = self.dropout(outputs.last_hidden_state)
                pooled = self.conv(outputs.last_hidden_state.permute(0, 2, 1)) ## bsz * seq_len * dim --> bsz * dim * seq_len
                pooled = pooled.permute(0, 2, 1) ## bsz * dim * seq_len --> bsz * seq_len * dim
                
                if self.opt.use_conv_relu:
                    pooled = self.relu(pooled)

                if self.conv.padding == 'same':
                    pad_size = self.tokens_k[1]//2
                    attention_mask_temp = F.pad(attention_mask, (pad_size, pad_size))
                else:
                    attention_mask_temp = attention_mask

                unfolded_attention_mask = attention_mask_temp.unfold(1, self.tokens_k[1], self.tokens_k[2])
                pooled_attention_mask, _ = unfolded_attention_mask.max(dim=-1)

                outputs.last_hidden_state = pooled
                outputs.attention_mask = pooled_attention_mask

            elif self.tokens_k[0] == 7: ## multi-channel conv.

                pooling = False
                conv_outputs, pooled_attention_mask_list = [], []
                for i, conv in enumerate(self.convs):
                    if conv is None:
                        pooling = True
                        break
                    kernel_size = self.tokens_k[1][i]
                    stride = self.tokens_k[2][i]

                    if self.opt.use_conv_dropout:
                        conv_output = conv(self.dropout(outputs.last_hidden_state.permute(0, 2, 1)))
                    else:
                        conv_output = conv(outputs.last_hidden_state.permute(0, 2, 1))

                    if self.opt.use_conv_relu:
                        conv_output = self.relu(conv_output)

                    conv_outputs.append(conv_output) ## bsz * seq_len * dim --> bsz * dim * seq_len
                    if conv.padding == 'same':
                        pad_size = kernel_size//2
                        attention_mask_temp = F.pad(attention_mask, (pad_size, pad_size))
                    else:
                        attention_mask_temp = attention_mask

                    unfolded_attention_mask = attention_mask_temp.unfold(1, kernel_size, stride)
                    pooled_attention_mask, _ = unfolded_attention_mask.max(dim=-1)
                    pooled_attention_mask_list.append(pooled_attention_mask)

                # pooled = conv_outputs[0].permute(0, 2, 1) ## (bsz, passage_length, dim)
                if self.convs[0].padding == 'same':
                    pooled = torch.max(torch.stack(conv_outputs, dim=0), dim=0)[0].permute(0, 2, 1) ## (bsz, dim, passage_length) --> (bsz, passage_length, dim)
                    pooled_attention_mask = torch.max(torch.stack(pooled_attention_mask_list), dim=0)[0] ## (bsz, passage_length)
                else:
                    pooled = torch.cat(conv_outputs, dim=-1).permute(0, 2, 1)
                    pooled_attention_mask = torch.cat(pooled_attention_mask_list, dim=-1)

                passage_length = pooled.size(1)
                # # Reshape input tensor using unfold
                if pooling:
                    window = self.tokens_k[1][-1]
                    stride = self.tokens_k[2][-1]
                    passage_length = (pooled.size(1) - window) // stride + 1
                    unfolded = pooled.unfold(1, window, stride)
                    unfolded_attention_mask = pooled_attention_mask.unfold(1, window, stride)
                    
                    # Perform max pooling along the unfolded dimension
                    pooled, _ = unfolded.max(dim=-1)
                    pooled_attention_mask, _ = unfolded_attention_mask.max(dim=-1)
                
                # original only need these lines.
                # pooled_attention_mask = torch.cat(pooled_attention_mask_list, dim=-1)
                # pooled = torch.cat(conv_outputs, dim=-1)

                outputs.last_hidden_state = pooled
                outputs.attention_mask = pooled_attention_mask
        
        ## sep_q_p is deprecated for now ... 2023-05-27
        if self.sep_q_p is not None: # sep_q_p == tokens, embs, embs_mean
            num_query_tokens = attention_mask[0].sum()
            if self.sep_q_p == 'tokens':
                outputs['q_last_hidden_state'] = self.encoder.embed_tokens(input_ids[0][:num_query_tokens])
            elif self.sep_q_p == 'embs':
                outputs['q_last_hidden_state'] = outputs.last_hidden_state[0][:num_query_tokens]
            elif self.sep_q_p == 'embs_mean':
                outputs['q_last_hidden_state'] = outputs.last_hidden_state[0][:num_query_tokens].mean(dim=0, keepdims=True)
            else:
                raise ValueError(f"sep_q_p should be one of ['tokens', 'embs', 'embs_mean'], got {self.sep_q_p}")
            if self.use_local_interaction:
                hidden_states = torch.cat((outputs['q_last_hidden_state'].repeat(n_passages-1, 1, 1), outputs.last_hidden_state[1:]), dim=1)
                outputs.last_hidden_state = self.attn_layer(hidden_states)[0][:, num_query_tokens:].contiguous().view(bsz, (n_passages-1) * passage_length, -1)
            else:
                outputs.last_hidden_state = outputs.last_hidden_state[1:].view(bsz, (n_passages-1) * passage_length, -1)
        else:
            if self.use_local_interaction:
                n_pair = n_passages - 1
                q_attention_mask = outputs.attention_mask[question_indices]
                max_q_len = q_attention_mask.sum(1).max()
                q_attention_mask = q_attention_mask[:, :max_q_len].repeat(1, n_pair).contiguous().view(bsz*n_pair, max_q_len)
                p_attention_mask = outputs.attention_mask[pair_indices]
                attn_mask = torch.cat((q_attention_mask, p_attention_mask), dim=1)
                
                ## Original
                # attn_mask =  (1.0 - attn_mask[:, None, None, :].to(dtype=torch.float32)) * torch.finfo(torch.float32).min
                
                ## For Q cross-attention
                attn_mask = (1.0 - p_attention_mask[:, None, None, :].to(dtype=torch.float32)) * torch.finfo(torch.float32).min ## p_attention_mask

                q_last_hidden_state = outputs.last_hidden_state[question_indices][:, :max_q_len]
                q_last_hidden_state = q_last_hidden_state.repeat(1, n_pair, 1).contiguous().view(bsz*n_pair, max_q_len, self.d_model)
                p_last_hidden_state = outputs.last_hidden_state[pair_indices]
                ## Original
                # attn_input = torch.cat((q_last_hidden_state, p_last_hidden_state), dim=1)
                # outputs.last_hidden_state = self.attn_layer(attn_input, attn_mask)[0][:, -passage_length:].contiguous().view(bsz, n_pair*passage_length, -1)
                # outputs.attention_mask = p_attention_mask.contiguous().view(bsz, n_pair*passage_length)
                
                ## For Q cross-attention
                passage_length = max_q_len
                o = self.attn_layer(q_last_hidden_state, key_value_states=p_last_hidden_state, attention_mask=attn_mask)[0].view(bsz, n_pair*passage_length, -1)
                outputs.last_hidden_state = o
                outputs.attention_mask = q_attention_mask.view(bsz, n_pair*passage_length)
            else:
                outputs.last_hidden_state = outputs.last_hidden_state.contiguous().view(bsz, n_passages*passage_length, -1)
                outputs.attention_mask = outputs.attention_mask.contiguous().view(bsz, n_passages*passage_length)
        outputs['n_passages'] = n_passages

        return outputs