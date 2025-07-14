from torch.autograd import Function
from latentplan.models.autoencoders import SymbolWiseTransformer
from latentplan.models.transformers import *
from latentplan.models.ein import EinLinear

class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)
            #print(codebook.shape)
            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)

vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply

class VQEmbeddingMovingAverage(nn.Module):
    def __init__(self, K, D, decay=0.99):
        super().__init__()
        embedding = torch.zeros(K, D)
        embedding.uniform_(-1./K, 1./K)
        self.decay = decay

        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.ones(K))
        self.register_buffer("ema_w", self.embedding.clone())

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents


    def straight_through(self, masked_z_e_x, z_e_x):
        K, D = self.embedding.size()

        masked_z_e_x_ = masked_z_e_x.contiguous()
        z_e_x_ = z_e_x.contiguous()
        #retrieve embedding and indices
        z_q_x_, indices = vq_st(masked_z_e_x_, self.embedding)
        z_q_x = z_q_x_.contiguous()


        if self.training:
            encodings = F.one_hot(indices, K).float()
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

            dw = encodings.transpose(1, 0)@z_e_x_.reshape([-1, D])
            self.ema_w = self.decay * self.ema_w + (1 - self.decay) * dw

            self.embedding = self.ema_w / (self.ema_count.unsqueeze(-1))
            self.embedding = self.embedding.detach()
            self.ema_w = self.ema_w.detach()
            self.ema_count = self.ema_count.detach()

        z_q_x_bar_flatten = torch.index_select(self.embedding, dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.contiguous()

        return z_q_x, z_q_x_bar


    # def straight_through(self, z_e_x):
    #     K, D = self.embedding.size()
    #
    #     z_e_x_ = z_e_x.contiguous()
    #     #retrieve embedding and indices
    #     z_q_x_, indices = vq_st(z_e_x_, self.embedding)
    #     z_q_x = z_q_x_.contiguous()
    #
    #
    #     if self.training:
    #         encodings = F.one_hot(indices, K).float()
    #         self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)
    #
    #         dw = encodings.transpose(1, 0)@z_e_x_.reshape([-1, D])
    #         self.ema_w = self.decay * self.ema_w + (1 - self.decay) * dw
    #
    #         self.embedding = self.ema_w / (self.ema_count.unsqueeze(-1))
    #         self.embedding = self.embedding.detach()
    #         self.ema_w = self.ema_w.detach()
    #         self.ema_count = self.ema_count.detach()
    #
    #     z_q_x_bar_flatten = torch.index_select(self.embedding, dim=0, index=indices)
    #     z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
    #     z_q_x_bar = z_q_x_bar_.contiguous()
    #
    #     return z_q_x, z_q_x_bar

class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.contiguous()

        return z_q_x, z_q_x_bar


class VQStepWiseTransformer(nn.Module):
    def __init__(self, config, feature_dim):
        super().__init__()
        self.K=config.K
        self.latent_size = config.trajectory_embd
        self.condition_size = config.observation_dim
        self.trajectory_input_length = config.block_size - config.transition_dim
        self.embedding_dim = config.n_embd
        self.trajectory_length = config.block_size//config.transition_dim-1
        self.block_size = config.block_size
        self.observation_dim = feature_dim
        self.action_dim = config.action_dim
        self.transition_dim = config.transition_dim
        self.latent_step = config.latent_step
        self.state_conditional = config.state_conditional
        if "masking" in config:
            self.masking = config.masking
        else:
            self.masking = "none"

        self.encoder = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        if "ma_update" in config and not (config.ma_update):
            self.codebook = VQEmbedding(config.K, config.trajectory_embd)
            self.ma_update = False
        else:
            self.codebook = VQEmbeddingMovingAverage(config.K, config.trajectory_embd)
            self.ma_update = True

        if "residual" not in config :
            self.residual = True
        else:
            self.residual = config.residual
        self.decoder = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.pos_emb = nn.Parameter(torch.zeros(1, self.trajectory_length, config.n_embd))
        #print("transition_dim:", self.transition_dim)
        self.embed = nn.Linear(self.transition_dim, self.embedding_dim)

        self.predict = nn.Linear(self.embedding_dim, self.transition_dim)
        self.cast_embed = nn.Linear(self.embedding_dim, self.latent_size)
        self.latent_mixing = nn.Linear(self.latent_size+self.observation_dim, self.embedding_dim)
        if "bottleneck" not in config:
            self.bottleneck = "pooling"
        else:
            self.bottleneck = config.bottleneck
        #max pooling operation
        if self.bottleneck == "pooling":
            self.latent_pooling = nn.MaxPool1d(self.latent_step, stride=self.latent_step)
        elif self.bottleneck == "attention":
            self.latent_pooling = AsymBlock(config, self.trajectory_length // self.latent_step)
            self.expand = AsymBlock(config, self.trajectory_length)
        else:
            raise ValueError(f'Unknown bottleneck type {self.bottleneck}')

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)


    def encode(self, joined_inputs):

        joined_inputs = joined_inputs.to(dtype=torch.float32)
        b, t, joined_dimension = joined_inputs.size()
        #print("dim:",b, t, joined_dimension)
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.embed(joined_inputs)

        ## [ 1 x T x embedding_dim ]
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        ## [ B x T x embedding_dim ]
        #print(token_embeddings.shape, position_embeddings.shape)
        x = self.drop(token_embeddings + position_embeddings)
        x = self.encoder(x)
        ## [ B x T x embedding_dim ]
        if self.bottleneck == "pooling":
            x = self.latent_pooling(x.transpose(1, 2)).transpose(1, 2)
        elif self.bottleneck == "attention":
            x = self.latent_pooling(x)
        else:
            raise ValueError()

        ## [ B x (T//self.latent_step) x embedding_dim ]
        x = self.cast_embed(x)
        return x



    def decode(self, latents, state):
        """
            latents: [B x (T//self.latent_step) x latent_size]
            state: [B x observation_dimension]
        """
        #print("input", latents.shape, latents[0])
        B, T, _ = latents.shape
        #print("shape check:",latents.shape,B,T)
        state_flat = torch.reshape(state, shape=[B, 1, -1]).repeat(1, T, 1)
        if not self.state_conditional:
            state_flat = torch.zeros_like(state_flat)
        inputs = torch.cat([state_flat, latents], dim=-1)

        inputs = self.latent_mixing(inputs)
        if self.bottleneck == "pooling":
            inputs = torch.repeat_interleave(inputs, self.latent_step, 1)
        elif self.bottleneck == "attention":
            inputs = self.expand(inputs)
        #print("input", inputs.shape)
        #print("self.pos_emb", self.pos_emb.shape, self.trajectory_length)
        #inputs = inputs + self.pos_emb[:, :inputs.shape[1]]
        #print("after pos_emb input", self.pos_emb.shape)
        #print("input", inputs.shape)
        #print("before decoder", inputs[0])
        inputs = inputs + self.pos_emb[:, :inputs.shape[1]]
        x = self.decoder(inputs)
        x = self.ln_f(x)
        #print("after decoder", x.shape)
        ## [B x T x obs_dim]
        #print("output",x.shape)
        joined_pred = self.predict(x)
        #print("joined_pred", joined_pred.shape)
        joined_pred[:, :, -1] = torch.sigmoid(joined_pred[:, :, -1])
        joined_pred[:, :, 1:self.observation_dim+1] += torch.reshape(state, shape=[B, 1, -1])
        return joined_pred


    def forward(self, joined_inputs, state):
        mask = torch.ones_like(joined_inputs)
        weight = 0
        mask[:,:,0] = weight
        masked_joined_inputs = joined_inputs * mask
        trajectory_feature = self.encode(joined_inputs)
        masked_trajectory_feature = self.encode(masked_joined_inputs)
        latents_st, latents = self.codebook.straight_through(masked_trajectory_feature, trajectory_feature)
        if self.bottleneck == "attention":
            if self.masking == "uniform":
                mask = torch.ones(latents_st.shape[0], latents_st.shape[1], 1).to(latents_st.device)
                mask_index = np.random.randint(0, latents_st.shape[1], size=[latents_st.shape[0]])
                for i, start in enumerate(mask_index):
                    mask[i, -start:, 0] = 0
                latents_st = latents_st * mask
                latents = latents * mask
                trajectory_feature = trajectory_feature * mask
            elif self.masking == "none":
                pass
            else:
                raise ValueError(f"Unknown masking type {self.masking}")
        joined_pred = self.decode(latents_st, state)
        return joined_pred, latents, trajectory_feature, masked_trajectory_feature


class VQContinuousVAE(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem (+1 for stop token)
        self.model = VQStepWiseTransformer(config, config.observation_dim)
        self.trajectory_embd = config.trajectory_embd
        self.vocab_size = config.vocab_size
        self.stop_token = config.vocab_size * config.transition_dim
        self.block_size = config.block_size
        self.observation_dim = config.observation_dim
        if "masking" in config:
            self.masking = config.masking
        else:
            self.masking = "none"

        self.action_dim = config.action_dim
        self.trajectory_length = config.block_size//config.transition_dim-1
        self.transition_dim = config.transition_dim
        self.action_weight = config.action_weight
        self.reward_weight = config.reward_weight
        self.value_weight = config.value_weight
        self.position_weight = config.position_weight
        self.first_action_weight = config.first_action_weight
        self.sum_reward_weight = config.sum_reward_weight
        self.last_value_weight = config.last_value_weight
        self.latent_step = config.latent_step

        self.padding_vector = torch.zeros(self.transition_dim-1)
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def set_padding_vector(self, padding):
        self.padding_vector = padding

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, EinLinear)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        if isinstance(self.model, SymbolWiseTransformer) or isinstance(self.model, VQStepWiseTransformer):
            no_decay.add('model.pos_emb')
            if self.model.bottleneck == "attention":
                no_decay.add('model.latent_pooling.query')
                no_decay.add('model.expand.query')
                no_decay.add('model.latent_pooling.attention.in_proj_weight')
                no_decay.add('model.expand.attention.in_proj_weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    @torch.no_grad()
    def encode(self, joined_inputs, terminals):
        b, t, joined_dimension = joined_inputs.size()
        mask = torch.ones_like(joined_inputs)
        weight = 0
        mask[:,:,0] = weight
        joined_inputs = joined_inputs * mask
        padded = torch.tensor(self.padding_vector, dtype=torch.float32,
                              device=joined_inputs.device).repeat(b, t, 1)
        terminal_mask = torch.clone(1 - terminals).repeat(1, 1, joined_inputs.shape[-1])
        joined_inputs = joined_inputs*terminal_mask+(1-terminal_mask)*padded
        trajectory_feature = self.model.encode(torch.cat([joined_inputs, terminals], dim=2))
        if self.model.ma_update:
            indices = vq(trajectory_feature, self.model.codebook.embedding)
        else:
            indices = vq(trajectory_feature, self.model.codebook.embedding.weight)
        return indices

    def decode(self, latent, state):
        return self.model.decode(latent, state)

    def decode_from_indices(self, indices, state):
        B, T = indices.shape
        if self.model.ma_update:
            latent = torch.index_select(self.model.codebook.embedding, dim=0, index=indices.flatten()).reshape([B, T, -1])
        else:
            latent = torch.index_select(self.model.codebook.embedding.weight, dim=0, index=indices.flatten()).reshape(
                [B, T, -1])
        if self.model.bottleneck == "attention":
            latent = torch.concat([latent, torch.zeros([B, self.trajectory_length//self.latent_step, latent.shape[2]],
                                                       device=latent.device)],
                                  dim=1)
        state = state[:,None,:]
        #print(latent.shape, state.repeat(latent.shape[0], 1, 1).shape)
        #print(latent.shape,state.shape)
        #print("shape checks:",latent.shape, state.repeat(latent.shape[0], 1, 1).shape)
        return self.model.decode(latent, state.repeat(latent.shape[0], 1, 1))

    def decode_for_ood(self, indices, state):
        B, T = indices.shape
        if self.model.ma_update:
            latent = torch.index_select(self.model.codebook.embedding, dim=0, index=indices.flatten()).reshape([B, T, -1])
        else:
            latent = torch.index_select(self.model.codebook.embedding.weight, dim=0, index=indices.flatten()).reshape(
                [B, T, -1])
        if self.model.bottleneck == "attention":
            latent = torch.concat([latent, torch.zeros([B, self.trajectory_length//self.latent_step, latent.shape[2]],
                                                       device=latent.device)],
                                  dim=1)
        state = state.unsqueeze(1)
        #print(indices.shape,state.shape)
        #print("shape checks:",latent.shape, state.shape)
        return self.model.decode(latent, state)

    def decode_for_ood_batch(self, indices, state):
        B, T = indices.shape
        if self.model.ma_update:
            latent = torch.index_select(self.model.codebook.embedding, dim=0, index=indices.flatten()).reshape([B, T, -1])
        else:
            latent = torch.index_select(self.model.codebook.embedding.weight, dim=0, index=indices.flatten()).reshape(
                [B, T, -1])
        if self.model.bottleneck == "attention":
            latent = torch.concat([latent, torch.zeros([B, self.trajectory_length//self.latent_step, latent.shape[2]],
                                                       device=latent.device)],
                                  dim=1)
        #state = state.unsqueeze(1)
        latent_expanded = latent.repeat(1, 512, 1)

        #print(latent_expanded.shape,state.shape)
        return self.model.decode(latent_expanded, state)

    def forward(self, joined_inputs, targets=None, mask=None, terminals=None, returnx=False):
        """
            joined_inputs : [ B x T x joined_dimension]
            values : [ B x 1 x 1 ]
        """
        joined_inputs = joined_inputs.to(dtype=torch.float32)
        b, t, joined_dimension = joined_inputs.size()
        padded = torch.tensor(self.padding_vector, dtype=torch.float32,
                              device=joined_inputs.device).repeat(b, t, 1)
        if terminals is not None:
            terminal_mask = torch.clone(1 - terminals).repeat(1, 1, joined_inputs.shape[-1])
            joined_inputs = joined_inputs*terminal_mask+(1-terminal_mask)*padded

        #state = joined_inputs[:,0,:self.observation_dim]
        state = joined_inputs[:, 0, 1:(self.observation_dim+1)]
        ## [ B x T x embedding_dim ]
        # forward the GPT model
        reconstructed, latents, feature, masked_trajectory_feature = self.model(torch.cat([joined_inputs, terminals], dim=2), state)
        pred_trajectory = torch.reshape(reconstructed[:, :, :-1], shape=[b, t, joined_dimension])
        pred_terminals = reconstructed[:, :, -1, None]

        # if we are given some desired targets also calculate the loss
        if targets is not None:
            #kl = torch.mean(-0.5 * torch.sum(1 + log_var - means.pow(2) - log_var.exp(), dim=1), dim=0)
            # weights = torch.cat([
            #     torch.ones(2, device=joined_inputs.device)*self.position_weight,
            #     torch.ones(self.observation_dim-2, device=joined_inputs.device),
            #     torch.ones(self.action_dim, device=joined_inputs.device) * self.action_weight,
            #     torch.ones(1, device=joined_inputs.device) * self.reward_weight,
            #     torch.ones(1, device=joined_inputs.device) * self.value_weight,
            # ])
            weights = torch.cat([
                torch.ones(1, device=joined_inputs.device) * self.value_weight,
                torch.ones(2, device=joined_inputs.device)*self.position_weight,
                torch.ones(self.observation_dim-2, device=joined_inputs.device),
                torch.ones(self.action_dim, device=joined_inputs.device) * self.action_weight,
                torch.ones(self.action_dim, device=joined_inputs.device) * self.action_weight,
                torch.ones(self.action_dim, device=joined_inputs.device) * self.action_weight,
            ])
            #print(pred_trajectory.shape, joined_inputs.shape)
            #print(pred_trajectory, joined_inputs)
            mse = F.mse_loss(pred_trajectory, joined_inputs, reduction='none')*weights[None, None, :]

            # first_action_loss = self.first_action_weight*F.mse_loss(joined_inputs[:, 0, self.observation_dim:self.observation_dim+self.action_dim],
            #                                                         pred_trajectory[:, 0, self.observation_dim:self.observation_dim+self.action_dim])
            # sum_reward_loss = self.sum_reward_weight*F.mse_loss(joined_inputs[:, :, -2].mean(dim=1),
            #                                                     pred_trajectory[:, :, -2].mean(dim=1))
            # last_value_loss = self.last_value_weight*F.mse_loss(joined_inputs[:, -1, -1],
            #                                                     pred_trajectory[:, -1, -1])
            first_action_loss = F.mse_loss(joined_inputs[:, 0, (1+self.observation_dim):(self.observation_dim+1) + self.action_dim],pred_trajectory[:, 0, (1+self.observation_dim):(self.observation_dim+1) + self.action_dim])
            value_loss = F.mse_loss(joined_inputs[:, :, 0].mean(dim=1),pred_trajectory[:, :, 0].mean(dim=1))
            current_state_loss = F.mse_loss(joined_inputs[:, 0, 1:(self.observation_dim+1)],pred_trajectory[:, 0, 1:(self.observation_dim+1)])
            next_state_loss = F.mse_loss(joined_inputs[:, 1, 1:(self.observation_dim + 1)],
                                           pred_trajectory[:, 1, 1:(self.observation_dim + 1)])
            cross_entropy = F.binary_cross_entropy(pred_terminals, torch.clip(terminals.float(), 0.0, 1.0))
            reconstruction_loss = (mse*mask*terminal_mask).mean()+cross_entropy
            #print(reconstruction_loss,first_action_loss,sum_reward_loss,last_value_loss)
            #reconstruction_loss = reconstruction_loss + first_action_loss + sum_reward_loss + last_value_loss
            #reconstruction_loss = reconstruction_loss + 1*value_loss + 1*next_state_loss + 1*first_action_loss
            reconstruction_loss = reconstruction_loss+ 0.3*value_loss + 0.3*next_state_loss + 0.3*first_action_loss
            #reconstruction_loss = torch.sqrt((mse * mask).sum(dim=1)).mean()
            if self.model.ma_update:
                loss_vq = 0
            else:
                loss_vq = F.mse_loss(latents, feature.detach())
            # Commitment objective
            loss_commit = F.mse_loss(masked_trajectory_feature, latents.detach())
            #loss_commit = 0
        else:
            reconstruction_loss = None
            loss_vq = None
            # Commitment objective
            loss_commit = None
        #loss_vq = F.mse_loss(feature, masked_trajectory_feature)
        loss_vq = F.l1_loss(feature, masked_trajectory_feature)

        return reconstructed, reconstruction_loss, loss_vq, loss_commit, current_state_loss, next_state_loss, value_loss, first_action_loss

class TransformerPrior(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()
        # input embedding stem (+1 for stop token)
        self.tok_emb = nn.Embedding(config.K, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.state_emb = nn.Linear(config.observation_dim, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        #self.n_heads = config.n_heads  # Add to config (e.g., n_heads=4)
        self.n_heads = 4
        # self.heads = nn.ModuleList([
        #     nn.Linear(config.n_embd, config.K, bias=False)
        #     for _ in range(self.n_heads)
        # ])

        self.head = nn.Linear(config.n_embd, config.K, bias=False)
        # self.head = EinLinear(config.transition_dim, config.n_embd, config.K, bias=False)
        self.observation_dim = config.observation_dim

        self.vocab_size = config.K
        self.block_size = config.block_size
        self.embedding_dim = config.n_embd
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, EinLinear)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, state, targets=None):
        """
            idx : [ B x T ]
            state: [ B ]
        """

        state = state.to(dtype=torch.float32)
        #batch_state = state.repeat(512, 1)        ## [ B x T x embedding_dim ]
        #print("batch state:",batch_state.shape)
        #batch_state_embeddings = self.state_emb(batch_state)[:, None]
        #print(batch_state_embeddings.shape)
        #print(idx.shape)
        if not idx is None:
            b, t = idx.size()
            #print(idx.size(), targets.size())
            #b = state.size(0)
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."
            token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
            #print(self.embedding_dim)
            #print(token_embeddings.shape)
            #print("shape 1",torch.zeros(size=(b, 1, self.embedding_dim)).to(token_embeddings).shape)
            #print("shape 2",token_embeddings.shape)
            token_embeddings = torch.cat([torch.zeros(size=(b, 1, self.embedding_dim)).to(token_embeddings), token_embeddings],
                                             dim=1)
            #print(token_embeddings)
        else:
            #b = 1
            b = state.size(0)  # Use the batch size from state if idx is None
            t = 0
            token_embeddings = torch.zeros(size=(b, 1, self.embedding_dim)).to(state)
            #print(token_embeddings)
        #print(token_embeddings.shape)
        ## [ 1 x T+1 x embedding_dim ]
        #print(t+1)
        position_embeddings = self.pos_emb[:, :t+1, :] # each position maps to a (learnable) vector
        state_embeddings = self.state_emb(state)[:, None]
        # print("tokens:", token_embeddings.shape)
        # print("pos:",position_embeddings.shape)
        # print("state:",state_embeddings.shape)
        ## [ B x T+1 x embedding_dim ]
        x = self.drop(token_embeddings + position_embeddings + state_embeddings)
        x = self.blocks(x)
        ## [ B x T+1 x embedding_dim ]
        x = self.ln_f(x)
        #print("before:",x.shape)
        #print("before:",x)
        logits = self.head(x)
        #print("logits", logits.shape)
        #print("after:", logits.shape)
        #print("logits:", logits)
        logits = logits.reshape(b, t + 1, self.vocab_size)
        logits = logits[:,:t+1]
        #print(logits.shape)
        # if we are given some desired targets also calculate the loss
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape([-1]), reduction='none')
            #print(logits.reshape(-1, self.vocab_size).shape,len(targets.reshape([-1])))
            #print(len(logits.reshape(-1, self.vocab_size)),len(logits.reshape(-1, self.vocab_size)[0]))
            loss = loss.mean()
        else:
            loss = None
        #print(logits.reshape(-1, self.vocab_size).shape)
        return logits, loss

    # def forward(self, idx, state, targets=None):
    #     """
    #         idx : [ B x T ]
    #         state: [ B ]
    #     """
    #     #
    #     state = state.to(dtype=torch.float32)
    #     # batch_state = state.repeat(512, 1)        ## [ B x T x embedding_dim ]
    #     # print("batch state:",batch_state.shape)
    #     # batch_state_embeddings = self.state_emb(batch_state)[:, None]
    #     # print(batch_state_embeddings.shape)
    #     if not idx is None:
    #         b, t = idx.size()
    #         # print(idx.size(), targets.size())
    #         # b = state.size(0)
    #         assert t <= self.block_size, "Cannot forward, model block size is exhausted."
    #         token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
    #         token_embeddings = torch.cat(
    #             [torch.zeros(size=(b, 1, self.embedding_dim)).to(token_embeddings), token_embeddings],
    #             dim=1)
    #
    #     else:
    #         # b = 1
    #         b = state.size(0)  # Use the batch size from state if idx is None
    #         t = 0
    #         token_embeddings = torch.zeros(size=(b, 1, self.embedding_dim)).to(state)
    #         # print(token_embeddings)
    #     # print(t+1)
    #     position_embeddings = self.pos_emb[:, :t + 1, :]  # each position maps to a (learnable) vector
    #     state_embeddings = self.state_emb(state)[:, None]
    #     x = self.drop(token_embeddings + position_embeddings + state_embeddings)
    #     x = self.blocks(x)
    #     x = self.ln_f(x)
    #     #print(x.shape)
    #     # Compute logits sequentially
    #     logits = []
    #     for head in self.heads:
    #         logits.append(head(x))  # Reuse shared trunk output
    #     logits = torch.stack(logits, dim=1)  # (B, n_heads, T+1, K)
    #     #print(idx.shape, state.shape, logits.shape, token_embeddings.shape)
    #
    #     # If targets provided, compute multi-token loss
    #
    #     if targets is not None:
    #         losses = []
    #         seq_length = targets.size(1)  # Original sequence length (T)
    #         for i in range(self.n_heads):
    #             valid_length = seq_length - i
    #             if valid_length <= 0:
    #                 continue  # No targets available for this head
    #             # Shift targets for i-th head (predicts token t+i)
    #             # shifted_targets = targets[:, i:i + logits.shape[2]]  # (B, T+1)
    #             # print(logits[:, i, :, :].reshape(-1, self.vocab_size).shape, shifted_targets.shape,logits.shape,logits.shape[2])
    #             # loss = F.cross_entropy(
    #             #     logits[:, i, :, :].reshape(-1, self.vocab_size),
    #             #     shifted_targets.reshape(-1),
    #             #     reduction='mean'
    #             # )
    #             # losses.append(loss)
    #
    #             # Slice logits and targets to match valid length
    #             head_logits = logits[:, i, :valid_length, :]  # (B, valid_length, vocab_size)
    #             head_targets = targets[:, i:i + valid_length]  # (B, valid_length)
    #             #print(head_logits.shape, head_targets.shape)
    #             loss = F.cross_entropy(
    #                 head_logits.reshape(-1, self.vocab_size),
    #                 head_targets.reshape(-1),
    #                 reduction='mean'
    #             )
    #             losses.append(loss)
    #         total_loss = torch.mean(torch.stack(losses))
    #     else:
    #         total_loss = None
    #
    #     return logits, total_loss

    # def generate(self, state, max_length=100, n_speculative=3):
    #     generated = []
    #     current_seq = None  # Initialize with start token
    #
    #     for _ in range(max_length):
    #         # 1. Predict n_speculative tokens using all heads
    #         logits, _ = self(current_seq, state)  # (B, n_heads, T+1, K)
    #
    #         # 2. Draft tokens from heads 1 to n_speculative
    #         draft_tokens = []
    #         for i in range(1, n_speculative + 1):
    #             draft = logits[:, i, -1, :].argmax(dim=-1)
    #             draft_tokens.append(draft)
    #
    #         # 3. Verify using head 0 (main model)
    #         full_candidates = torch.cat([current_seq] + draft_tokens, dim=-1)
    #         verification_logits, _ = self(full_candidates, state)
    #         verified = verification_logits[:, 0].argmax(dim=-1)
    #
    #         # 4. Find first mismatch
    #         match_mask = (verified == full_candidates).all(dim=-1)
    #         if match_mask.all():
    #             # Accept all draft tokens
    #             generated.extend(draft_tokens)
    #             current_seq = full_candidates
    #         else:
    #             # Fallback to autoregressive
    #             next_token = verification_logits[:, 0, -1, :].argmax(dim=-1)
    #             generated.append(next_token)
    #             current_seq = torch.cat([current_seq, next_token], dim=-1)
    #
    #     return current_seq

    '''
    def forward(self, idx, state, targets=None):
        """
        idx : [ B x T ] or None
        state: [ B x M ]
        """
        state = state.to(dtype=torch.float32)
        b, _ = state.size()  # Adjusted for batched state input

        if idx is not None:
            t = idx.size(1)
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."
            token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
            token_embeddings = torch.cat([torch.zeros(size=(b, 1, self.embedding_dim)).to(token_embeddings.device), token_embeddings], dim=1)
        else:
            t = 0
            token_embeddings = torch.zeros(size=(b, 1, self.embedding_dim)).to(state.device)

        position_embeddings = self.pos_emb[:, :t+1, :]  # Adjust position embeddings for batch
        state_embeddings = self.state_emb(state)[:, None, :]  # Adjust for 2D state input, adding a new dimension for time
        x = self.drop(token_embeddings + position_embeddings + state_embeddings.expand(-1, t+1, -1))  # Make sure the dimensions match
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.head(x)
        logits = logits.reshape(b, t + 1, self.vocab_size)
        logits = logits[:, :t+1, :]  # Adjust slice for batch

        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape([-1]), reduction='none')
            loss = loss.mean()
        else:
            loss = None

        return logits, loss
    '''