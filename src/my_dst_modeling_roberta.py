# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch RoBERTa model. """


import logging
import warnings
import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.configuration_roberta import RobertaConfig
# from transformers.file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_callable
# from transformers.modeling_bert import BertLayerNorm, BertModel, BertPreTrainedModel, gelu
from transformers.modeling_bert import ACT2FN, BertLayerNorm, BertPreTrainedModel, gelu
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer


logger = logging.getLogger(__name__)

_TOKENIZER_FOR_DOC = "RobertaTokenizer"

ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "roberta-base",
    "roberta-large",
    "roberta-large-mnli",
    "distilroberta-base",
    "roberta-base-openai-detector",
    "roberta-large-openai-detector",
    # See all RoBERTa models at https://huggingface.co/models?filter=roberta
]

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.utterance_type_embeddings = nn.Embedding(4, config.hidden_size)


        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None,
                inputs_embeds=None, utterance_type_ids=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        batch_size = input_ids.shape[0]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        dim = inputs_embeds.shape[2]
        dtype = inputs_embeds.dtype
        if utterance_type_ids is None:
            # utterance_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            utterance_type_embeddings = torch.zeros_like(inputs_embeds, device=device)
        else:
            #
            # utterance_type_embeddings = torch.cat(  # DS~~~ + CLS (becuase of cls + 1)
            #     [utterance_type_ids.new_zeros((batch_size, len(self.config.domain_cls_tokens) + 1, dim),
            #                                   dtype=torch.float),
            #      self.utterance_type_embeddings(utterance_type_ids)], 1)

            if self.config.ds_type == 'split':
                ds_utterance_ids = torch.cat([2 * torch.ones((batch_size, 5), device=device, dtype=torch.long),
                                              3 * torch.ones((batch_size, 17), device=device, dtype=torch.long)], 1)
            elif self.config.ds_type=='merged':
                ds_utterance_ids = 2 * torch.ones((batch_size, 30), device=device, dtype=torch.long)

            if self.config.ds_type in ['split', 'merged']:
                utterance_type_embeddings = torch.cat(
                    [self.utterance_type_embeddings(ds_utterance_ids),
                        torch.zeros((batch_size, 1, dim), dtype=dtype, device=device),
                     self.utterance_type_embeddings(utterance_type_ids)], 1)
            elif self.config.ds_type == 'none':
                utterance_type_embeddings = torch.cat(
                    [torch.zeros((batch_size, 1, dim), dtype=dtype, device=device),
                     self.utterance_type_embeddings(utterance_type_ids)], 1)


        if position_ids is None:

            if self.config.ds_type in ['split', 'merged']:
                seq_length = input_shape[1] - len(self.config.domain_cls_tokens)

                position_embeddings = torch.cat(
                    [
                    torch.zeros((batch_size, len(self.config.domain_cls_tokens), dim), dtype=dtype, device=device),
                    self.position_embeddings(torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).expand((batch_size, -1)))
                    ], 1)
            else:
                seq_length = input_shape[1]
                position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand(input_shape)
                position_embeddings = self.position_embeddings(position_ids)

        # if position_ids is None:
        #     position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        #     position_ids = position_ids.unsqueeze(0).expand(input_shape)
        #     position_embeddings = self.position_embeddings(position_ids)

        # else:
        #     seq_length = input_shape[1]
        #     device = input_ids.device if input_ids is not None else inputs_embeds.device
        #     if position_ids is None:
        #         position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        #         position_ids = position_ids.unsqueeze(0).expand(input_shape)
        #     position_embeddings = self.position_embeddings(position_ids)


        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings + utterance_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
        self.utterance_type_embeddings = nn.Embedding(4, config.hidden_size)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None,
                utterance_type_ids=None):
        # if position_ids is None:
        #     if input_ids is not None:
        #         # Create the position ids from the input token ids. Any padded tokens remain padded.
        #         position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
        #     else:
        #         position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        return super().forward(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds,
            utterance_type_ids=utterance_type_ids
        )

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """ We are provided embeddings directly. We cannot infer which are padded so just generate
        sequential position ids.
        :param torch.Tensor inputs_embeds:
        :return torch.Tensor:
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


ROBERTA_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.
    Parameters:
        config (:class:`~transformers.RobertaConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

ROBERTA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using :class:`transformers.RobertaTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.__call__` for details.
            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the attentions tensors of all attention layers are returned. See ``attentions`` under returned tensors for more detail.
"""


# @add_start_docstrings(
#     "The bare RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
#     ROBERTA_START_DOCSTRING,
# )


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
            domain_attn_mask=None
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        if domain_attn_mask is not None:
            attention_scores = attention_scores + domain_attn_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
            domain_attn_mask=None
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions,
            domain_attn_mask
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
            domain_attn_mask=None
    ):
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, output_attentions=output_attentions,
            domain_attn_mask=domain_attn_mask
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
            domain_attn_mask=None
    ):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    domain_attn_mask
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                    domain_attn_mask
                )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.

    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762

    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.pooler = BertPooler(config)

        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_domain = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_slot = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()

        self.cls_index = len(self.config.domain_cls_tokens)

        self.is_domain_attention = self.config.is_domain_attention
        self.num_domain_cls_tokens = len(self.config.domain_cls_tokens)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    # @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="bert-base-uncased")
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
            utterance_type_ids=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds,
            utterance_type_ids=utterance_type_ids
        )

        if self.is_domain_attention:
            domain_attn_mask = None
        else:
            domain_attn_mask = torch.zeros((1, 1, input_ids.shape[1], input_ids.shape[1]), device=device, dtype=self.dtype)
            domain_attn_mask[:, :, :self.num_domain_cls_tokens, :self.num_domain_cls_tokens] = -10000.0
            domain_attn_mask[0, 0, :self.num_domain_cls_tokens, :self.num_domain_cls_tokens] \
                += torch.eye(self.num_domain_cls_tokens, dtype=self.dtype, device=device) * 10000.0

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            # output_attentions=output_attentions,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            domain_attn_mask=domain_attn_mask
        )

        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, self.cls_index]))

        if self.config.ds_type == 'split':
            domain_output = self.pooler_activation(self.pooler_domain(sequence_output[:, 0:5]))
            slot_output = self.pooler_activation(self.pooler_slot(sequence_output[:, 5:17 + 5]))
            domain_slot_output = None
        elif self.config.ds_type == 'merged':
            domain_slot_output = self.pooler_activation(self.pooler_domain(sequence_output[:, 0:30]))
            domain_output = None
            slot_output = None
        elif self.config.ds_type == 'none':
            domain_output = pooled_output
            slot_output = pooled_output
            domain_slot_output = pooled_output


        outputs = (pooled_output, domain_output, slot_output, domain_slot_output) #+ encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class RobertaModel(BertModel):
    """
    This class overrides :class:`~transformers.BertModel`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.embeddings = RobertaEmbeddings(config)

        # self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        # # self.pooler_pooler_domain_slot = nn.Linear(config.hidden_size, config.hidden_size)
        # self.pooler_domain = nn.Linear(config.hidden_size, config.hidden_size)
        # self.pooler_slot = nn.Linear(config.hidden_size, config.hidden_size)
        # self.pooler_activation = nn.Tanh()

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


# @add_start_docstrings("""RoBERTa Model with a `language modeling` head on top. """, ROBERTA_START_DOCSTRING)
class RobertaForMaskedLM(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    # @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    # @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="roberta-base")
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                DeprecationWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x


# @add_start_docstrings(
#     """RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer
#     on top of the pooled output) e.g. for GLUE tasks. """,
#     ROBERTA_START_DOCSTRING,
# )
class RobertaForDST(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        config.classifier_dropout_prob = 0.1

        self.roberta = RobertaModel(config)
        # self.classifier = RobertaClassificationHead(config)

        self.num_domain_slot = 30
        self.num_gate_label = 1
        # self.domain_classifier = nn.Linear(config.hidden_size, 1)
        # self.dropout = nn.Dropout(config.classifier_dropout_prob)
        if config.ds_type == 'split':
            gate_hidden = config.hidden_size * 2
        elif self.config.ds_type == 'merged':
            gate_hidden = config.hidden_size
        elif self.config.ds_type == 'none':
            gate_hidden = config.hidden_size

        self.domain_gate_classifier_list = \
            nn.ModuleList([nn.Linear(gate_hidden, self.num_gate_label) for _ in range(self.num_domain_slot)])
        # self.domain_gate_classifier = nn.Linear(config.hidden_size, self.num_gate_label)
        self.dropout_gate = nn.Dropout(config.classifier_dropout_prob)

        import pickle
        if config.version == '1':
            with open(f'data/trade_slot_value_dict.pkl', 'rb') as f:
                slot_value_dict = pickle.load(f)
        elif config.version == '2':
            with open('data/WOZ22_slot_value_dict', 'rb') as f:
                slot_value_dict = pickle.load(f)

        if self.config.ds_type == 'split':
            with open(f'trade-dst/data/processed_train_fixed', 'rb') as f:
                data = pickle.load(f)
            _, _, slot_type_list = data

            domain = set()
            slot = set()
            for x in slot_type_list:
                a, b = x.split('-')
                domain.add(a)
                slot.add(b)
            domain = sorted(domain)

            slot = sorted(slot)
            self.domain_slot_idx = list()
            for x in slot_type_list:
                a, b = x.split('-')
                self.domain_slot_idx.append((domain.index(a), slot.index(b)))
            slot_hidden = config.hidden_size * 3
        elif self.config.ds_type == 'merged':
            slot_hidden = config.hidden_size * 2
        elif self.config.ds_type == 'none':
            slot_hidden = config.hidden_size

        self.domain_slot_value_classifier_list = \
            nn.ModuleList(
                [nn.Linear(slot_hidden, len(slot_value_list)) for slot_value_list in slot_value_dict.values()])

        self.dropout_slot = nn.Dropout(config.classifier_dropout_prob)

        slot_type_list = ['hotel-price range',
                          'hotel-type',
                          'hotel-parking',
                          'hotel-book stay',
                          'hotel-book day',
                          'hotel-book people',
                          'hotel-area',
                          'hotel-stars',
                          'hotel-internet',
                          'train-destination',
                          'train-day',
                          'train-departure',
                          'train-arrive by',
                          'train-book people',
                          'train-leave at',
                          'attraction-area',
                          'restaurant-food',
                          'restaurant-price range',
                          'restaurant-area',
                          'attraction-name',
                          'restaurant-name',
                          'attraction-type',
                          'hotel-name',
                          'taxi-leave at',
                          'taxi-destination',
                          'taxi-departure',
                          'restaurant-book time',
                          'restaurant-book day',
                          'restaurant-book people',
                          'taxi-arrive by']

        # from transformers import AlbertTokenizer
        #
        # tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        # # tokenizer.add_special_tokens({'additional_special_tokens': ['[SOS]']})
        #
        # slot_type_list = [x + ' ' + tokenizer.eos_token for x in slot_type_list]
        #
        # domain_slot_tokenized = tokenizer(slot_type_list, add_special_tokens=False, return_token_type_ids=False, padding=True)
        #
        # domain_slot_token_input = torch.tensor(domain_slot_tokenized['input_ids'])
        #
        # num_domain = domain_slot_token_input.shape[0]
        #
        # vocab = torch.unique(torch.tensor(domain_slot_tokenized ['input_ids']))
        # hidden_size = 128
        # self.domain_slot_word_embedding = nn.Embedding(len(vocab) + 1, hidden_size)
        # self.domain_slot_word_embedding.weight.data[1:] = self.albert.embeddings.word_embeddings(vocab)
        #
        # for new_i, old_i in enumerate(vocab, start=1): # 0 reserved for sos
        #     domain_slot_token_input[domain_slot_token_input == old_i] = new_i
        #
        # self.domain_slot_token_input = domain_slot_token_input
        #
        # self.domain_slot_attn_mask = torch.cat([torch.zeros(num_domain, 1, dtype=torch.long),
        #                                         torch.tensor(domain_slot_tokenized['attention_mask'])], 1)

        # encoder_layer = torch.nn.TransformerEncoderLayer(hidden_size, nhead=6, dim_feedforward=hidden_size * 4, dropout=0.0)
        # trsf_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)

        # self.domain_slot_gru = torch.nn.GRU(input_size=hidden_size, hidden_size = hidden_size, batch_first=True)

        # self.domain_slot_encoded = self.albert.embeddings.word_embeddings(domain_slot_tokenized)

        # label_smoothing= 0.01
        # self.label_smooth_loss_slot_value = nn.ModuleList([LabelSmoothingLoss(label_smoothing, len(slot_value_list), -100)
        #                                                    for slot_value_list in slot_value_dict.values()])
        # self.label_smooth_loss_domain = LabelSmoothingLoss(label_smoothing, self.num_domain)
        # self.label_smooth_loss_gate = LabelSmoothingLoss(label_smoothing, self.num_gate_label)
        self.label_smooth_loss_slot_value = CrossEntropyLoss()
        # self.label_smooth_loss_domain = CrossEntropyLoss()
        # self.label_smooth_loss_gate = CrossEntropyLoss()
        self.label_smooth_loss_gate = torch.nn.BCEWithLogitsLoss()

        self.init_weights()

    def resize_positional_embeddings(self, new_max_position_length:int):
        old_position_embeddings = self.roberta.embeddings.position_embeddings
        old_num_tokens, old_embedding_dim = old_position_embeddings.weight.size()
        new_position_embeddings = nn.Embedding(new_max_position_length, old_embedding_dim)
        self._init_weights(new_max_position_length)
        new_position_embeddings.weight.data[:old_num_tokens, :] = old_position_embeddings.weight.data[:old_num_tokens, :]
        self.roberta.embeddings.position_embeddings = new_position_embeddings


    # @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    # @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="roberta-base")
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
            utterance_type_ids=None,
            gate_labels=None,
            slot_labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            utterance_type_ids=utterance_type_ids,
        )
        pooled_output, pooled_domain_output, pooled_slot_output, pooled_domain_slot_output = outputs

        # pooled_output, domain_slot_pooled_output = outputs

        # slot_values = self.albert(slot_values,
        #                           attention_mask=slot_values_attn_mask,
        #                           run_slot_value=True,
        #                           dialog_cls_token_embedding=pooled_output
        #                           )

        # domain_pooled_output, domain_slot_pooled_output = torch.split(domain_and_domain_slot_pooled_output, [1,30], dim=1)
        # domain_pooled_output = domain_pooled_output.squeeze(1)
        # sequence_output = self.dropout(sequence_output)
        # domain_prediction_scores = self.domain_classifier(sequence_output)

        outputs = dict()
        # outputs = (domain_prediction_scores, domain_gate_prediction_scores) #+ outputs[2:]  # add hidden states and attention if they are here

        total_loss = None
        # if domain_labels is not None:
        #     loss_fct = self.label_smooth_loss_domain
        #     # domain_pred_pooled_output = self.dropout(pooled_output)
        #     # domain_pred_pooled_output = self.dropout(domain_pooled_output)
        #     domain_prediction_scores = self.domain_classifier(domain_pooled_output)
        #     domain_loss = loss_fct(domain_prediction_scores, domain_labels.view(-1))
        #     if total_loss is None:
        #         total_loss = domain_loss
        #     else:
        #         total_loss += domain_loss
        #     outputs['domain_logits'] = domain_prediction_scores
        #     outputs['domain_loss'] = domain_loss

        if gate_labels is not None:
            loss_fct = self.label_smooth_loss_gate

            if self.config.ds_type == 'split':
                domain_gate_prediction_scores = [mlp(
                    self.dropout_gate(
                        torch.cat([pooled_domain_output[:, x[0]], pooled_slot_output[:, x[1]]], 1)))
                    for mlp, x in
                    zip(self.domain_gate_classifier_list, self.domain_slot_idx)]
            elif self.config.ds_type == 'merged':
                domain_gate_prediction_scores = [mlp(self.dropout_gate(pooled_domain_slot_output[:, x]))
                                                 for x, mlp in enumerate(self.domain_gate_classifier_list)]
            elif self.config.ds_type == 'none':
                domain_gate_prediction_scores = [mlp(self.dropout_gate(pooled_domain_slot_output[:, x]))
                                                 for x, mlp in enumerate(self.domain_gate_classifier_list)]

            assert (gate_labels <= 2).all(), f'gate label gone wrong in dst_modeling_albert, {gate_labels}'

            domain_gate_prediction_scores = torch.stack(domain_gate_prediction_scores).permute(1, 0, 2)
            # gate_loss = loss_fct(domain_gate_prediction_scores.reshape(-1, self.num_gate_label), gate_labels.view(-1))
            gate_loss = loss_fct(domain_gate_prediction_scores.reshape(-1, self.num_gate_label),
                                 gate_labels.view(-1, self.num_gate_label).to(domain_gate_prediction_scores.dtype))

            assert not torch.isnan(gate_loss), 'gate loss nan'
            assert not torch.isinf(gate_loss), f'gate loss inf'
            # sentence_order_loss = loss_fct(sop_sept_scores.view(-1, self.config.sept_num_labels), sentence_order_label.view(-1))
            if total_loss is None:
                total_loss = gate_loss
            else:
                total_loss += gate_loss
            outputs['domain_gate_logits'] = domain_gate_prediction_scores
            outputs['domain_gate_loss'] = gate_loss

            assert not torch.isnan(outputs['domain_gate_logits']).any(), 'domain_gate_logit nan'

        if slot_labels is not None:
            # domain_slot_value_prediction_scores = [mlp(self.dropout_slot(
            #     torch.cat([pooled_output, domain_slot_pooled_output[:, i]], 1))) for i, mlp in
            #     enumerate(self.domain_slot_value_classifier_list)]
            if self.config.ds_type == 'split':
                domain_slot_value_prediction_scores = [
                    mlp(
                        self.dropout_slot(
                            torch.cat([pooled_output, pooled_domain_output[:, x[0]], pooled_slot_output[:, x[1]]], 1)))
                    for mlp, x in zip(self.domain_slot_value_classifier_list, self.domain_slot_idx)]
            elif self.config.ds_type == 'merged':
                domain_slot_value_prediction_scores = [
                    mlp(
                        self.dropout_slot(
                            torch.cat([pooled_output, pooled_domain_slot_output[:, x]], 1)))
                    for x, mlp in enumerate(self.domain_slot_value_classifier_list)]
            else:
                domain_slot_value_prediction_scores = [
                    mlp(
                        self.dropout_slot(pooled_output))
                    for x, mlp in enumerate(self.domain_slot_value_classifier_list)]

            for idx, pred_score in enumerate(domain_slot_value_prediction_scores):
                # loss_fct_tmp = self.label_smooth_loss_slot_value[idx]
                loss_fct_tmp = self.label_smooth_loss_slot_value

                # try:
                if idx == 0:
                    slot_loss = loss_fct_tmp(pred_score, slot_labels[:, idx])
                else:
                    slot_loss += loss_fct_tmp(pred_score, slot_labels[:, idx])
                assert not torch.isnan(slot_loss), f'slot loss nan, {idx}'
                assert not torch.isinf(slot_loss), f'slot loss inf, {idx}'
                for x in domain_slot_value_prediction_scores:
                    assert not torch.isnan(x).any(), 'slot pred nan'
                # except:
                #     print('WHAT * 2222')
            outputs['slot_value_logits_list'] = domain_slot_value_prediction_scores
            outputs['slot_value_loss'] = slot_loss

            if total_loss is None:
                total_loss = slot_loss
            else:
                total_loss += slot_loss

        outputs['loss'] = total_loss

        return outputs  # (loss), logits, (hidden_states), (attentions)


# @add_start_docstrings(
#     """Roberta Model with a multiple choice classification head on top (a linear layer on top of
#     the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
#     ROBERTA_START_DOCSTRING,
# )
class RobertaForMultipleChoice(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    # @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING.format("(batch_size, num_choices, sequence_length)"))
    # @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="roberta-base")
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor`` of shape `(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification loss.
        classification_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
            `num_choices` is the second dimension of the input tensors. (see `input_ids` above).
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


# @add_start_docstrings(
#     """Roberta Model with a token classification head on top (a linear layer on top of
#     the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
#     ROBERTA_START_DOCSTRING,
# )
class RobertaForTokenClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    # @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    # @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="roberta-base")
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, hidden_size, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = features  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


# @add_start_docstrings(
#     """Roberta Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
#     the hidden-states output to compute `span start logits` and `span end logits`). """,
#     ROBERTA_START_DOCSTRING,
# )
class RobertaForQuestionAnswering(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    # @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    # @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="roberta-base")
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """ Replace non-padding symbols with their position numbers. Position numbers begin at
    padding_idx+1. Padding symbols are ignored. This is modified from fairseq's
    `utils.make_positions`.
    :param torch.Tensor x:
    :return torch.Tensor:
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx