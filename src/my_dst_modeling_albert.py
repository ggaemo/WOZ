# coding=utf-8
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
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
"""PyTorch ALBERT model. """

import logging
import math
import os
import warnings

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.configuration_albert import AlbertConfig
from transformers.file_utils import add_code_sample_docstrings, add_start_docstrings#, add_start_docstrings_to_callable
from transformers.modeling_bert import ACT2FN, BertSelfAttention, prune_linear_layer
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices


logger = logging.getLogger(__name__)

_TOKENIZER_FOR_DOC = "AlbertTokenizer"


ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "albert-base-v1",
    "albert-large-v1",
    "albert-xlarge-v1",
    "albert-xxlarge-v1",
    "albert-base-v2",
    "albert-large-v2",
    "albert-xlarge-v2",
    "albert-xxlarge-v2",
    # See all ALBERT models at https://huggingface.co/models?filter=albert
]
BertLayerNorm = torch.nn.LayerNorm

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.utterance_type_embeddings = nn.Embedding(2, config.hidden_size)


        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, utterance_type_ids=None):
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
                ds_utterance_ids = torch.cat([2 * torch.ones((batch_size, 5), device=device, dtype=torch.long), 3 * torch.ones((batch_size, 17), device=device, dtype=torch.long)], 1)
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
                seq_length = input_shape[1] - len(self.config.domain_special_tokens)

                position_embeddings = torch.cat(
                    [
                        torch.zeros((batch_size, len(self.config.domain_special_tokens), dim), dtype=dtype, device=device),
                        self.position_embeddings(
                            torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(
                                (batch_size, -1)))
                    ], 1)
            else:
                seq_length = input_shape[1]
                position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand(input_shape)
                position_embeddings = self.position_embeddings(position_ids)

            # position_embeddings = position_embeddings.unsqueeze(0).expand(input_shape)

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


def load_tf_weights_in_albert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model."""
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        print(name)

    for name, array in zip(names, arrays):
        original_name = name

        # If saved from the TF HUB module
        name = name.replace("module/", "")

        # Renaming and simplifying
        name = name.replace("ffn_1", "ffn")
        name = name.replace("bert/", "albert/")
        name = name.replace("attention_1", "attention")
        name = name.replace("transform/", "")
        name = name.replace("LayerNorm_1", "full_layer_layer_norm")
        name = name.replace("LayerNorm", "attention/LayerNorm")
        name = name.replace("transformer/", "")

        # The feed forward layer had an 'intermediate' step which has been abstracted away
        name = name.replace("intermediate/dense/", "")
        name = name.replace("ffn/intermediate/output/dense/", "ffn_output/")

        # ALBERT attention was split between self and output which have been abstracted away
        name = name.replace("/output/", "/")
        name = name.replace("/self/", "/")

        # The pooler is a linear layer
        name = name.replace("pooler/dense", "pooler")

        # The classifier was simplified to predictions from cls/predictions
        name = name.replace("cls/predictions", "predictions")
        name = name.replace("predictions/attention", "predictions")

        # Naming was changed to be more explicit
        name = name.replace("embeddings/attention", "embeddings")
        name = name.replace("inner_group_", "albert_layers/")
        name = name.replace("group_", "albert_layer_groups/")

        # Classifier
        if len(name.split("/")) == 1 and ("output_bias" in name or "output_weights" in name):
            name = "classifier/" + name

        # No ALBERT model currently handles the next sentence prediction task
        if "seq_relationship" in name:
            name = name.replace("seq_relationship/output_", "sop_classifier/classifier/")
            name = name.replace("weights", "weight")

        name = name.split("/")

        # Ignore the gradients applied by the LAMB/ADAM optimizers.
        if (
            "adam_m" in name
            or "adam_v" in name
            or "AdamWeightDecayOptimizer" in name
            or "AdamWeightDecayOptimizer_1" in name
            or "global_step" in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            continue

        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]

            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]

        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {} from {}".format(name, original_name))
        pointer.data = torch.from_numpy(array)

    return model


class AlbertEmbeddings(BertEmbeddings):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__(config)

        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)
        self.utterance_type_embeddings = nn.Embedding(2, config.embedding_size)
        self.LayerNorm = torch.nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)




class AlbertAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)

        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.num_attention_heads, self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)
        self.dense = prune_linear_layer(self.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_ids, attention_mask=None, head_mask=None, output_attentions=False, domain_attn_mask=None):
        mixed_query_layer = self.query(input_ids)
        mixed_key_layer = self.key(input_ids)
        mixed_value_layer = self.value(input_ids)

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

        # Should find a better way to do this
        w = (
            self.dense.weight.t()
            .view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
            .to(context_layer.dtype)
        )
        b = self.dense.bias.to(context_layer.dtype)

        projected_context_layer = torch.einsum("bfnd,ndh->bfh", context_layer, w) + b
        projected_context_layer_dropout = self.dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(input_ids + projected_context_layer_dropout)
        return (layernormed_context_layer, attention_probs) if output_attentions else (layernormed_context_layer,)


class AlbertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.full_layer_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = AlbertAttention(config)
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]

    def forward(
        self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, output_hidden_states=False,
            domain_attn_mask=None
    ):
        attention_output = self.attention(hidden_states, attention_mask, head_mask, output_attentions, domain_attn_mask)
        ffn_output = self.ffn(attention_output[0])
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])

        return (hidden_states,) + attention_output[1:]  # add attentions if we output them


class AlbertLayerGroup(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.albert_layers = nn.ModuleList([AlbertLayer(config) for _ in range(config.inner_group_num)])

    def forward(
        self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, output_hidden_states=False,
            domain_attn_mask=None
    ):
        layer_hidden_states = ()
        layer_attentions = ()

        for layer_index, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(hidden_states, attention_mask, head_mask[layer_index], output_attentions, output_hidden_states, domain_attn_mask)
            hidden_states = layer_output[0]

            if output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)

            if output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (layer_hidden_states,)
        if output_attentions:
            outputs = outputs + (layer_attentions,)
        return outputs  # last-layer hidden state, (layer hidden states), (layer attentions)


class AlbertTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.albert_layer_groups = nn.ModuleList([AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)])

    def forward(
        self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, output_hidden_states=False,
            domain_attn_mask=None
    ):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        all_attentions = ()

        if output_hidden_states:
            all_hidden_states = (hidden_states,)

        for i in range(self.config.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

            # Index of the hidden group
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                attention_mask,
                head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group],
                output_attentions,
                output_hidden_states,
                domain_attn_mask
            )
            hidden_states = layer_group_output[0]

            if output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class AlbertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = AlbertConfig
    base_model_prefix = "albert"

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


ALBERT_START_DOCSTRING = r"""

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Args:
        config (:class:`~transformers.AlbertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

ALBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.AlbertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_gs - 1]``.

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
            If set to ``True``, the attentions tensors of all attention layers are returned. See ``fattentions`` under returned tensors for more detail.
"""



class AlbertModel(AlbertPreTrainedModel):

    config_class = AlbertConfig
    load_tf_weights = load_tf_weights_in_albert
    base_model_prefix = "albert"

    def __init__(self, config):
        super().__init__(config)

        config.sept_token_id = 30000
        # config.value_sept_token_id = 30001
        logger.info(f'@@@IMPORTANT@@@@@@@ FOR ALBERT sept_token id is {config.sept_token_id}')

        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertTransformer(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_domain_slot = nn.Linear(config.hidden_size, config.hidden_size)
        # self.pooler_pooler_domain_slot = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_domain = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_slot = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()

        self.cls_index = len(self.config.domain_special_tokens)

        self.is_domain_attention = self.config.is_domain_attention
        self.num_domain_slot_special_tokens = len(self.config.domain_special_tokens)

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
            ALBERT has a different architecture in that its layers are shared across groups, which then has inner groups.
            If an ALBERT model has 12 hidden layers and 2 hidden groups, with two inner groups, there
            is a total of 4 different layers.

            These layers are flattened: the indices [0,1] correspond to the two inner groups of the first hidden layer,
            while [2,3] correspond to the two inner groups of the second hidden layer.

            Any layer with in index other than [0,1,2,3] will result in an error.
            See base class PreTrainedModel for more information about head pruning
        """
        for layer, heads in heads_to_prune.items():
            group_idx = int(layer / self.config.inner_group_num)
            inner_group_idx = int(layer - group_idx * self.config.inner_group_num)
            self.encoder.albert_layer_groups[group_idx].albert_layers[inner_group_idx].attention.prune_heads(heads)

    # @add_start_docstrings_to_callable(ALBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="albert-base-v2")
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        utterance_type_ids= None,
        domain_slot_embeddings=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.AlbertConfig`) and inputs:
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

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)



        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds,
            utterance_type_ids=utterance_type_ids
        )

        # if self.config.ds_type == 'split':
        #     pass
        #     # embedding_output[:, 0:5] +=
        #     # domain_output = self.pooler_activation(self.pooler_domain(sequence_output[:, 0:5]))
        #     # slot_output = self.pooler_activation(self.pooler_slot(sequence_output[:, 5:17 + 5]))
        #     # domain_slot_output = None
        # elif self.config.ds_type == 'merged':
        #     embedding_output[:, 0:30] += domain_slot_embeddings

        if self.is_domain_attention:
            domain_attn_mask = None
        else:
            domain_attn_mask = torch.zeros((1, 1, input_ids.shape[1], input_ids.shape[1]), device=device, dtype=self.dtype)
            domain_attn_mask[:, :, :self.num_domain_slot_special_tokens, :self.num_domain_slot_special_tokens] = -10000.0
            domain_attn_mask[0, 0, :self.num_domain_slot_special_tokens, :self.num_domain_slot_special_tokens] \
                += torch.eye(self.num_domain_slot_special_tokens, dtype=self.dtype, device=device) * 10000.0

        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            domain_attn_mask=domain_attn_mask
        )

        sequence_output = encoder_outputs[0]

        # if run_slot_value:
        #     value_list = list()
        #     value_sept_mask = input_ids == self.config.value_sept_token_id
        #     for i in range(30):
        #         value_sept_mask_sept = value_sept_mask[:, i]
        #         sept_value_tmp = sequence_output[value_sept_mask_sept]
        #         value_list.append(sept_value_tmp)
        #     outputs = value_list
        # else:
            ## DOMAIN
        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, self.cls_index]))
        if self.config.ds_type == 'split':
            domain_output = self.pooler_activation(self.pooler_domain(sequence_output[:, 0:5]))
            slot_output = self.pooler_activation(self.pooler_slot(sequence_output[:, 5:17+5]))
            domain_slot_output = None
        elif self.config.ds_type == 'merged':
            domain_slot_output = self.pooler_activation(self.pooler_domain_slot(sequence_output[:, 0:30]))
            domain_output = None
            slot_output = None
        elif self.config.ds_type == 'none':
            domain_output = pooled_output
            slot_output = pooled_output
            domain_slot_output = pooled_output

        # domain_token_mask = input_ids >= min(self.config.domain_cls_tokens)
        # pooled_domain_output = self.pooler_activation(self.pooler_domain_slot(sequence_output[domain_token_mask]))

        # pooled_domain_output = pooled_domain_output.view(pooled_output.shape[0], len(self.config.domain_cls_tokens), -1)

        # outputs = (pooled_output, domain_slot_output)

        outputs = (pooled_output, domain_output, slot_output, domain_slot_output, sequence_output) #+ encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs



@add_start_docstrings(
    """Albert Model with two heads on top as done during the pre-training: a `masked language modeling` head and
    a `sentence order prediction (classification)` head. """,
    ALBERT_START_DOCSTRING,
)


class AlbertForDST(AlbertPreTrainedModel):
    def __init__(self, config):
        # config.num_labels = 4  # 0 1 2 3
        super().__init__(config)
        self.albert = AlbertModel(config)
        # self.predictions = AlbertMLMHead(config)
        # self.sop_classifier = AlbertSOPHead(config)
        # self.num_domain = 5

        self.num_domain_slot = 30
        self.num_gate_label = 3
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
            # print('AUGMENTED IN MY DST_MODELING_ALBERT, 789')
            # with open(f'data/trade_slot_value_dict_augmented.pkl', 'rb') as f:
            #     slot_value_dict = pickle.load(f)

            # print('AUGMENTED ALL SLOT 10 IN MY DST_MODELING_ALBERT, 789')
            # with open(f'data/trade_slot_value_dict_augmented_10_all_slot.pkl', 'rb') as f:
            #     slot_value_dict = pickle.load(f)
        elif config.version =='2':
            with open('data/WOZ22_slot_value_dict', 'rb') as f:
                slot_value_dict = pickle.load(f)

        self.slot_value_dict = slot_value_dict

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
        elif self.config.ds_type =='merged':
            slot_hidden = config.hidden_size * 2
        elif self.config.ds_type == 'none':
            slot_hidden = config.hidden_size

        self.domain_slot_value_classifier_list = \
            nn.ModuleList([nn.Linear(slot_hidden , len(slot_value_list)) for slot_value_list in slot_value_dict.values()])

        self.dropout_slot = nn.Dropout(config.classifier_dropout_prob)

        self.modified_slots_classifier = nn.Linear(config.hidden_size, 1)

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

        self.label_smooth_loss_slot_value = CrossEntropyLoss()

        self.label_smooth_loss_gate = torch.nn.CrossEntropyLoss()
        self.bce_loss_fct = torch.nn.BCEWithLogitsLoss()
        self.init_weights()

    def resize_positional_embeddings(self, new_max_position_length:int):
        old_position_embeddings = self.albert.embeddings.position_embeddings
        old_num_tokens, old_embedding_dim = old_position_embeddings.weight.size()
        new_position_embeddings = nn.Embedding(new_max_position_length, old_embedding_dim)
        self._init_weights(new_max_position_length)
        new_position_embeddings.weight.data[:old_num_tokens, :] = old_position_embeddings.weight.data[:old_num_tokens, :]
        self.albert.embeddings.position_embeddings = new_position_embeddings

    # @add_start_docstrings_to_callable(ALBERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        utterance_type_ids=None,
        gate_labels=None,
        slot_labels=None,
            modified_slots=None,
        **kwargs,
    ):

        # if self.config.ds_type == 'merged':
        #     domain_slot_word_embed = self.domain_slot_word_embedding(self.domain_slot_token_input.to(input_ids.device))
        #     # domain_slot_average_word_embedding = torch.sum(
        #     #     domain_slot_word_embed * self.domain_slot_attn_mask.unsqueeze(2).to(input_ids.device), 1) / torch.sum(self.domain_slot_attn_mask, 1, keepdim=True).to(input_ids.device)
        #
        #     domain_slot_token_length = self.domain_slot_attn_mask.sum(1).to(input_ids.device)
        #     packed_input = torch.nn.utils.rnn.pack_padded_sequence(domain_slot_word_embed, domain_slot_token_length.tolist(), batch_first=True, enforce_sorted=False)
        #     _, domain_slot_gru_output = self.domain_slot_gru(packed_input)
        # else:
        #     domain_slot_gru_output = None

        domain_slot_gru_output = None

        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            utterance_type_ids=utterance_type_ids,
            domain_slot_embeddings=domain_slot_gru_output
        )



        # sequence_output, pooled_output, domain_and_domain_slot_pooled_output = outputs[:3]

        pooled_output, pooled_domain_output, pooled_slot_output, pooled_domain_slot_output, sequence_output = outputs

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

        if modified_slots is not None:
            modified_slots_scores = self.modified_slots_classifier(pooled_domain_slot_output)

            modified_slot_loss = self.bce_loss_fct(modified_slots_scores.squeeze(2), modified_slots.to(torch.float32))

            outputs['modified_slots_logits'] = modified_slots_scores
            outputs['modified_slots_loss'] = modified_slot_loss
            if total_loss is None:
                total_loss = modified_slot_loss
            else:
                total_loss += modified_slot_loss

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
                domain_gate_prediction_scores = [mlp(self.dropout_gate(pooled_domain_slot_output))
                                                 for mlp in self.domain_gate_classifier_list]

            assert (gate_labels <= 2).all(), f'gate label gone wrong in dst_modeling_albert, {gate_labels}'

            domain_gate_prediction_scores = torch.stack(domain_gate_prediction_scores).permute(1, 0, 2)
            gate_loss = loss_fct(domain_gate_prediction_scores.reshape(-1, self.num_gate_label), gate_labels.view(-1))
            # gate_loss = loss_fct(domain_gate_prediction_scores.reshape(-1, self.num_gate_label),
            #                      gate_labels.view(-1, self.num_gate_label).to(domain_gate_prediction_scores.dtype)) # for BCE

            assert not torch.isnan(gate_loss), 'gate loss nan'
            assert not torch.isinf(gate_loss), f'gate loss inf'
            # sentence_order_loss = loss_fct(sop_sept_scores.view(-1, self.config.sept_num_labels), sentence_order_label.view(-1))
            if total_loss is None:
                total_loss = gate_loss
            else:
                total_loss += gate_loss
            outputs['gate_logits'] = domain_gate_prediction_scores
            outputs['gate_loss'] = gate_loss

            assert not torch.isnan(outputs['gate_logits']).any(), 'domain_gate_logit nan'

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

                '''
                sln
                
                # target = slot_labels[:, idx]
                # mask = target != -100
                # if mask.sum() == 0:
                #     continue
                # target = target[mask]
                # pred_score = pred_score[mask]
                # num_class = len(self.slot_value_dict[idx])
                # sigma = self.config.sln
                # if len(target.size()) == 1:
                #     target = pred_score.new_zeros(target.size(0), num_class).scatter_(1, target.view(-1, 1), 1)  # convert label to one-hot
                # 
                # if self.training:
                #     target += sigma * torch.randn(target.size()).to(pred_score.device)
                # 
                # slot_loss_tmp = -torch.mean(torch.sum(torch.nn.functional.log_softmax(pred_score, dim=1) * target, dim=1))
                '''

                if idx == 0:
                    # slot_loss = slot_loss_tmp


                    #normal cross entropy loss
                    slot_loss = loss_fct_tmp(pred_score, slot_labels[:, idx])

                else:

                    #normal cross entropy loss
                    slot_loss += loss_fct_tmp(pred_score, slot_labels[:, idx])

                    # slot_loss += slot_loss_tmp
                # if torch.isnan(slot_loss_tmp):
                #     print('nan')
                # elif torch.isinf(slot_loss_tmp):
                #     print('inf')
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
        # outputs = (total_loss,) + outputs
            # outputs = (total_loss, masked_lm_loss, sentence_order_loss, ) + outputs

        # if gate_labels is not None:
        #     sentence_order_label = sentence_order_label[sentence_order_label != -100]
        #     sentence_order_loss = loss_fct(sop_scores.view(-1, self.config.num_labels), sentence_order_label.view(-1))
        #     # sentence_order_loss = loss_fct(sop_sept_scores.view(-1, self.config.sept_num_labels), sentence_order_label.view(-1))
        #     total_loss += sentence_order_loss
        #     outputs = (total_loss, ) + outputs
        #     # outputs = (total_loss, masked_lm_loss, sentence_order_loss, ) + outputs

        return outputs  # (loss), prediction_scores, sop_scores, (hidden_states), (attentions)



class AlbertMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.LayerNorm = nn.LayerNorm(config.embedding_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size)
        self.activation = ACT2FN[config.hidden_act]

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)

        prediction_scores = hidden_states

        return prediction_scores


class AlbertSOPHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, pooled_output):
        dropout_pooled_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_pooled_output)
        return logits


@add_start_docstrings(
    "Albert Model with a `language modeling` head on top.", ALBERT_START_DOCSTRING,
)
class AlbertForMaskedLM(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.albert = AlbertModel(config)
        self.predictions = AlbertMLMHead(config)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        self._tie_or_clone_weights(self.predictions.decoder, self.albert.embeddings.word_embeddings)

    def get_output_embeddings(self):
        return self.predictions.decoder

    # @add_start_docstrings_to_callable(ALBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="albert-base-v2")
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
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with
            labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.AlbertConfig`) and inputs:
        loss (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
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

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_outputs = outputs[0]

        prediction_scores = self.predictions(sequence_outputs)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs


@add_start_docstrings(
    """Albert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    ALBERT_START_DOCSTRING,
)
class AlbertForSequenceClassification(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    # @add_start_docstrings_to_callable(ALBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="albert-base-v2")
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
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.AlbertConfig`) and inputs:
        loss: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        logits ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
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

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


@add_start_docstrings(
    """Albert Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    ALBERT_START_DOCSTRING,
)
class AlbertForTokenClassification(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    # @add_start_docstrings_to_callable(ALBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="albert-base-v2")
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
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.AlbertConfig`) and inputs:
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

        outputs = self.albert(
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
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


@add_start_docstrings(
    """Albert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """,
    ALBERT_START_DOCSTRING,
)
class AlbertForQuestionAnswering(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        # self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.num_domain_slot = 30
        self.num_gate_label = 3
        # self.domain_classifier = nn.Linear(config.hidden_size, 1)
        # self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
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
        self.label_smooth_loss_slot_value = CrossEntropyLoss()

        self.label_smooth_loss_gate = torch.nn.CrossEntropyLoss()
        self.init_weights()

    def resize_positional_embeddings(self, new_max_position_length:int):
        old_position_embeddings = self.albert.embeddings.position_embeddings
        old_num_tokens, old_embedding_dim = old_position_embeddings.weight.size()
        new_position_embeddings = nn.Embedding(new_max_position_length, old_embedding_dim)
        self._init_weights(new_max_position_length)
        new_position_embeddings.weight.data[:old_num_tokens, :] = old_position_embeddings.weight.data[:old_num_tokens, :]
        self.albert.embeddings.position_embeddings = new_position_embeddings

    # @add_start_docstrings_to_callable(ALBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="albert-base-v2")
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
            gate_labels=None,
            slot_labels=None,
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
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.AlbertConfig`) and inputs:
        loss: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_scores ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        end_scores: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
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

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output, pooled_domain_output, pooled_slot_output, pooled_domain_slot_output, sequence_output = outputs

        outputs = dict()

        # sequence_output = outputs[0]



        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs['start_logits'] = start_logits
        outputs['end_logits'] = end_logits

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

            outputs['']

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


@add_start_docstrings(
    """Albert Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
    ALBERT_START_DOCSTRING,
)
class AlbertForMultipleChoice(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    # @add_start_docstrings_to_callable(ALBERT_INPUTS_DOCSTRING.format("(batch_size, num_choices, sequence_length)"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="albert-base-v2")
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
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices-1]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape `(1,)`, `optional`, returned when :obj:`labels` is provided):
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

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
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
