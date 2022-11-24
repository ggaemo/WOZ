from transformers.data.data_collator import DataCollatorForLanguageModeling, dataclass
import torch
from typing import Dict, List, Tuple
from torch.nn.utils.rnn import pad_sequence

@dataclass
class MyDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):

    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    # def __call__(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    def __call__(self, examples: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tensor_dict = dict()
        key_list = examples[0].keys()
        for key in key_list:
            # if key == "sentence_order_label":
            #     tensor_dict[key] = torch.tensor([x[key] for x in examples])
            # else:
            #     tensor_dict[key] = self._tensorize_batch([x[key] for x in examples])
            tensor_dict[key] = self._tensorize_batch([x[key] for x in examples])

        # inputs_ids = self._tensorize_batch([x['input_ids'] for x in examples])
        # token_type_ids = self._tensorize_batch([x['token_type_ids'] for x in examples])
        # attention_mask = self._tensorize_batch([x['attention_mask'] for x in examples])
        # positional_ids = self._tensorize_batch([x['position_ids'] for x in examples])

        # positional_ids = torch.cat((torch.arange(offset_sys), torch.arange(offset_usr - offset_sys)))
        if self.mlm:
            inputs_ids, labels = self.mask_tokens(tensor_dict['input_ids'])
            tensor_dict['input_ids'] = inputs_ids
            tensor_dict['labels'] = labels
            return tensor_dict
            # return {"input_ids": inputs_ids, "labels": labels}
            # return {"input_ids": inputs_ids, "labels": labels, 'token_type_ids': token_type_ids}
            # return {"input_ids": inputs_ids,
            #         "labels": labels,
            #         'token_type_ids': token_type_ids,
            #         'attention_mask': attention_mask,
            #         'position_ids' : positional_ids}
        else:
            labels = tensor_dict['input_ids'].clone().detach()
            labels[labels == self.tokenizer.pad_token_id] = -100
            tensor_dict['labels'] = labels
            return tensor_dict
            # return {"input_ids": inputs_ids, "labels": labels}
            # return {"input_ids": inputs_ids, "labels": labels, 'token_type_ids': token_type_ids}
            # return {"input_ids": inputs_ids,
            #         "labels": labels,
            #         'token_type_ids': token_type_ids,
            #         'attention_mask':attention_mask,
            #         'position_ids' : positional_ids}

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:

        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # for val in labels.tolist():
        #     x = self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]

        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        vocab_size_wo_sept_token = len(self.tokenizer) - len(self.tokenizer.additional_special_tokens)  # minus 1 for sept token 30000 mins 31 for domain cls
        random_words = torch.randint(vocab_size_wo_sept_token, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels



class MyDataCollatorForDST():


    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    # def __call__(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    def __call__(self, examples: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tensor_dict = dict()
        key_list = examples[0].keys()
        for key in key_list:
            if key == "domain_labels":
                tensor_dict[key] = torch.tensor([x[key] for x in examples])
            else:
                tensor_dict[key] = self._tensorize_batch([x[key] for x in examples])
            # tensor_dict[key] = self._tensorize_batch([x[key] for x in examples])

        # inputs_ids = self._tensorize_batch([x['input_ids'] for x in examples])
        # token_type_ids = self._tensorize_batch([x['token_type_ids'] for x in examples])
        # attention_mask = self._tensorize_batch([x['attention_mask'] for x in examples])
        # positional_ids = self._tensorize_batch([x['position_ids'] for x in examples])

        # positional_ids = torch.cat((torch.arange(offset_sys), torch.arange(offset_usr - offset_sys)))


        return tensor_dict
        # return {"input_ids": inputs_ids, "labels": labels}
        # return {"input_ids": inputs_ids, "labels": labels, 'token_type_ids': token_type_ids}
        # return {"input_ids": inputs_ids,
        #         "labels": labels,
        #         'token_type_ids': token_type_ids,
        #         'attention_mask':attention_mask,
        #         'position_ids' : positional_ids}

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:

        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)