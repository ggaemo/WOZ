from torch.utils.data.dataset import Dataset
import json
import os
import pickle
import tqdm
import torch
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Dict
from itertools import permutations
import numpy as np
import collections
import re

class WoZDSTDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def prepare_data(self, tokenizer, data_type):
        with open(f'trade-dst/data/processed_{data_type}_fixed', 'rb') as f:
            data = pickle.load(f)
        pair_data, data_max_len, slot_type_list = data
        del data
        self.lines = list()
        slot_value_set = collections.defaultdict(set)

        for row in pair_data:
            for slot_idx, slot_value in enumerate(row['generate_y']):
                slot_value_set[slot_idx].add(slot_value)
        if data_type == 'train':
            slot_value_dict = collections.defaultdict(dict)
            for key, val in slot_value_set.items():
                val.remove('none')
                if 'dontcare' in val:
                    val.remove('dontcare')
                slot_value_dict[key].update({'none': self.none_idx})
                slot_value_dict[key].update({'dontcare': self.none_idx + 1})
                slot_value_dict[key].update({slot_val:idx for idx, slot_val in enumerate(sorted(val), start=self.none_idx + 2)})


                slot_value_dict[key].update({tokenizer.unk_token: -100})
                # slot_value_dict[key].update({'dontcare': -101})
        else:
            with open(f'data/trade_slot_value_dict.pkl', 'rb') as f:
                slot_value_dict = pickle.load(f)

        self.gate_label = list()
        self.slot_label = list()
        # self.turn_domain = list()


        for cnt, row in enumerate(pair_data):
            if 'police' in row['domains'] or 'hospital' in row['domains']:
                continue

            dialog_history = re.sub(' -s', 's',row['dialog_history'])  # include -s , restaurant -s place -s hotel -s
            dialog_history = re.sub(' it s ', " it's ", dialog_history)  # include -s , restaurant -s place -s hotel -s
            dialog_history = re.sub(' there s ', " there's ", dialog_history)  # include -s , restaurant -s place -s hotel -s
            dialog_history = re.sub(' -ly', "ly",
                                    dialog_history)  # moderate -ly
            turns = dialog_history.split(';')[1:-1] # last one is equivalent to the last ; so it is a blank state.
            # dialog_context = turns[-max_length:] #ALL TURNS
            dialog_context = turns
            dialog_context = [x.strip() for x in dialog_context]

            # dialog_context = add_special_token(dialog_context)
            self.lines.append(dialog_context)
            # for slot_idx, slot_value in enumerate(row['generate_y']):
            #     slot_value_dict[slot_idx].add(slot_value)
            gate_array = np.array(row['gating_label'])
            gate_array = np.where(gate_array == 1, 0, gate_array)
            gate_array = np.where(gate_array == 2, 1, gate_array)
            row['gating_label'] = gate_array.tolist()

            self.gate_label.append(row['gating_label'])
            slot_label_tmp = list()
            for idx, slot_val_tmp in enumerate(row['generate_y']):
                if slot_val_tmp not in slot_value_dict[idx]:
                    slot_label_tmp.append(slot_value_dict[idx][tokenizer.unk_token])
                else:
                    slot_label_tmp.append(slot_value_dict[idx][slot_val_tmp])
                # except KeyError:
                #     print('whats')

            self.slot_label.append(slot_label_tmp)
            if row['turn_domain'] not in self.domains_to_idx:
                print('what')
                print(dialog_context)
                print('@@', row['turn_domain'])
                # if dialog_context[0] in ['what is the address to the hospital in north?', 'what is the address to the hospital in cambridge?']:
                #     print('BREAK')
            # self.turn_domain.append(self.domains_to_idx[row['turn_domain']])

            # data_dict = {'dialog': self.lines,
            #              'gate_label': self.gate_label,
            #              'slot_label': self.slot_label,
            #              'turn_domain': self.turn_domain
            #              }
            #
            # with open(f'data/trade_dialog_{max_length}_{data_option}_{data_type}.pkl', 'wb') as f:
            #     pickle.dump(data_dict, f)
            if data_type == 'train':
                with open(f'data/trade_slot_value_dict.pkl', 'wb') as f:
                    pickle.dump(slot_value_dict, f)
            # del data_dict

    def __init__(self, tokenizer, data_type, max_length, data_option, is_pretrain, pretrain_gate=False, no_utt_embed=False):

        self.tokenizer = tokenizer
        self.data_type = data_type
        self.sep_token_id = tokenizer.sep_token_id
        self.sept_token_id = tokenizer.additional_special_tokens_ids[0]
        self.domain_slot_cls_tokens = tokenizer.additional_special_tokens[1:]
        self.data_option = data_option
        self.none_idx = 0
        self.is_pretrain = is_pretrain
        self.pretrain_gate = pretrain_gate
        self.no_utt_embed = no_utt_embed
        # if data_option not in ['sept_sept_sep', 'sept_sept_2sep']:
        #     raise KeyError('wrong data option')

        self.sys_start = 0
        self.usr_start = 1
        self.max_length = max_length
        # self.domains_to_idx = {"attraction":0, "restaurant":1, "taxi":2, "train":3, "hotel":4, "hospital":5, "bus":6, "police":7}
        self.domains_to_idx = {"attraction": 0, "restaurant": 1, "taxi": 2, "train": 3, "hotel": 4}

        self.domain_slot_list = ['hotel-price range',
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

        def add_special_token(seq_list):
            # sequence = [tokenizer.cls_token]
            # sequence.extend(tokenizer.additional_special_tokens[1:])
            # sequence.append(tokenizer.sep_token)
            sequence = list()
            for seq in seq_list:
                sequence.append(seq)
                sequence.append(tokenizer.additional_special_tokens[0])
            sequence.append(tokenizer.sep_token)
            return ' '.join(sequence)

        if is_pretrain:
            pretrain_or_finetune = 'pretrain'
        else:
            pretrain_or_finetune = 'finetune'

        if os.path.exists(f'data/{data_option}_{pretrain_or_finetune}_{data_type}.pkl'):
            with open(f'data/{data_option}_{pretrain_or_finetune}_{data_type}.pkl', 'rb') as f:
                data_dict = pickle.load(f)
                self.batch_encoding = data_dict['batch_encoding']
                self.special_tokenize = data_dict['special_tokenize']
                if is_pretrain:
                    self.sentence_order_list = data_dict['sentence_order_label']
                else:
                    self.slot_label = data_dict['slot_label']
                # self.turn_domain = data_dict['turn_domain']
                self.gate_label = data_dict['gate_label']

        else:
            self.prepare_data(tokenizer, data_type)

            if is_pretrain:
                print('original data length', len(self.lines))
                permuted_lines = list()
                sentence_order_list = list()
                turn_domain = list()
                gate_label = list()
                permute_frac = 0.5
                for ii, line in enumerate(self.lines):
                    dialog_len = len(line)
                    # if dialog_len < max_length:
                    #     perms_list = list(permutations(np.arange(dialog_len // 2)))[1:]  # first permutation is the original permutation
                    # else:
                    #     perms_list = list(permutations(np.arange(max_length // 2)))[1:]  # first permutation is the original permutation
                    #
                    # if int(len(perms_list) * permute_frac) < 1:
                    #     permute_data_len = int(len(perms_list))
                    # else:
                    #     permute_data_len = int(len(perms_list) * permute_frac)
                    #
                    # rand_choice = sorted(
                    #     np.random.choice(range(len(perms_list)), permute_data_len, replace=False))
                    # perms_list = [perms_list[x] for x in rand_choice]

                    sentence_order = tuple(np.arange(dialog_len)) #+ (-100,) * (max_length - dialog_len)
                    # sentence_order = tuple(np.arange(dialog_len))

                    permuted_lines.append(line)
                    sentence_order_list.append(sentence_order)
                    turn_domain.append(self.turn_domain[ii])
                    gate_label.append(self.gate_label[ii])
                    # for perm in perms_list:
                    #     sentence_order = list()
                    #     for i in perm:
                    #         sentence_order.append(i * 2)
                    #         sentence_order.append(i * 2 + 1)
                    #     permuted_input = [line[x] for x in sentence_order]  # first permutation is the original permutation
                    #     permuted_lines.append(permuted_input)
                    #
                    #     sentence_order = tuple(sentence_order) + (-100,) * (max_length - dialog_len)
                    #     sentence_order_list.append(sentence_order)
                    #     turn_domain.append(self.turn_domain[ii])
                    #     gate_label.append(self.gate_label[ii])

                self.lines = permuted_lines
                self.sentence_order_list = sentence_order_list
                # self.turn_domain = turn_domain
                self.gate_label = gate_label

            if 'none' in data_type:
                domain_slot_speical_tokens_str = tokenizer.cls_token
            else:
                domain_slot_speical_tokens_str = ' '.join(self.domain_slot_cls_tokens + [tokenizer.cls_token])
            # if self.max_length < 1024:
                # len_special_tokens = 1 + len(tokenizer.additional_special_tokens_ids[1:]) + 1  # 1, speical tokens, 1
                # trunc_lines = list()
                # for x in self.lines:
                #     xx = add_special_token(x)
                #     if len(xx) > self.max_length:
                #         print('before', xx)
                #         xx = xx[:len_special_tokens] + xx[-(self.max_length-len_special_tokens):]
                #         print('after', xx)
                #
                #     assert len(xx) <= self.max_length
                #     trunc_lines.append(xx)
                # self.lines = trunc_lines
            # else:
            #     self.lines = [add_special_token(x) for x in self.lines]
            self.lines = [add_special_token(x) for x in self.lines]

            self.batch_encoding = tokenizer(text=self.lines,
                                           truncation=False,
                                           add_special_tokens=False,
                                           return_token_type_ids=None,
                                           return_attention_mask=False)

            self.special_tokenize = tokenizer(text=domain_slot_speical_tokens_str,
                                              truncation=False,
                                              add_special_tokens=False,
                                              return_token_type_ids=None,
                                              return_attention_mask=False
                                              )

            # self.batch_encoding = tokenizer(text=[domain_slot_speical_tokens_str] * len(self.lines),
            #                                 text_pair = self.lines,
            #                                 truncation=False,
            #                                 max_length=1024,
            #                                 add_special_tokens=False,
            #                                 return_token_type_ids=True,
            #                                 return_attention_mask=True
            #                                 )

            del self.lines
            data_dict = {}
            data_dict['batch_encoding'] = self.batch_encoding
            data_dict['special_tokenize'] = self.special_tokenize
            if is_pretrain:
                data_dict['sentence_order_label'] = self.sentence_order_list
            else:
                data_dict['slot_label'] = self.slot_label
            # data_dict['turn_domain'] = self.turn_domain
            data_dict['gate_label'] = self.gate_label
            with open(f'data/{data_option}_{pretrain_or_finetune}_{data_type}.pkl', 'wb') as f:
                pickle.dump(data_dict, f)

        self.domain_slot_cls_special_token_length = len(tokenizer.additional_special_tokens[1:] + [tokenizer.cls_token])


        with open(f'data/trade_slot_value_dict.pkl', 'rb') as f:
            self.slot_value_dict = pickle.load(f)

        # slot_value_text = [f'{tokenizer.cls_token} ' + f' {tokenizer.sep_token} '.join(self.slot_value_dict[i])
        #                         for i in self.slot_value_dict]

        # self.slot_value_text = self.tokenizer(slot_value_text, padding=True, return_attention_mask=True)
        # self.usr_or_sys_start_list = usr_or_sys_start_list
        # self.sequence_order_list = sequence_order_list
        # data_dict = dict()
        # data_dict['batch_encoding'] = self.batch_encoding
        # data_dict['usr_or_sys_start_list'] = self.usr_or_sys_start_list
        # data_dict['sequence_order_list'] = self.sequence_order_list
        # with open(f'data/dialog_{max_length}_{data_option}_{data_type}.pkl', 'wb') as f:
        #     pickle.dump(data_dict, f)
        # del data_dict


        print('total number of data', len(self.batch_encoding['input_ids']))

        self.lengths_list = [len(x) for x in self.batch_encoding['input_ids']]
        # if self.max_length == 512:
        #     if self.data_type == 'train':
        #         self.reset_dataset(0, self.max_length)
        if max_length <= 512:
            self.max_line_length = 512 - len(self.special_tokenize['input_ids'])
        else:
            self.max_line_length = 1024


    def reset_dataset(self, start_length, end_inclusive_length):
        selection_idx_list = list()
        for idx, l in enumerate(self.lengths_list):
            if end_inclusive_length >= l > start_length:
                selection_idx_list.append(idx)
        print('train', 'TAKE ONLY 512')
        self.batch_encoding['input_ids'] = [self.batch_encoding['input_ids'][x] for x in selection_idx_list]
        self.batch_encoding['attention_mask'] = [self.batch_encoding['attention_mask'][x] for x in selection_idx_list]
        # self.batch_encoding['token_type_ids'] = [self.batch_encoding['token_type_ids'][x] for x in selection_idx_list]
        self.gate_label = [self.gate_label[x] for x in selection_idx_list]
        self.slot_label = [self.slot_label[x] for x in selection_idx_list]


    def __len__(self):
        return len(self.batch_encoding['input_ids'])

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sample = dict()
        lines = self.batch_encoding['input_ids'][i]

        if len(lines) > self.max_line_length:
            sept_idx_list = np.where(np.array(lines) == self.sept_token_id)[0]
            for sept_idx in sept_idx_list[1::2]:
                if len(lines[sept_idx+1:]) <= self.max_line_length:
                    truncation_idx = sept_idx + 1
                    lines_truncated = lines[truncation_idx:]
                    break
        else:
            lines_truncated = lines

        sample['input_ids'] = torch.tensor(self.special_tokenize['input_ids'] + lines_truncated , dtype=torch.long)

        sample['attention_mask'] = torch.ones_like(sample['input_ids'], dtype=torch.long)

        # sample['token_type_ids'] = token_type_ids
        # sample['token_type_ids'] = torch.tensor(self.batch_encoding['token_type_ids'][i], dtype=torch.long)
        if not self.no_utt_embed:
            token_type_ids, utterance_type_ids = self.get_token_utt_type_ids(sample['input_ids'], i)
            sample['utterance_type_ids'] = utterance_type_ids

        # sample['position_ids'] = torch.cat((torch.zeros(len(self.domain_slot_cls_tokens), dtype=torch.long), torch.arange(len(sample['input_ids']) - len(self.domain_slot_cls_tokens),dtype=torch.long)))

        if self.is_pretrain:
            if self.pretrain_gate:
            # sample['domain_labels'] = torch.tensor([self.turn_domain[i]], dtype=torch.long)
                sample['gate_labels'] = torch.tensor(self.gate_label[i], dtype=torch.long)
            # sample['sentence_order_label'] = torch.tensor(self.sentence_order_list[i], dtype=torch.long)
            # pass
        else:
            # sample['domain_labels'] = torch.tensor([self.turn_domain[i]], dtype=torch.long)
            sample['gate_labels'] = torch.tensor(self.gate_label[i], dtype=torch.long)
            sample['slot_labels'] = torch.tensor(self.slot_label[i], dtype=torch.long)
            # sample['slot_values'] = torch.tensor(self.slot_value_text['input_ids'], dtype=torch.long)
            # sample['slot_values_attn_mas k'] = torch.tensor(self.slot_value_text['attention_mask'], dtype=torch.long)

            # if self.data_type == 'train':
            #     slot_drop_frac = 0.1
            #     indices_replaced = torch.bernoulli(torch.full(sample['slot_labels'].shape, slot_drop_frac)).bool()
            #     sample['slot_labels'][indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)


        # print(idx, {k:v.shape for k, v in sample.items()})
        assert (sample['gate_labels'] <= 2).all(), f"{sample['gate_labels']}, {self.gate_label[i]}"
        return sample

    def get_token_utt_type_ids(self, input_ids, i):
        sep_idx_list = torch.where(input_ids == self.sep_token_id)[0]
        sept_idx_list = torch.where(input_ids == self.sept_token_id)[0]

        # offset_seg_1, offset_seg_2 = sep_idx_list
        #
        #
        # token_type_ids = torch.cat(
        #             (torch.zeros(offset_seg_1 + 1, dtype=torch.long),  # include segment token(sep or sept)
        #              torch.ones(offset_seg_2 - offset_seg_1 ,dtype=torch.long)))  # include sept and sep

        token_type_ids = torch.zeros_like(input_ids)

        # elif self.data_option == 'sept_sept_2sep':
        #     if len(sep_idx_list) == 2:
        #         offset_seg_1, offset_seg_2 = sep_idx_list
        #         token_type_ids = torch.cat(
        #             (torch.zeros(offset_seg_1 + 1, dtype=torch.long),  # include segment token(sep or sept)
        #              torch.ones(offset_seg_2 - offset_seg_1 ,dtype=torch.long)))  # include sept and sep
        #         # token_type_ids = torch.cat((torch.zeros(offset_seg_1 + 1, dtype=torch.long), # include segment token(sep or sept)
        #         #                                       torch.ones(offset_seg_2 - offset_seg_1 + len(self.domain_cls_tokens), dtype=torch.long))) # include sept and sep
        #     elif len(sep_idx_list) == 3:
        #         offset_seg_1, offset_seg_2, offset_seg_3 = sep_idx_list
        #         token_type_ids = torch.cat(
        #             (torch.zeros(offset_seg_1 + 1, dtype=torch.long),  # include segment token(sep or sept)
        #              torch.ones(offset_seg_3 - offset_seg_1, dtype=torch.long)))  # include sept and sep

        # token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)

        if 'nosept' in self.data_option:
            utterance_type_ids_list = None
        else:
            utterance_type_ids_list = list()

            zero_or_one = [torch.zeros, torch.ones]

            # prev_sept_idx = offset_seg_1

            prev_sept_idx = self.domain_slot_cls_special_token_length - 1

            for idx, sept_idx in enumerate(sept_idx_list):
                if sept_idx + 1 == sep_idx_list[-1]: # if [SEP] exists in the next token, include it
                    sept_idx = sept_idx + 1
                curr_utt_length = sept_idx - prev_sept_idx
                if self.is_pretrain:
                    utt_type = zero_or_one[self.sentence_order_list[i][idx] % 2]
                else:
                    utt_type = zero_or_one[idx % 2]
                utterance_type_ids_list.append(utt_type(curr_utt_length, dtype=torch.long))
                prev_sept_idx = sept_idx


            # utterance_type_ids_list.append(torch.zeros(len(self.domain_cls_tokens), dtype=torch.long))
            utterance_type_ids_list = torch.cat(utterance_type_ids_list)

        return token_type_ids, utterance_type_ids_list


if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModelWithLMHead

    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
    # x = tokenizer.add_special_tokens({'additional_special_tokens': ['[SEPT]']})
    domain_special_tokens = [f'[D_{i}]' for i in range(5)] + [f'[S_{i}]' for i in range(17)]
    tokenizer.add_special_tokens({'additional_special_tokens': ['[SEPT]'] + domain_special_tokens})
    # print(tokenizer.encode('What is [SEPT]'))

    print(tokenizer.additional_special_tokens)
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")


    # for dialog_type in ['pair_org_seq_consec']:
    dataset = WoZDSTDataset(tokenizer, 'test', 512, 'split_new_512', False, False, False)
    for i in range(len(dataset)):
        print(dataset[i])
        print(tokenizer.decode(dataset[i]['input_ids']))

    # for i in [113,  76, 129,  43, 275, 567, 634, 350]:
    #     print(dataset[i]['attention_mask'].shape)
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    # for i in [143, 128, 151,  71, 171,  38, 770, 134]:
    #     print(dataset[i]['attention_mask'].shape)
    #
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    # for i in [213, 280, 666,  54,  80,  70,  77,  67]:
    #     print(dataset[i]['attention_mask'].shape)

    #
    # print('@@')
    # print(dataset[3])
    # print(tokenizer.decode(dataset[3]['input_ids']))

    # x = dataset[0]


    # print(dataset[0])
    # for i in tqdm.tqdm(range(len(dataset))):
    #     if len(dataset[i]['input_ids']) == 512:
    #         print(dataset[i]['input_ids'])
    #         print(dataset[i]['utterance_type_ids'].shape)
    #         print('@@', dataset[i]['utterance_type_ids'], '##')
    #         print('WHAT')

    # print('0' * 100)
    # print(dataset[0])
    # print(tokenizer.decode(dataset[0]['input_ids']))
    # print('LENGTH', len(dataset))
    # for i in range(len(dataset)):
    #     if any(dataset[i]['gate_labels'] > 2):
    #         print(i, dataset[i])


    # print('0' * 100)
    # print(dataset[10013])
    #
    # print(tokenizer.decode(dataset[0]['input_ids']))
    #
    # print(dataset[1])
    # print(tokenizer.decode(dataset[1]['input_ids']))



