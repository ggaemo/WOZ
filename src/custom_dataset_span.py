import copy
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
import random

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
        self.turn_domain = list()
        self.prev_slot_label = list()
        self.curr_slot_label = list()
        self.ids = list()
        self.turn_ids = list()

        prev_row = None

        cached_file = os.path.join('/home/jinwon/dialoglue/trippy', 'cached_train_features_albert')
        features = torch.load(cached_file)

        ds_token_length = len(self.domain_slot_special_tokens)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        f_start_pos = [f.start_pos + ds_token_length for f in features]
        f_end_pos = [f.end_pos + ds_token_length for f in features]
        f_inform_slot_ids = [f.inform_slot for f in features]
        f_refer_ids = [f.refer_id for f in features]
        f_diag_state = [f.diag_state for f in features]
        f_class_label_ids = [f.class_label_id for f in features]
        all_start_positions = {}
        all_end_positions = {}
        all_inform_slot_ids = {}
        all_refer_ids = {}
        all_diag_state = {}
        all_class_label_ids = {}
        for s in model.slot_list:
            all_start_positions[s] = torch.tensor([f[s] for f in f_start_pos], dtype=torch.long)
            all_end_positions[s] = torch.tensor([f[s] for f in f_end_pos], dtype=torch.long)
            all_inform_slot_ids[s] = torch.tensor([f[s] for f in f_inform_slot_ids], dtype=torch.long)
            all_refer_ids[s] = torch.tensor([f[s] for f in f_refer_ids], dtype=torch.long)
            all_diag_state[s] = torch.tensor([f[s] for f in f_diag_state], dtype=torch.long)
            all_class_label_ids[s] = torch.tensor([f[s] for f in f_class_label_ids], dtype=torch.long)
        dataset = TensorListDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_start_positions, all_end_positions,
                                    all_inform_slot_ids,
                                    all_refer_ids,
                                    all_diag_state,
                                    all_class_label_ids, all_example_index)


        for row_idx, row in enumerate(pair_data):
            if 'police' in row['domains'] or 'hospital' in row['domains']:
                # print(row_idx, row['ID'])
                continue
            if row['turn_domain'] not in self.domains_to_idx:
                print(row['domains'])
                print(row['turn_domain'])
                print(row_idx, row['turn_uttr'])
                print('@'*100)
            dialog_history = re.sub(' -s', 's',row['dialog_history'])  # include -s , restaurant -s place -s hotel -s
            dialog_history = re.sub(' it s ', " it's ", dialog_history)  # include -s , restaurant -s place -s hotel -s
            dialog_history = re.sub(' there s ', " there's ", dialog_history)  # include -s , restaurant -s place -s hotel -s
            dialog_history = re.sub(' -ly', "ly", dialog_history)  # moderate -ly
            turns = dialog_history.split(';')[1:-1] # last one is equivalent to the last ; so it is a blank state.
            turns = turns[::-1]
            print("REVERSED TURNS! custom dataset autoregressive.py")
            dialog_context = turns[:self.max_turns] #ALL TURNSz
            # dialog_context = turns
            dialog_context = [x.strip() for x in dialog_context]

            # dialog_context = add_special_token(dialog_context)
            self.lines.append(dialog_context)
            # for slot_idx, slot_value in enumerate(row['generate_y']):
            #     slot_value_dict[slot_idx].add(slot_value)
            gate_array = np.array(row['gating_label'])
            # gate_array = np.where(gate_array == 1, 0, gate_array)
            # gate_array = np.where(gate_array == 2, 1, gate_array)
            row['gating_label'] = gate_array.tolist()
            self.ids.append(row['ID'])
            self.turn_ids.append(row['turn_id'])
            self.gate_label.append(row['gating_label'])
            self.curr_slot_label.append(row['generate_y'])
            slot_label_tmp = list()
            for idx, slot_val_tmp in enumerate(row['generate_y']):
                if slot_val_tmp not in slot_value_dict[idx]:
                    slot_label_tmp.append(slot_value_dict[idx][tokenizer.unk_token])
                else:
                    slot_label_tmp.append(slot_value_dict[idx][slot_val_tmp])
                # except KeyError:
                #     print('whats')

            if row['turn_id'] == 0:
                self.prev_slot_label.append(['none'] * 30)
                prev_row = row
            else:
                # if row_idx == 5315:
                #     print('WHAT')
                assert prev_row['ID'] == row['ID']
                if not (row['turn_id'] - prev_row['turn_id']) == 1:
                    print('BREAK')
                assert (row['turn_id'] - prev_row['turn_id']) == 1,(row['turn_id'], prev_row['turn_id'])
                self.prev_slot_label.append(prev_row['generate_y'])
                prev_row = row

            self.slot_label.append(slot_label_tmp)
            if row['turn_domain'] in self.domains_to_idx:
                self.turn_domain.append(self.domains_to_idx[row['turn_domain']])
            else:
                self.turn_domain.append(self.domains_to_idx['null'])



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

    def __init__(self, tokenizer, data_type, max_length, data_option, is_pretrain, max_turns, is_debug, delta_ds, order_shuffle):
        self.tokenizer = tokenizer
        self.data_type = data_type
        self.sep_token_id = tokenizer.sep_token_id
        if 'wo_none' in data_option:
            self.wo_none = True
        elif 'w_none' in data_option:
            self.wo_none = False

        if delta_ds:
            self.delta_ds = True
        else:
            self.delta_ds = False

        self.order_shuffle = order_shuffle
        # elif 'no_ds' in data_option:
        #     self.delta_ds = False

        # self.sept_token_id = tokenizer.additional_special_tokens_ids[0]
        # self.sept_value_token_id = tokenizer.additional_special_tokens_ids[1]
        self.domain_slot_special_tokens = tokenizer.domain_special_tokens
        self.data_option = data_option
        self.none_idx = 0
        self.max_turns = max_turns
        self.is_pretrain = is_pretrain
        self.is_debug = is_debug


        self.sys_start = 0
        self.usr_start = 1
        self.max_length = max_length

        # self.dialog_state_change_ratio = dialog_state_change_ratio
        # self.domains_to_idx = {"attraction":0, "restaurant":1, "taxi":2, "train":3, "hotel":4, "hospital":5, "bus":6, "police":7}
        self.domains_to_idx = {"attraction": 0, "restaurant": 1, "taxi": 2, "train": 3, "hotel": 4, "null": 5}

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

        def get_usr_sys_dialog(dialog_history_list):

            dialog_history_string_list = list()
            for seq_list in dialog_history_list:
                sequence_1 = list()
                sequence_1.extend(self.domain_slot_special_tokens)
                sequence_1.append(tokenizer.cls_token)
                for idx, seq in enumerate(seq_list):
                    if idx % 2 == 0:
                        sequence_1.append(tokenizer.usr_token)
                    else:
                        sequence_1.append(tokenizer.sys_token)
                    sequence_1.append(seq)
                sequence_1.append(tokenizer.sep_token)
                dialog_history_string_list.append(' '.join(sequence_1))
            return dialog_history_string_list



        if is_pretrain:
            pretrain_or_finetune = 'pretrain'
        else:
            pretrain_or_finetune = 'finetune'

        if os.path.exists(f'data/{data_option}_{max_turns}_{data_type}.pkl'):
            with open(f'data/{data_option}_{max_turns}_{data_type}.pkl', 'rb') as f:
                data_dict = pickle.load(f)
                self.dialog_history = data_dict['dialog_history']
                self.prev_slot_label = data_dict['prev_slot_label']
                # self.batch_encoding = data_dict['batch_encoding']
                # if is_pretrain:
                #     self.sentence_order_list = data_dict['sentence_order_label']
                # else:
                self.slot_label = data_dict['slot_label']
                self.curr_slot_label = data_dict['curr_slot_label']
                self.turn_domain = data_dict['turn_domain']
                self.gate_label = data_dict['gate_label']
                self.modified_slots = data_dict['modified_slots']
                self.prev_ds_history = self.get_prev_ds_string(self.prev_slot_label, False)
                self.ids = data_dict['ids']
                self.turn_ids = data_dict['turn_ids']
        else:
            self.prepare_data(tokenizer, data_type)

            # domain_slot_speical_tokens_str = ' '.join(self.domain_slot_cls_tokens + [tokenizer.cls_token])
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
            # self.lines = [add_special_token(x) for x in self.linesself.prev_slot_label]

            self.modified_slots = list()
            for prev_slot_label_elem, slot_label_elem in zip(self.prev_slot_label, self.curr_slot_label):
                tmp = np.zeros(len(slot_label_elem))
                indices = [idx for idx, (x, y) in enumerate(zip(prev_slot_label_elem, slot_label_elem)) if x != y]
                tmp[indices] = 1
                self.modified_slots.append(tmp)

            self.dialog_history = get_usr_sys_dialog(self.lines)
            self.prev_ds_history = self.get_prev_ds_string(self.prev_slot_label, False)

            del self.lines
            data_dict = {}
            # data_dict['batch_encoding'] = self.batch_encoding
            data_dict['dialog_history'] = self.dialog_history
            data_dict['prev_slot_label'] = self.prev_slot_label
            data_dict['curr_slot_label'] = self.curr_slot_label
            # if is_pretrain:
            #     data_dict['sentence_order_label'] = self.sentence_order_list
            data_dict['slot_label'] = self.slot_label
            data_dict['turn_domain'] = self.turn_domain
            data_dict['gate_label'] = self.gate_label
            data_dict['modified_slots'] = self.modified_slots
            data_dict['ids'] = self.ids
            data_dict['turn_ids'] = self.turn_ids

            with open(f'data/{data_option}_{max_turns}_{data_type}.pkl', 'wb') as f:
                pickle.dump(data_dict, f)



        # self.domain_slot_cls_special_token_length = len(tokenizer.additional_special_tokens[1:] + [tokenizer.cls_token])

        if is_debug:
            self.dialog_history = self.dialog_history[:1000]
            self.prev_ds_history = self.prev_ds_history[:1000]

        self.batch_encoding = tokenizer(text=self.dialog_history,
                                        text_pair=self.prev_ds_history,
                                        truncation=True,
                                        max_length=max_length,
                                        add_special_tokens=False,
                                        return_token_type_ids=True,
                                        return_attention_mask=True
                                        )

        self.in_order_label = [0] * len(self.dialog_history)

        with open(f'data/trade_slot_value_dict.pkl', 'rb') as f:
            self.slot_value_dict = pickle.load(f)

        self.idx_to_slot_value_dict = collections.defaultdict(dict)
        for key, val in self.slot_value_dict.items():
            self.idx_to_slot_value_dict[key] = {v: k for k, v in val.items()}

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


        print('total number of data', len(self.dialog_history))

        self.lengths_list = [len(x) for x in self.dialog_history]
        # if self.max_length == 512:
        #     if self.data_type == 'train':
        #         self.reset_dataset(0, self.max_length)


    def get_prev_ds_string(self, prev_ds_list, shuffle):
        prev_ds_string_list = list()
        for prev_ds in prev_ds_list:
            sequence_2 = list()
            if self.wo_none:
                not_none_indices = [idx for idx, val in enumerate(prev_ds) if val != 'none']
            else:
                not_none_indices = list(range(len(prev_ds)))
            if shuffle:
                random.shuffle(not_none_indices)
            for not_none_idx in not_none_indices:
                try:
                    dsp = self.domain_slot_list[not_none_idx]
                except IndexError:
                    print('what?')
                prev_ds_val = prev_ds[not_none_idx]
                sequence_2.append(dsp)
                sequence_2.append(':')
                sequence_2.append(prev_ds_val)
                sequence_2.append(self.tokenizer.sep_ds_token)
            sequence_2.append(self.tokenizer.sep_token)
            prev_ds_string_list.append(' '.join(sequence_2))
        return prev_ds_string_list

    def reset_dataset(self, start_length, end_inclusive_length):
        selection_idx_list = list()
        for idx, l in enumerate(self.lengths_list):
            if end_inclusive_length >= l > start_length:
                selection_idx_list.append(idx)
        # print('train', 'TAKE ONLY 512')
        self.batch_encoding['input_ids'] = [self.batch_encoding['input_ids'][x] for x in selection_idx_list]
        self.batch_encoding['attention_mask'] = [self.batch_encoding['attention_mask'][x] for x in selection_idx_list]
        self.batch_encoding['token_type_ids'] = [self.batch_encoding['token_type_ids'][x] for x in selection_idx_list]
        self.gate_label = [self.gate_label[x] for x in selection_idx_list]
        self.slot_label = [self.slot_label[x] for x in selection_idx_list]

    def change_dialog_state(self):

        # new_dialog_history = copy.deepcopy(self.dialog_history)
        # new_slot_label = copy.deepcopy(self.slot_label)
        slot_label = self.slot_label
        # curr_slot_label = self.curr_slot_label
        dialog_history = self.dialog_history
        slot_value_dict = self.slot_value_dict
        # dialog_state_change_ratio = self.dialog_state_change_ratio
        idx_to_slot_value_dict = self.idx_to_slot_value_dict
        prev_slot_label = self.prev_slot_label

        new_dialog_history = copy.deepcopy(dialog_history)
        # new_slot_label = copy.deepcopy(slot_label)
        new_prev_slot_label = copy.deepcopy(prev_slot_label)

        def get_slot_to_change(idx):
            applicable_slot_found = False
            for chosen_slot_idx in np.random.permutation(np.where(np.array(prev_slot_label[idx]) != 'none')[0]):
                substitue_slot_value = prev_slot_label[idx][chosen_slot_idx]
                if len(substitue_slot_value) > 4 and substitue_slot_value not in ['dontcare', '<unk>']:
                    applicable_slot_found = True
                    break
            if applicable_slot_found:
                return substitue_slot_value, chosen_slot_idx
            else:
                return None, None

        for idx in range(len(dialog_history)):
            number_of_none_slots = len(np.where(np.array(prev_slot_label[idx]) != 'none')[0])
            #     print(idx, number_of_none_slots)
            cnt = 0
            # max_cnt = max(1, int(number_of_none_slots * dialog_state_change_ratio))
            try_cnt = 0
            recorded = False
            if number_of_none_slots > 0:
                new_dialog = dialog_history[idx]
                while try_cnt <= 5:
                    if cnt == 1:
                        break
                    substitue_slot_value, chosen_slot_idx = get_slot_to_change(idx)
                    if not substitue_slot_value:
                        break

                    vocab_tmp = len(slot_value_dict[chosen_slot_idx]) - 3  # none, dontcare, <unk>
                    vocab_range_wihtout_current = list(np.arange(vocab_tmp) + 2)
                    current_slot_value_idx = slot_value_dict[chosen_slot_idx][substitue_slot_value]
                    #             print(vocab_range_wihtout_current, current_slot_value_idx)
                    vocab_range_wihtout_current.remove(current_slot_value_idx)
                    chosen_slot_value_idx = np.random.choice(vocab_range_wihtout_current)
                    candidate_slot_value = idx_to_slot_value_dict[chosen_slot_idx][chosen_slot_value_idx]

                    try_cnt += 1
                    if re.search(' ' + substitue_slot_value + ' ', new_dialog):
                        new_dialog = re.sub(substitue_slot_value, candidate_slot_value, new_dialog)
                        # print("NEW Dialog", new_dialog)
                        new_prev_slot_label[idx][chosen_slot_idx] = candidate_slot_value
                        # new_slot_label[idx][chosen_slot_idx] = chosen_slot_value_idx # 나중에 label 바꿀때
                        idx_to_change = np.where(np.array(new_prev_slot_label[idx]) == substitue_slot_value)[0]
                        for idx_tmp_2 in idx_to_change:
                            #                         if candidate_slot_value not in slot_value_dict[idx_tmp_2]:
                            #                             이건 나중에 그냥 한꺼번에 destination에 넣는 걸로 해결하기
                            #                         new_slot_label[idx][idx_tmp_2] = slot_value_dict[idx_tmp_2][candidate_slot_value]
                            new_prev_slot_label[idx][idx_tmp_2] = candidate_slot_value
                            # print('##', self.domain_slot_list[chosen_slot_idx], 'CURR', substitue_slot_value, 'NEW',
                            #       candidate_slot_value)
                            # print('@@', new_dialog[230:])

                        recorded = True
                        #                                     print('Wew label', idx_to_slot_value_dict[chosen_slot_idx][new_slot_label[idx][chosen_slot_value_idx]])
                        #                     if domain_slot_list[chosen_slot_idx] not in recorded:
                        #                         recorded.add(domain_slot_list[chosen_slot_idx])

                        cnt += 1
                if recorded:
                    #                 print('RECORDED', idx)
                    new_dialog_history[idx] = new_dialog

        new_prev_ds_history = self.get_prev_ds_string(new_prev_slot_label, True)

        self.batch_encoding = self.tokenizer(text=new_dialog_history,
                                        text_pair=new_prev_ds_history,
                                        truncation=True,
                                        max_length=self.max_length,
                                        add_special_tokens=False,
                                        return_token_type_ids=True,
                                        return_attention_mask=True
                                        )

    def reset_batch_encoding_wrt_prev_ds(self):

        def get_usr_sys_dialog_individual(seq_list):

            sequence_1 = list()
            sequence_1.extend(self.domain_slot_special_tokens)
            sequence_1.append(self.tokenizer.cls_token)
            for idx, seq in enumerate(seq_list):
                if idx % 2 == 0:
                    sequence_1.append(self.tokenizer.usr_token)
                else:
                    sequence_1.append(self.tokenizer.sys_token)
                sequence_1.append(seq)
            # sequence_1.append(self.tokenizer.sep_token) # already added
            return ' '.join(sequence_1)

        prev_slot_label = copy.deepcopy(self.prev_slot_label)
        prev_ds_history = self.get_prev_ds_string(prev_slot_label, True)

        if self.is_pretrain and self.in_order_label:
            swapped_history = list()
            self.in_order_label = list()

            for idx in range(len(self.dialog_history)):
                current_dialog = self.dialog_history[idx]
                cnt = current_dialog.count('[USR]')
                if cnt < 3:
                    self.in_order_label.append(0)
                    swapped_history.append(current_dialog)
                else:
                    if np.random.uniform(0, 1) > 0.5:
                        self.in_order_label.append(0)
                        swapped_history.append(current_dialog)
                    else:
                        current_dialog_list = re.split(' \[SYS\] | \[USR\] ', current_dialog)[1:]
                        self.in_order_label.append(1)
                        while(True):
                            chosen = np.random.choice(list(range(0, idx-30)) + list(range(idx+30, len(self.dialog_history))))
                            to_be_swapped = self.dialog_history[chosen]
                            to_be_swapped_dialog_list = re.split(' \[SYS\] | \[USR\] ', to_be_swapped)[1:]
                            if len(to_be_swapped_dialog_list) >= 2:
                                break
                        shorter_length = min(len(to_be_swapped_dialog_list), len(current_dialog_list))
                        # low, high = sorted(np.random.choice(range(2, shorter_length - 2), size=2, replace=False))
                        start = np.random.randint(shorter_length - 1)
                        end = start + max(1, np.random.randint(shorter_length // 2))
                        for idx_tmp in range(start, end):
                            current_dialog_list[idx_tmp] = to_be_swapped_dialog_list[idx_tmp]
                        swapped_history.append(get_usr_sys_dialog_individual(current_dialog_list))
        else:
            swapped_history = self.dialog_history
        # swapped_history = get_usr_sys_dialog(swapped_history)
        # print(idx, swapped_history)


                    # np.random.choice(range(len(to_be_swapped))


                    # to_be_swapped_turn = np.random.randint(0, len(to_be_swapped_dialog_list))
                    #
                    # current_dialog_list = re.split(' \[SYS\] | \[USR\] ', current_dialog)[1:]
                    # to_be_swapped_turn = np.random.randint(0, len(to_be_swapped_dialog_list))
                    #
                    # get_usr_sys_dialog()
                    #
                    # ii = re.split(' \[SYS\] | \[USR\] ', to_be_swapped)[1:]
                    # current_dialog

        self.batch_encoding = self.tokenizer(text=swapped_history,
                                        text_pair=prev_ds_history,
                                        truncation=True,
                                        max_length=self.max_length,
                                        add_special_tokens=False,
                                        return_token_type_ids=True,
                                        return_attention_mask=True
                                        )
    def __len__(self):
        return len(self.dialog_history)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sample = dict()
        sample['input_ids'] = torch.tensor(self.batch_encoding['input_ids'][i], dtype=torch.long)
        sample['attention_mask'] = torch.tensor(self.batch_encoding['attention_mask'][i], dtype=torch.long)
        if self.delta_ds and self.is_pretrain:
            sample['modified_slots'] = torch.tensor(self.modified_slots[i], dtype=torch.long)

        sample['token_type_ids'] = torch.tensor(self.batch_encoding['token_type_ids'][i], dtype=torch.long)
        if self.in_order_label:
            sample['in_order_labels'] = torch.tensor([self.in_order_label[i]], dtype=torch.long)

        if not self.is_pretrain:
            sample['gate_labels'] = torch.tensor(self.gate_label[i], dtype=torch.long)
            sample['slot_labels'] = torch.tensor(self.slot_label[i], dtype=torch.long)

        return sample

    def get_token_utt_type_ids(self, input_ids, i):
        sep_idx_list = torch.where(input_ids == self.sep_token_id)[0]
        sept_idx_list = torch.where(input_ids == self.sept_token_id)[0]

        # offset_seg_1, offset_seg_2 = sep_idx_list
        #
        #
        token_type_ids = torch.cat(
                    (torch.zeros(sep_idx_list[0] + 1, dtype=torch.long),  # include segment token(sep or sept)
                     torch.ones(sep_idx_list[1] - sep_idx_list[0] ,dtype=torch.long)))  # include sept and sep

        # token_type_ids = torch.zeros_like(input_ids)
        # token_type_ids

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
                if sept_idx + 1 == sep_idx_list[0]: # if [SEP] exists in the next token, include it
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
    dataset = WoZDSTDataset(tokenizer, 'train', 1024, 'ds_split_test', True, False, False)

    print(dataset[0])
    print(tokenizer.decode(dataset[0]['input_ids']))

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



