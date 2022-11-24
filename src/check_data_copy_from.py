import json
import pickle
import numpy as np
import copy
import os
import re

with open('MultiWOZ_2.2/schema.json') as f:
    schema = json.load(f)

schema_slots = dict()
for tmp_schema in schema:
    for slot in tmp_schema['slots']:
        schema_slots[slot['name']] = slot

processed_data = list()

def levenshtein(s1, s2, debug=False):
    if len(s1) < len(s2):
        return levenshtein(s2, s1, debug)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))

        if debug:
            print(current_row[1:])

        previous_row = current_row

    return previous_row[-1]

def process_data(data, cnt):
    for idx in range(len(data)):
        num_turns = len(data[idx]['turns'])
        dialog_history = list()
        import collections
        all_frames_with_tid = collections.defaultdict(list)
        for turn_num in range(0, num_turns, 2):
            turn = data[idx]['turns']
            turn_processed_data = dict()
            if turn_num == 0:
                sys = None
                user = turn[turn_num]
                dialog_history.append(user['utterance'])
            else:
                sys = turn[turn_num - 1]
                user = turn[turn_num]
                dialog_history.append(sys['utterance'])
                dialog_history.append(user['utterance'])
            turn_processed_data['dialog_context'] = copy.deepcopy(dialog_history)

            user_frames = user['frames']
            if turn_num > 0:
                sys_frames = sys['frames']
            else:
                sys_frames = []


            for sys_frame_tmp in sys_frames:
                if len(sys_frame_tmp['slots']) > 0:
                    for frame in sys_frame_tmp['slots']:
                        all_frames_with_tid[frame['slot']].append((frame, user['turn_id']))
            for user_frame_tmp in user_frames:
                curr_state = user_frame_tmp['state']['slot_values']
                if len(user_frame_tmp['slots']) > 0:
                    for frame in user_frame_tmp['slots']:
                        if 'copy_from' not in frame:
                            all_frames_with_tid[frame['slot']].append((frame, user['turn_id']))

            categorical_state = dict()
            non_categorical_state = dict()

            for user_frame in user_frames:
                # categorical
                curr_state = user_frame['state']['slot_values']

                curr_non_cat_slots_dict = collections.defaultdict(list)
                for x in user_frame['slots']:
                    curr_non_cat_slots_dict[x['slot']].append(x)

                for curr_idx, (key, val) in enumerate(curr_state.items()):
                    accounted_for = False
                    is_categorical = schema_slots[key]['is_categorical']
                    if is_categorical:
                        categorical_state[key] = val
                        accounted_for = True
                    else:
                        for frame in curr_non_cat_slots_dict[key]:
                            if 'copy_from' in frame.keys():
                                copy_slot_name = frame['copy_from']
                                for frame_other, tid in all_frames_with_tid[copy_slot_name]:
                                    if frame_other['value'].lower() in frame['value']:
                                        non_categorical_state[key] = {'turn_num': tid,
                                                                      'start': frame_other['start'],
                                                                      'end': frame_other['exclusive_end'],
                                                                      'span_slot_value': frame_other['value'],
                                                                      'state_candidate_val_list': val,
                                                                      'frame_candidate_val': frame['value'],
                                                                      'copy_from': copy_slot_name
                                                                      }
                                        all_frames_with_tid[key].append((frame_other, tid))
                                        accounted_for = True
                                        break


                            elif not accounted_for and 'start' in frame:
                                non_categorical_state[key] = {'turn_num': turn_num,
                                                              'start': frame['start'],
                                                              'end': frame['exclusive_end'],
                                                              'span_slot_value': frame['value'],
                                                              'state_candidate_val_list': val
                                                              }
                                accounted_for = True


                        if not accounted_for:

                            if val[0] == 'dontcare':
                                non_categorical_state[key] = {'turn_num': -100,
                                                              'start': -100,
                                                              'end': -100,
                                                              'span_slot_value': -100,
                                                              'state_candidate_val_list': val,
                                                              'dontcare':True
                                                              }
                                accounted_for = True
                            else:
                                for frame_other, tid in all_frames_with_tid[key]:

                                    if len(val) > 1 and levenshtein(val[0], val[1]) < 6:
                                        print(f'^^^^ {idx}-{turn_num}-{key}-{val} ^^ multiple values simliar but not the same^^^', val)
                                    # for frame in frames:
                                        # if frame['slot'] == key:
                                    if frame_other['value'].lower() in val:
                                        non_categorical_state[key] = {'turn_num': tid,
                                                                      'start': frame_other['start'],
                                                                      'end': frame_other['exclusive_end'],
                                                                      'span_slot_value': frame_other['value'],
                                                                      'state_candidate_val_list': val
                                                                      }
                                        accounted_for =True
                                        break
                                    elif levenshtein(frame_other['value'].lower(), val[0]) < 4:
                                        print('#####similar considered match',f'{idx}-{turn_num}-{key}-{val}', frame_other['value'].lower(), '#', val[0], '#####')
                                        non_categorical_state[key] = {'turn_num': tid,
                                                                      'start': frame_other['start'],
                                                                      'end': frame_other['exclusive_end'],
                                                                      'span_slot_value': frame_other['value'],
                                                                      'state_candidate_val_list': val
                                                                      }
                                        accounted_for = True


                    if not accounted_for:
                        cnt +=1
                        print(f'Non match $$${idx}-{turn_num}-{key}-{val}$$$$$$$$$$$$$$')



            turn_processed_data['categorical_slot'] = copy.deepcopy(categorical_state)
            turn_processed_data['non_categorical_slot'] = copy.deepcopy(non_categorical_state)
            #         turn_processed_data['state'] = all_state
            #     print(turn_num, turn_processed_data)
            processed_data.append(turn_processed_data)


    return cnt

cnt = 0
for data_file in sorted(os.listdir('MultiWOZ_2.2/train')):
    with open(f'MultiWOZ_2.2/train/{data_file}') as f:
        data = json.load(f)
    cnt = process_data(data, cnt)
    print('@@@@@@@@@@@@@@', data_file, cnt, '@@@@@@@@@@@@@@')