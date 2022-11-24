import pickle

from typing import Dict
import os
import sklearn
import re
import time
import copy
import argparse
import torch

train_arg_parser = argparse.ArgumentParser(description="WOZ")
train_arg_parser.add_argument("--local_rank", default=0, type=int)
train_arg_parser.add_argument("--model-type", default='albert-base-v2', type=str)
train_arg_parser.add_argument("--restore", default='', type=str)
train_arg_parser.add_argument("--restore-finetune", default='', type=str)
train_arg_parser.add_argument("--seed", default=1234, type=int)
train_arg_parser.add_argument("--max-length", default=512, type=int)
train_arg_parser.add_argument("--option", default='base', type=str)
train_arg_parser.add_argument("--ds-type", default='split', type=str)
train_arg_parser.add_argument("--is-debug", action='store_true')
train_arg_parser.add_argument("--batch-size", default=32, type=int)
train_arg_parser.add_argument("--learning-rate", default=5e-5, type=float)
train_arg_parser.add_argument("--warmup-steps", default=1000, type=int)
train_arg_parser.add_argument("--num-train-epochs", default=70, type=int)
train_arg_parser.add_argument("--gradient-accumulation-steps", default=1, type=int)
train_arg_parser.add_argument("--pret_idx", default=1, type=int)
train_arg_parser.add_argument("--no-utt-embed", action='store_true')
train_arg_parser.add_argument("--fp-16",action='store_true')
train_arg_parser.add_argument("--no-d-attn",action='store_true')
train_arg_parser.add_argument("--restore_pretrain", type=str)
train_arg_parser.add_argument("--max-turn", default=5, type=int)

train_arg_parser.add_argument("--data-option", type=str)
train_arg_parser.add_argument("--sln", type=float)
train_arg_parser.add_argument("--version", type=str, default='1')

train_arg_parser.add_argument("--scheduler", type=str, default='linear')

args = train_arg_parser.parse_args()

args.output_dir = os.path.join('result',
                               'dst_'
                               f'{args.model_type}_'                               
                               f'b-{args.batch_size}_'
                               f'g-{args.gradient_accumulation_steps}_'                               
                               f'd-{args.data_option}_'                               
                               f'ds-{args.ds_type}_'
                               f'w-{args.warmup_steps}_'
                               f'lr-{args.learning_rate}_'
                               f'tr-{args.num_train_epochs}_'
                               f'pret_idx-{args.pret_idx}_'
                               f'sc-{args.scheduler}_'
                               f'mt-{args.max_turn}_'
                               f'o-{args.option}_'
                               f'{args.seed}'
                         )
if args.restore:
    args.restore =os.path.join('result', args.restore)
    args.output_dir = args.output_dir+'_restored'
    best_restored = sorted([x for x in os.listdir(args.restore) if re.search('best-model-(\d+)', x)],
                           key = lambda x: int(re.search('best-model-(\d+)', x).group(1)), reverse=True)
    for idx, i in enumerate(best_restored):
        print(idx, i)
    idx = input('select index of restored?')
    args.restore = os.path.join(args.restore, best_restored[idx])
    print('restore from', args.restore)

if args.restore_finetune:
    args.restore_finetune =os.path.join('result', args.restore_finetune)
    args.output_dir = args.output_dir+'_restore_finetune'
    # best_restored = sorted([x for x in os.listdir(args.restore_finetune) if re.search('best-acc-model-(\d+)', x)],
    #                        key = lambda x: int(re.search('best-acc-model-(\d+)', x).group(1)), reverse=True)
    # args.restore_finetune = os.path.join(args.restore_finetune, best_restored[0])
    # print('candidate checkpoint dir', os.listdir(args.restore_finetune))


    best_restored = sorted([x for x in os.listdir(args.restore_finetune) if re.search('interval-(\d+)', x)],
                           key=lambda x: int(re.search('interval-(\d+)', x).group(1)), reverse=True)
    for idx, i in enumerate(best_restored):
        print(idx, i)

    # idx = 0
    idx = input('which to choose?')
    last_global_step = int(re.search('interval-(\d+)', best_restored[int(idx)]).group(1))
    # idx = input('best restored?')

    args.restore_finetune = os.path.join(args.restore_finetune, best_restored[int(idx)])
    print('checkpoint final', os.listdir(args.restore_finetune))
    print('restore finetune from', args.restore_finetune)
    # last_global_step = int(best_restored[0].split('interval-')[1])

if args.is_debug:
    args.output_dir = args.output_dir + '_test'

if args.model_type.startswith('albert'):
    from transformers import AlbertTokenizer
    tokenizer = AlbertTokenizer.from_pretrained(args.model_type)
elif args.model_type.startswith('robert'):
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.model_type)
elif args.model_type.startswith('bert'):
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_type)


# if args.restore_pretrain:
domain_special_tokens = [f'[DS_{i}]' for i in range(30)]
# elif args.ds_type == 'none':
#     domain_special_tokens = []
other_special_tokens = ['[USR]', '[SYS]', '[SEP_DS]']
tokenizer.add_special_tokens({'additional_special_tokens': other_special_tokens + domain_special_tokens})
tokenizer.usr_token = '[USR]'
tokenizer.sys_token = '[SYS]'
tokenizer.sep_ds_token = '[SEP_DS]'
tokenizer.domain_special_tokens = domain_special_tokens
# else:
#     if args.ds_type == 'split':
#         domain_special_tokens = [f'[D_{i}]' for i in range(5)] + [f'[S_{i}]' for i in range(17)]
#     elif args.ds_type == 'merged':
#         domain_special_tokens = [f'[DS_{i}]' for i in range(30)]
#     elif args.ds_type == 'none':
#         domain_special_tokens = []
    # tokenizer.add_special_tokens({'additional_special_tokens': ['[SEPT]'] + domain_special_tokens })

if args.restore_pretrain:
    args.restore_pretrain =os.path.join('pretrain_result', args.restore_pretrain)
    args.output_dir = args.output_dir+'_restored'
    best_restored = sorted([x for x in os.listdir(args.restore_pretrain) if re.search('steps-(\d+)', x)],
                           key = lambda x: int(re.search('steps-(\d+)', x).group(1)), reverse=True)
    for idx, i in enumerate(best_restored):
        print(idx, i)
    # idx = input('select index of restored?')
    args.restore_pretrain = os.path.join(args.restore_pretrain, best_restored[args.pret_idx])
    print('restore from', args.restore_pretrain)

    from custom_dataset_autoregressive import WoZDSTDataset

    train_dataset = WoZDSTDataset(tokenizer, 'train', args.max_length, args.data_option, False, args.max_turn,
                                  args.is_debug, False, False)
    test_dataset = WoZDSTDataset(tokenizer, 'test', args.max_length, args.data_option, False, args.max_turn,
                                 args.is_debug, False, False)
    validation_dataset = WoZDSTDataset(tokenizer, 'dev', args.max_length, args.data_option, False, args.max_turn,
                                       args.is_debug, False, False)
else:
    # from custom_dataset import WoZDSTDataset
    # train_dataset = WoZDSTDataset(tokenizer, 'train', args.max_length, args.data_option, False, 100, args.is_debug)
    # test_dataset = WoZDSTDataset(tokenizer, 'test', args.max_length, args.data_option, False, 100, args.is_debug)
    # validation_dataset = WoZDSTDataset(tokenizer, 'dev', args.max_length, args.data_option, False, 100, args.is_debug)
    from custom_dataset_autoregressive import WoZDSTDataset
    # from custom_dataset_span import WoZDSTDataset
    train_dataset = WoZDSTDataset(tokenizer, 'train', args.max_length, args.data_option, False, args.max_turn,
                                  args.is_debug, False, False)
    test_dataset = WoZDSTDataset(tokenizer, 'test', args.max_length, args.data_option, False, args.max_turn,
                                 args.is_debug, False, False)
    validation_dataset = WoZDSTDataset(tokenizer, 'dev', args.max_length, args.data_option, False, args.max_turn,
                                       args.is_debug, False, False)

# model = AutoModelWithLMHead.from_pretrained(args.model_type)
# from transformers.modeling_albert import AlbertForPreTraining
# model = AlbertForPreTraining.from_pretrained(args.model_type)

time.sleep(2) # waiting for the data/trade_slot_value_dict.pkl to be created

# from my_modeling_albert import AlbertForPreTraining as myAlbertForPreTraining

if args.model_type.startswith('albert'):
    from my_dst_modeling_albert import AlbertForDST as modelforDST
    # from my_dst_modeling_albert import AlbertForQuestionAnswering as modelforDST
elif args.model_type.startswith('robert'):
    from my_dst_modeling_roberta import RobertaForDST as modelforDST
elif args.model_type.startswith('bert'):
    from my_dst_modeling_bert_real import BERTForDST as modelforDST


from transformers import AutoConfig



if args.restore:
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path=args.restore)
    config.version = args.version
    model = modelforDST.from_pretrained(args.restore, config=config)
elif args.restore_finetune:
    print('RESTORING FINETUNE', args.restore_finetune)
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path=args.restore_finetune)
    config.version = args.version
    config.domain_special_tokens = tokenizer.domain_special_tokens
    config.ds_type = args.ds_type
    config.data_option = args.data_option
    config.is_domain_attention = not args.no_d_attn
    config.vocab_size = len(tokenizer)
    model = modelforDST.from_pretrained(args.restore_finetune, config=config)
elif args.restore_pretrain:
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path=args.model_type)
    config.version = args.version
    config.domain_special_tokens = tokenizer.domain_special_tokens
    config.ds_type = args.ds_type
    config.data_option = args.data_option
    config.is_domain_attention = not args.no_d_attn

    # weights = torch.load(os.path.join(args.restore_pretrain, 'pytorch_model.bin'))
    # w_key = list(weights.keys())
    # for k in w_key:
    #     if 'domain_gate' in k:
    #         weights.pop(k)
    # model = modelforDST.from_pretrained(None, config=config, state_dict=weights)
    if 'convbert_dg' in args.restore_pretrain:
        model = modelforDST.from_pretrained(args.restore_pretrain, config=config)
        if config.vocab_size != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
        config.vocab_size = len(tokenizer)
    else:
        config.vocab_size = len(tokenizer)
        model = modelforDST.from_pretrained(args.restore_pretrain, config=config)
else:
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path=args.model_type)
    config.version = args.version
    config.domain_special_tokens = tokenizer.domain_special_tokens
    config.ds_type = args.ds_type
    config.data_option = args.data_option
    config.is_domain_attention = not args.no_d_attn
    model = modelforDST.from_pretrained(args.model_type, config=config)

    if args.max_length > 512:
        max_position_embeddings = args.max_length
        model.resize_positional_embeddings(max_position_embeddings)
        config.max_position_embeddings = max_position_embeddings

    model.resize_token_embeddings(len(tokenizer))


model.config.train_mode = 'dst'
model.config.restore_dir = args.restore
model.config.restore_finetune_dir = args.restore_finetune
model.config.domain_cls_tokens = tokenizer.domain_special_tokens
model.config.sln = args.sln



# from my_utils_multiWOZ_DST import prepare_data_seq
# train, dev, test, test_special, lang, SLOTS_LIST, gating_dict, max_word = \
#     prepare_data_seq(True, args['task'], False, batch_size=int(args['batch']))




from transformers import DataCollatorForLanguageModeling
#
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer, mlm=True, mlm_probability=0.15
# )


from my_data_collator import MyDataCollatorForDST

data_collator = MyDataCollatorForDST(
    tokenizer=tokenizer
)


from transformers import TrainingArguments

from my_dst_trainer import Trainer

# from transformers.trainer_utils import EvalPrediction
from my_trainer_utils import DSTEvalPrediction
#
#     def __call__(self, predictions, label_ids):

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
else:
    device_count = 1
if args.is_debug:
    eval_steps = 1
    save_steps = 100
    logging_steps = 1
    fp_16=False
else:
    eval_steps = 1_000
    save_steps = eval_steps
    # logging_steps = int(500 // ((args.gradient_accumulation_steps * args.batch_size * device_count)))
    logging_steps = 500
    fp_16 = args.fp_16

print('ARGS', args)
if 'eval_only' in args.option:
    eval_batch_size = 1
else:
    eval_batch_size = 2

training_args = TrainingArguments(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=eval_batch_size,
    save_steps=save_steps,
    save_total_limit=None,
    fp16=fp_16,
    fp16_opt_level='O1',
    logging_steps=logging_steps,
    logging_dir=args.output_dir,
    local_rank=args.local_rank,
    evaluate_during_training=True,
    do_eval=True,
    eval_steps=eval_steps,
    logging_first_step=False, #This acutaully means log early ( after 10 steps)
    seed = args.seed,
    learning_rate=args.learning_rate,
    warmup_steps=args.warmup_steps,
)
print('TRAINING ARGS EVAL BATCH SIZE', training_args.eval_batch_size)
# import logging
# logger = logging.getLogger(__name__)
# logger.warning(
#     "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
#     training_args.local_rank,
#     training_args.device,
#     training_args.n_gpu,
#     training_args.local_rank != -1,
# )

def compute_metrics_dst(p: DSTEvalPrediction) -> Dict:
    # sequential = list(range(len(dataset)))
    domain_slot_list = train_dataset.domain_slot_list


    report_dict = dict()

    gate_accuracy = (p.gate_predictions == p.gate_label_ids).sum() / p.gate_label_ids.size

    joint_gate_accuracy = (p.gate_predictions == p.gate_label_ids).all(1).sum() / p.gate_label_ids.shape[0]
    # gate_accuracy = (p.sop_predictions == p.sentence_order_label_ids).sum() / p.sentence_order_label_ids.size
    # gating_dict = {"ptr": 0, "dontcare": 1, "none": 2}
    gating_dict = {0: "ptr", 1: "dontcare",2: "none"}
    # gating_dict = {0:"ptr", 1:"dont"}
    gate_none_idx = 2

    gate_prec = sklearn.metrics.precision_score(p.gate_label_ids.flatten(), p.gate_predictions.flatten(), average=None)
    gate_recall = sklearn.metrics.recall_score(p.gate_label_ids.flatten(), p.gate_predictions.flatten(), average=None)

    # RAW SLOT ACCURACY
    raw_joint_slot_accuracy = (p.slot_predictions == p.slot_label_ids).all(1).sum() / p.slot_label_ids.shape[0]
    raw_slot_accuracy = (p.slot_predictions == p.slot_label_ids).sum() / p.slot_label_ids.size

    slot_pred = copy.deepcopy(p.slot_predictions)
    both_none_mask = (p.slot_predictions == 0) & (p.gate_predictions == gate_none_idx) # 2
    slot_pred[both_none_mask] = 0

    both_none_joint_slot_accuracy = (slot_pred == p.slot_label_ids).all(1).sum() / p.slot_label_ids.shape[0]
    both_none_slot_accuracy = (slot_pred == p.slot_label_ids).sum() / p.slot_label_ids.size
    # gating none
    pred_none_mask = p.gate_predictions == gate_none_idx # 1
    p.slot_predictions[pred_none_mask] = 0

    none_joint_slot_accuracy = (p.slot_predictions == p.slot_label_ids).all(1).sum() / p.slot_label_ids.shape[0]
    none_slot_accuracy = (p.slot_predictions == p.slot_label_ids).sum() / p.slot_label_ids.size

    # gating dontcare
    # with open(f'data/trade_slot_value_dict.pkl', 'rb') as f
    #     slot_value_dict = pickle.load(f)
    # pred_dontcare_mask = p.gate_predictions == 1
    # for i in range(30):
    #     slot_domain_dontcare_mask = pred_dontcare_mask[:, i]
    #     # p.slot_predictions[slot_domain_dontcare_mask] = slot_value_dict[i]['dontcare']
    #     p.slot_predictions[slot_domain_dontcare_mask, i] = slot_value_dict[i]['dontcare']
    #
    # dontcare_none_joint_slot_accuracy = (p.slot_predictions == p.slot_label_ids).all(1).sum() / p.slot_label_ids.shape[0]
    # dontcare_none_slot_accuracy = (p.slot_predictions == p.slot_label_ids).sum() / p.slot_label_ids.size

    # joint_slot_accuracy = (p.slot_predictions == p.slot_label_ids).all(1).sum() / p.slot_label_ids.shape[0]
    # slot_accuracy = (p.slot_predictions == p.slot_label_ids).sum() / p.slot_label_ids.size



    true_not_none_mask = p.slot_label_ids != 0 # index for value 'none' is 0 # custom_dataset.py 466
    ptr_slot_correct = 0
    ptr_joint_slot_correct = 0

    for idx, mask_tmp in enumerate(true_not_none_mask):
        match = p.slot_predictions[idx][mask_tmp] == p.slot_label_ids[idx][mask_tmp]
        ptr_slot_correct += match.sum()
        ptr_joint_slot_correct += int(match.all())

    ptr_slot_accuracy = ptr_slot_correct / p.slot_label_ids[true_not_none_mask].size
    ptr_joint_slot_accuracy = ptr_joint_slot_correct / p.slot_label_ids.shape[0]

    report_dict.update({"joint_gate_accuracy": joint_gate_accuracy,
                   'gate_accuracy': gate_accuracy,
                   'raw_slot_accuracy': raw_slot_accuracy,
                   'raw_joint_slot_accuracy': raw_joint_slot_accuracy,
                        'none_slot_accuracy': none_slot_accuracy,
                        'none_joint_slot_accuracy': none_joint_slot_accuracy,
                        'both_none_slot_accuracy': both_none_slot_accuracy,
                        'both_none_joint_slot_accuracy': both_none_joint_slot_accuracy,
                        'joint_slot_accuracy': raw_joint_slot_accuracy,
                   'ptr_slot_accuracy' : ptr_slot_accuracy,
                   'ptr_joint_slot_accuracy': ptr_joint_slot_accuracy})

    for idx, name in enumerate(domain_slot_list):
        slot_accuracy = (p.slot_predictions[:, idx] == p.slot_label_ids[:, idx]).sum() / p.slot_label_ids[:, idx].size
        report_dict.update({f'{name}_individual_accuracy': slot_accuracy})

    for idx, name in enumerate(domain_slot_list):
        not_none_value_mask = p.slot_label_ids[:, idx] != 0
        not_none_value_slot_accuracy = (p.slot_predictions[not_none_value_mask, idx] == p.slot_label_ids[not_none_value_mask, idx]).sum() / p.slot_label_ids[not_none_value_mask, idx].size
        report_dict.update({f'{name}_not_none_individual_accuracy': not_none_value_slot_accuracy})

    try:
        for idx, val in enumerate(gate_prec):
            report_dict[f'prec_{gating_dict[idx]}'] = val
        for idx, val in enumerate(gate_recall):
            report_dict[f'rec_{gating_dict[idx]}'] = val
    except KeyError:
        print('@@@@@@@@@@@@@@@@@@@@', 'ERROR' , '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        import numpy as np
        xx = np.unique(p.gate_label_ids)
        print('gate label', xx)
        print('#####################')
        print(gate_prec.shape)

    return report_dict

trainer = Trainer(
    model=model,
    args=training_args,
    model_config = vars(args),
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    validation_dataset=validation_dataset,
    prediction_loss_only=False,
    compute_metrics=compute_metrics_dst,
    metrics_name_list=[]
)

if args.restore_finetune:
    model_path = args.restore_finetune
else:
    model_path = None

if 'eval_only' in args.option:
    trainer.evaluate_autoregressive(test_dataset, 'evaluation', last_global_step)
else:
    trainer.train(model_path=model_path)