import logging
import collections
import math
import os
import pickle
import re
import shutil
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from tqdm.auto import tqdm, trange

from transformers.data.data_collator import DataCollator, default_data_collator
from transformers.file_utils import is_apex_available, is_torch_tpu_available
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from my_trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BEST_PREFIX_CHECKPOINT_DIR,
    BEST_ACC_PREFIX_CHECKPOINT_DIR,
    DSTEvalPrediction,
    DSTPredictionOutput,
    TrainOutput,
    is_wandb_available,
    set_seed,
)

# from transformers.trainer_utils import (
#     PREFIX_CHECKPOINT_DIR,
#     EvalPrediction,
#     PredictionOutput,
#     TrainOutput,
#     is_wandb_available,
#     set_seed,
# )
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if is_apex_available():
    from apex import amp

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

try:
    from torch.utils.tensorboard import SummaryWriter

    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter

        _has_tensorboard = True
    except ImportError:
        _has_tensorboard = False


def is_tensorboard_available():
    return _has_tensorboard


if is_wandb_available():
    import wandb


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.

    Args:
        local_rank (:obj:`int`): The rank of the local process.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.

    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank * self.num_samples: (self.rank + 1) * self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


import torch.distributed as dist
class MyDistributedSampler(Sampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        quartile = np.quantile(self.dataset.lengths_list, q=np.arange(0, 1.0, 0.25))[1:-1] #exclude min
        self.quartile = list(quartile) + [max(self.dataset.lengths_list)]


    def __iter__(self):

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        # indices = torch.randperm(len(self.dataset), generator=g).tolist()
        indices = list(range(len(self.dataset)))
        extra_indices = list(np.random.randint(0, len(indices), size = self.total_size - len(indices)))
        indices += extra_indices
        assert len(indices) == self.total_size
        indices = np.array(indices)
        # print('extra', extra_indices)
        # print('before', len(self.dataset.lengths))
        # print('length extra', [self.dataset.lengths[x] for x in extra_indices])
        lengths = self.dataset.lengths_list + [self.dataset.lengths_list[x] for x in extra_indices]
        lengths = np.array(lengths)
        # print('after', len(lengths))
        shuffled_indices = list()
        prev_tmp_len = min(self.dataset.lengths_list) - 1
        print('PREV11', prev_tmp_len, 'PREV22')
        for tmp_len in self.quartile:
            tmp_mask = (lengths > prev_tmp_len) & (lengths <= tmp_len)
            tmp_indices = indices[tmp_mask]
            permed_indices = torch.randperm(len(tmp_indices))
            print('##', tmp_len, permed_indices, '@@')
            print('***', tmp_indices[permed_indices], '%%%')
            shuffled_indices.extend(tmp_indices[permed_indices].tolist())
            prev_tmp_len = tmp_len
        assert len(shuffled_indices) == len(indices)

        # subsample
        shuffled_indices = shuffled_indices[self.rank:self.total_size:self.num_replicas]
        assert len(shuffled_indices) == self.num_samples

        return iter(shuffled_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class MySequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.

    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        self.quartile = np.quantile(self.dataset.lengths, q=np.arange(0, 1.1, 0.3))[1:] # first elem is the smallest length, 1.1 to include max value

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        extra_indices = list(np.random.randint(0, len(indices), size = self.total_size - len(indices)))
        indices += extra_indices
        assert len(indices) == self.total_size

        shuffled_indices = list()
        prev_tmp_len = 0
        for tmp_len in self.quartile:
            tmp_mask = self.dataset.lengths <= tmp_len & self.dataset.lengths > prev_tmp_len
            shuffled_indices.extend(np.random.shuffle(indices[tmp_mask]))
            prev_tmp_len = tmp_len
        assert len(shuffled_indices) == len(indices)

        # subsample
        indices = indices[self.rank * self.num_samples: (self.rank + 1) * self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

def get_tpu_sampler(dataset: Dataset):
    if xm.xrt_world_size() <= 1:
        return RandomSampler(dataset)
    return DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())


class Trainer:
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch,
    optimized for 🤗 Transformers.

    Args:
        model (:class:`~transformers.PreTrainedModel`):
            The model to train, evaluate or use for predictions.
        args (:class:`~transformers.TrainingArguments`):
            The arguments to tweak training.
        data_collator (:obj:`DataCollator`, `optional`, defaults to :func:`~transformers.default_data_collator`):
            The function to use to from a batch from a list of elements of :obj:`train_dataset` or
            :obj:`eval_dataset`.
        train_dataset (:obj:`Dataset`, `optional`):
            The dataset to use for training.
        eval_dataset (:obj:`Dataset`, `optional`):
            The dataset to use for evaluation.
        compute_metrics (:obj:`Callable[[EvalPrediction], Dict]`, `optional`):
            The function that will be used to compute metrics at evaluation. Must take a
            :class:`~transformers.EvalPrediction` and return a dictionary string to metric values.
        prediction_loss_only (:obj:`bool`, `optional`, defaults to `False`):
            When performing evaluation and predictions, only returns the loss.
        tb_writer (:obj:`SummaryWriter`, `optional`):
            Object to write to TensorBoard.
        optimizers (:obj:`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR`, `optional`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of
            :class:`~transformers.AdamW` on your model and a scheduler given by
            :func:`~transformers.get_linear_schedule_with_warmup` controlled by :obj:`args`.
    """

    model: PreTrainedModel
    args: TrainingArguments
    model_config: Dict
    metrics_name_list: List
    data_collator: DataCollator
    train_dataset: Optional[Dataset]
    validation_dataset: Optional[Dataset]
    eval_dataset: Optional[Dataset]
    compute_metrics: Optional[Callable[[DSTEvalPrediction], Dict]] = None
    prediction_loss_only: bool
    tb_writer: Optional["SummaryWriter"] = None
    optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None
    global_step: Optional[int] = None
    epoch: Optional[float] = None

    def __init__(
            self,
            model: PreTrainedModel,
            args: TrainingArguments,
            model_config: Dict,
            metrics_name_list: List,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            validation_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            compute_metrics: Optional[Callable[[DSTEvalPrediction], Dict]] = None,
            prediction_loss_only=False,
            tb_writer: Optional["SummaryWriter"] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None,
    ):
        # logger = logging.getLogger(__name__)
        # logging.basicConfig(level=logging.INFO)
        # Create handlers
        # c_handler = logging.StreamHandler()
        # f_handler = logging.FileHandler(os.path.join(args.logging_dir, 'output.log'))
        # f_handler = logging.FileHandler('output.log')
        # c_handler.setLevel(logging.INFO)
        # f_handler.setLevel(logging.INFO)
        #
        # # Create formatters and add it to handlers
        # c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

        # c_handler.setFormatter(c_format)

        #
        # # Add handlers to the logger
        # logger.addHandler(c_handler)

        self.model = model.to(args.device)
        self.args = args
        self.metrics_name_list = metrics_name_list
        self.model_config = model_config
        self.data_collator = data_collator if data_collator is not None else default_data_collator
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.prediction_loss_only = prediction_loss_only
        self.optimizers = optimizers
        self.cuda = torch.cuda.is_available()


        assert self.args.save_steps % self.args.eval_steps == 0
        if tb_writer is not None:
            self.tb_writer = tb_writer
        elif is_tensorboard_available() and self.is_world_master():
            self.tb_writer = SummaryWriter(log_dir=self.args.logging_dir)
        if is_wandb_available():
            self._setup_wandb()
        else:
            logger.info(
                "You are instantiating a Trainer but W&B is not installed. To use wandb logging, "
                "run `pip install wandb; wandb login` see https://docs.wandb.com/huggingface."
            )
        set_seed(self.args.seed)
        # Create output directory if needed
        if self.is_world_master():
            os.makedirs(self.args.output_dir, exist_ok=True)

            for file in os.listdir('src'):
                if file.endswith('.py'):
                    shutil.copy('src/' + file, os.path.join(args.output_dir, file))
            with open(os.path.join(args.output_dir, 'training_args.txt'), 'w') as f:
                for key, val in self.model_config.items():
                    f.writelines(f'{key}: {val}\n')

            f_handler = logging.FileHandler(os.path.join(self.args.output_dir, 'output.log'))
            f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            f_handler.setFormatter(f_format)
            logger.addHandler(f_handler)

        if is_torch_tpu_available():
            # Set an xla_device flag on the model's config.
            # We'll find a more elegant and not need to do this in the future.
            self.model.config.xla_device = True
        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            self.data_collator = self.data_collator.collate_batch
            warnings.warn(
                (
                        "The `data_collator` should now be a simple callable (function, class with `__call__`), classes "
                        + "with a `collate_batch` are deprecated and won't be supported in a future version."
                ),
                FutureWarning,
            )

        ## add logger output

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if is_torch_tpu_available():
            train_sampler = get_tpu_sampler(self.train_dataset)
        else:
            train_sampler = (
                RandomSampler(self.train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )

            # train_sampler = (
            #     SequentialSampler(self.train_dataset)
            #     if self.args.local_rank == -1
            #     else MyDistributedSampler(self.train_dataset)
            # )
            #
            # import copy
            # dataset_split_1 = copy.deepcopy(self.train_dataset)
            #
            # reset_dataset(self, start_length, end_inclusive_length)
            #
            # indices = list(range(len(self.train_dataset)))
            # for tmp_len in self.quartile:
            #     tmp_mask = (lengths > prev_tmp_len) & (lengths <= tmp_len)
            #     tmp_indices = indices[tmp_mask]
            #
            # quartile = np.quantile(self.dataset.lengths_list, q=np.arange(0, 1.0, 0.2))[1:-1]  # exclude min
            # self.quartile = list(quartile) + [max(self.dataset.lengths_list)]
            #
            # g = torch.Generator()
            # g.manual_seed(self.seed + self.epoch)
            # indices = np.array(indices)
            # lengths = self.dataset.lengths_list
            # lengths = np.array(lengths)
            #
            # shuffled_indices = list()
            # prev_tmp_len = min(self.dataset.lengths_list) - 1
            # print('PREV11', prev_tmp_len, 'PREV22')
            # for tmp_len in self.quartile:
            #     tmp_mask = (lengths > prev_tmp_len) & (lengths <= tmp_len)
            #     tmp_indices = indices[tmp_mask]
            #     permed_indices = torch.randperm(len(tmp_indices))
            #     print('##', tmp_len, permed_indices, '@@')
            #     print('***', tmp_indices[permed_indices], '%%%')
            #     shuffled_indices.extend(tmp_indices[permed_indices].tolist())
            #     prev_tmp_len = tmp_len
            # assert len(shuffled_indices) == len(indices)




        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            # drop_last=self.args.dataloader_drop_last,
            num_workers=2,
            pin_memory=True
        )

        return data_loader


    def get_divided_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if is_torch_tpu_available():
            train_sampler = get_tpu_sampler(self.train_dataset)
        else:
            train_sampler = (
                RandomSampler(self.train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )

            # train_sampler = (
            #     SequentialSampler(self.train_dataset)
            #     if self.args.local_rank == -1
            #     else MyDistributedSampler(self.train_dataset)
            # )


        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            # drop_last=self.args.dataloader_drop_last,
            num_workers=2,
            pin_memory=True
        )

        return data_loader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                If provided, will override `self.eval_dataset`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if is_torch_tpu_available():
            sampler = SequentialDistributedSampler(
                eval_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
            )
        elif self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(eval_dataset)
            # sampler = DistributedSampler(eval_dataset, shuffle=False)
            print('EVAL DATA LOADER Not Sequential DS, ordinary DS')
        else:
            sampler = SequentialSampler(eval_dataset)
            # sampler = DistributedSampler(eval_dataset, shuffle=False)

        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            # drop_last=self.args.dataloader_drop_last,
            num_workers=0,
            pin_memory=True
        )

        return data_loader

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test :class:`~torch.utils.data.DataLoader`.

        Args:
            test_dataset (obj:`Dataset`): The test dataset to use.
        """
        # We use the same batch_size as for eval.
        if is_torch_tpu_available():
            sampler = SequentialDistributedSampler(
                test_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
            )
        elif self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(test_dataset)
        else:
            sampler = SequentialSampler(test_dataset)

        data_loader = DataLoader(
            test_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

        return data_loader

    def get_optimizers(
            self, num_training_steps: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or override this method in a subclass.
        """
        if self.optimizers is not None:
            return self.optimizers
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)

        if self.model_config['scheduler'] == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = num_training_steps//3, gamma=0.1)
        else:
            scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def _setup_wandb(self):
        """
        Setup the optional Weights & Biases (`wandb`) integration.

        One can override this method to customize the setup if needed.  Find more information at https://docs.wandb.com/huggingface
        You can also override the following environment variables:

        Environment:
            WANDB_WATCH:
                (Optional, ["gradients", "all", "false"]) "gradients" by default, set to "false" to disable gradient logging
                or "all" to log gradients and parameters
            WANDB_PROJECT:
                (Optional): str - "huggingface" by default, set this to a custom string to store results in a different project
            WANDB_DISABLED:
                (Optional): boolean - defaults to false, set to "true" to disable wandb entirely
        """
        if self.is_world_master():
            logger.info(
                'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
            )
            wandb.init(project=os.getenv("WANDB_PROJECT", "huggingface"), config=vars(self.args))
            # keep track of model topology and gradients, unsupported on TPU
            if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
                wandb.watch(
                    self.model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, self.args.logging_steps)
                )

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a :class:`~torch.utils.data.DataLoader` by accessing its Dataset.
        """
        return len(dataloader.dataset)

    def train(self, model_path: Optional[str] = None):
        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
        """
        train_dataloader = self.get_train_dataloader()
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                    self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
                model_path is not None
                # and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
                # and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            checkpoint_dict = torch.load(os.path.join(model_path, 'check_point_dict.pth'),
                                         map_location=self.args.device)

            # optimizer.load_state_dict(checkpoint_dict['optimizer'])
            #
            # scheduler.load_state_dict(checkpoint_dict['scheduler'])


        model = self.model
        if self.args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)


        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            from apex.parallel import DistributedDataParallel as DDP
            model = DDP(model, delay_allreduce=True)
            # model = torch.nn.parallel.DistributedDataParallel(
            #     model,
            #     device_ids=[self.args.local_rank],
            #     output_device=self.args.local_rank,
            #     find_unused_parameters=True,
            # )
        elif self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            # self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        if is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                    self.args.train_batch_size
                    * self.args.gradient_accumulation_steps
                    * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        if self.is_world_master():
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", self.num_examples(train_dataloader))
            logger.info("  Num Epochs = %d", num_train_epochs)
            logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
            logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                        total_train_batch_size)
            logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
            logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                        len(train_dataloader) // self.args.gradient_accumulation_steps
                )
                if self.is_world_master():
                    logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                    logger.info("  Continuing training from epoch %d", epochs_trained)
                    logger.info("  Continuing training from global step %d", self.global_step)
                    logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = 0.0
        logging_loss = 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master()
        )
        best_eval_loss = 1e4
        best_acc = -1e4
        oom=False
        for epoch in train_iterator:
            log_first = True

            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if 'shuffle' in self.args.output_dir and (epoch + 1) % 10 ==0:
                print('CHANGE PREV DS STATE EVERY 10 EPOCHs', epoch)
                self.train_dataset.reset_batch_encoding_wrt_prev_ds()
                train_dataloader = self.get_train_dataloader()

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_master())
            else:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=not self.is_local_master())

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            for step, inputs in enumerate(epoch_iterator, start=0):
                # logger.info(f'STEPS {step}')
                # Skip past any already trained steps if resuming training
                # if steps_trained_in_current_epoch > 0:
                #     steps_trained_in_current_epoch -= 1
                #     continue
                # tr_loss += self._training_step(model, inputs, optimizer)
                try:
                    tr_loss += float(self._training_step(model, inputs, optimizer))
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning(
                            f"attempting to recover from OOM in forward/backward pass {step}"
                        )
                        oom = True
                    else:
                        raise e
                if oom:
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad
                    optimizer.zero_grad()
                    if self.cuda:
                        torch.cuda.empty_cache()
                    continue




                if self.cuda and step == 0:
                    torch.cuda.empty_cache()

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        len(epoch_iterator) <= self.args.gradient_accumulation_steps
                        and (step + 1) == len(epoch_iterator)
                ):
                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if is_torch_tpu_available():
                        xm.optimizer_step(optimizer)
                    else:
                        optimizer.step()

                    scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0):
                        logs: Dict[str, float] = {}
                        logs["loss"] = (tr_loss - logging_loss) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else scheduler.get_lr()[0]
                        )
                        logging_loss = tr_loss

                        self._log(logs)
                    # if self.args.evaluate_during_training and ((self.global_step % self.args.eval_steps == 0)
                    #                                            or log_first):
                    if self.args.evaluate_during_training and self.global_step % self.args.eval_steps == 0:
                        log_first=False

                        logger.debug('eval start')
                        eval_output = self.evaluate(self.eval_dataset, description='evaluation')
                        eval_output_metrics = eval_output.metrics
                        # eval_output_dir = os.path.join(self.args.output_dir,
                        #                           f"eval_output-{self.global_step}")
                        # with open(eval_output_dir, 'wb') as f:
                        #     pickle.dump(eval_output, f)
                        # del eval_output


                        logger.debug('done eval start')

                        validation_output = self.evaluate(self.validation_dataset, description='validation')
                        validation_output_metrics = validation_output.metrics

                        # eval_output_dir = os.path.join(self.args.output_dir,
                        #                           f"validation_output-{self.global_step}")
                        # with open(eval_output_dir, 'wb') as f:
                        #     pickle.dump(validation_output, f)
                        # del validation_output
                    # if self.model_config.eval_milestone_steps == 0:
                    #     self.model_config['step'] = self.global_step
                    #     self.tb_writer.add_hparams(self.model_config,
                    #                            metric_dict={k: output_metrics[k] for k in self.metrics_name_list})

                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:

                        # if self.global_step % (self.args.save_steps * 8) == 0:
                        #     logger.debug('start train eval')
                        #     output_metrics = self.evaluate(self.train_dataset,
                        #                                    description='train')  # evaluate train when saving
                        #     del output_metrics
                        #     logger.debug('done train eval')
                        #     logger.debug(f'CHECKING {self.global_step}')
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(model, "module"):
                            assert model.module is self.model
                        else:
                            assert model is self.model
                        # Save model checkpoint
                        output_dir = os.path.join(self.args.output_dir,
                                                  f"interval-{self.global_step}")
                        self.save_model(output_dir)

                        # logger.debug(f'before eval loss best {self.global_step}')
                        # if validation_output_metrics['validation_slot_value_loss'] < best_eval_loss:
                        #     prefix_checkpoint_dir = BEST_PREFIX_CHECKPOINT_DIR
                        #     best_eval_loss = validation_output_metrics['validation_slot_value_loss']
                        #     output_dir = os.path.join(self.args.output_dir,
                        #                               f"{prefix_checkpoint_dir}-{self.global_step}")
                        #     self.save_model(output_dir)
                        #     # logger.info('SAVE ONLY BEST MODEL')
                        #
                        # if validation_output_metrics['validation_joint_slot_accuracy'] > best_acc:
                        #     prefix_checkpoint_dir = BEST_ACC_PREFIX_CHECKPOINT_DIR
                        #     best_acc = validation_output_metrics['validation_joint_slot_accuracy']
                        #     output_dir = os.path.join(self.args.output_dir,
                        #                               f"{prefix_checkpoint_dir}-{self.global_step}")
                        #     self.save_model(output_dir)
                            # logger.info('SAVE ONLY BEST MODEL')



                        # if self.is_world_master():
                        #     logger.info('CHECK rotate')
                        #     self._rotate_checkpoints(checkpoint_prefix=prefix_checkpoint_dir)
                        #     logger.info('after rotate')
                        # logger.info('after SAVE ONLY BEST MODEL')
                        if is_torch_tpu_available():
                            xm.rendezvous("saving_optimizer_states")
                            xm.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            xm.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        elif self.is_world_master():
                            logger.debug('save optimier and scheduler')
                            # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            ckpt_file_path = os.path.join(output_dir, 'check_point_dict.pth')
                            torch.save({
                                        'optimizer': optimizer.state_dict(),
                                        'scheduler': scheduler.state_dict(),
                                        # 'amp': amp.state_dict(),
                                        'steps': self.global_step
                                        }
                                       , ckpt_file_path)
                            logger.debug('done optimier and scheduler')
                        logger.debug('going on saving optimier and scheduler')

                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    logger.debug(f'BEFORE CLOSE {self.global_step}')
                    epoch_iterator.close()
                    break
            # except KeyboardInterrupt:
            #     eval_output_metrics = self.evaluate()
            #     validation_output_metrics = self.evaluate(self.validation_dataset, description='validation')
            #     output_dir = os.path.join(self.args.output_dir,
            #                                           f"manual_save-{self.global_step}")
            #     self.save_model(output_dir)

            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                logger.debug(f'TRAIN BEFORE CLOSE {self.global_step}')
                train_iterator.close()
                break
            if self.args.tpu_metrics_debug or self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        if self.tb_writer:
            self.tb_writer.close()
        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step)

    def _log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        # print('log_keys', logs.keys())
        if self.epoch is not None:
            logs["epoch"] = self.epoch
        if self.global_step is None:
            # when logging evaluation metrics without training
            self.global_step = 0
        if self.tb_writer:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, self.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        '"%s" of type %s for key "%s" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute.",
                        v,
                        type(v),
                        k,
                    )
            self.tb_writer.flush()
        if is_wandb_available():
            if self.is_world_master():
                wandb.log(logs, step=self.global_step)
        output = {**logs, **{"step": self.global_step}}
        # if iterator is not None:
        #     iterator.write(output)
        # else:
        #     logger.info(f"output metrics {output}")

    def _training_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        outputs = model(**inputs)
        loss = outputs['loss']  # model outputs are always tuple in transformers (see doc)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.item()

    def is_local_master(self) -> bool:
        if is_torch_tpu_available():
            return xm.is_master_ordinal(local=True)
        else:
            return self.args.local_rank in [-1, 0]

    def is_world_master(self) -> bool:
        """
        This will be True only in one process, even in distributed mode,
        even when training on multiple machines.
        """
        if is_torch_tpu_available():
            return xm.is_master_ordinal(local=False)
        else:
            return self.args.local_rank == -1 or torch.distributed.get_rank() == 0

    def save_model(self, output_dir: Optional[str] = None):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.

        Will only save from the world_master process (unless in TPUs).
        """

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif self.is_world_master():
            self._save(output_dir)

    def _save_tpu(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        logger.info("Saving model checkpoint to %s", output_dir)

        if xm.is_master_ordinal():
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")

        xm.rendezvous("saving_checkpoint")
        self.model.save_pretrained(output_dir)

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")
        self.model.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _sorted_checkpoints(self, checkpoint_prefix, use_mtime=False) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(self.args.output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(self, checkpoint_prefix, use_mtime=False) -> None:
        logger.debug('rotate CHECKING')
        print('rotate CHECKING')
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            print('rotate CHECKING outget')
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(checkpoint_prefix, use_mtime=use_mtime)
        print('rotate CHECKING after sort')
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return
        print('rotate CHECKING number check')
        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)

    def evaluate(self, eval_dataset: Optional[Dataset] = None, description='eval') -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`.
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        logger.debug('before evaluator')
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        logger.debug('before pred loop')
        output = self._prediction_loop(eval_dataloader, description)
        logger.debug('after pred')
        self._log(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output

    def evaluate_autoregressive(self, eval_dataset: Optional[Dataset] = None, description='eval', global_step=0) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`.
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        self.global_step = global_step
        logger.debug('before evaluator')

        assert self.args.eval_batch_size == 1
        # eval_dataloader = self.get_eval_dataloader(eval_dataset)
        logger.debug('before pred loop')
        output = self._prediction_loop_autoregressive(eval_dataset, description)
        logger.debug('after pred')
        self._log(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output

    def predict(self, test_dataset: Dataset) -> DSTPredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:obj:`Dataset`):
                Dataset to run the predictions on.
        Returns:
            `NamedTuple`:
            predictions (:obj:`np.ndarray`):
                The predictions on :obj:`test_dataset`.
            label_ids (:obj:`np.ndarray`, `optional`):
                The labels (if the dataset contained some).
            metrics (:obj:`Dict[str, float]`, `optional`):
                The potential dictionary of metrics (if the dataset contained labels).
        """
        test_dataloader = self.get_test_dataloader(test_dataset)

        return self._prediction_loop(test_dataloader, description="Prediction")

    def _prediction_loop_autoregressive(
            self, dataset: Dataset, description: str, prediction_loss_only: Optional[bool] = None
    ) -> DSTPredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """
        # logger.debug('pred loop start')
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        model = self.model


        # eval_losses: List[float] = []
        losses_dict: Dict[str, list] = collections.defaultdict(list)
        gate_label_ids: torch.Tensor = None
        gate_preds: torch.Tensor = None
        slot_preds: torch.Tensor = None
        slot_label_ids: torch.Tensor = None

        model.eval()

        # with open(f'data/trade_slot_value_dict.pkl', 'rb') as f:
        #     slot_value_dict = pickle.load(f)

        slot_value_dict = dataset.slot_value_dict

        slot_idx_to_value = {key: {v: k for k, v in tmp.items()} for key, tmp in slot_value_dict.items()}
        for kk, vv in slot_idx_to_value.items():
            vv[len(vv)-1] = '<unk>'


        domain_slot_list = dataset.domain_slot_list
        id_list = dataset.ids

        prev_slot_label = ['none'] * 30
        # prev_ds_history = dataset.get_prev_ds_string(prev_slot_label, True)
        prev_id = ''
        prev_turn_id = -1

        for idx_tmp in tqdm(range(len(dataset)), desc=description):
            with torch.no_grad():
                if prev_id == id_list[idx_tmp]:

                    # for slot_idx, val in enumerate(outputs['slot_value_logits_list']):
                    #     print('@@', 'slot_idx:', slot_idx, 'pred:', val.argmax().item(), val.argmax().item() in slot_idx_to_value[slot_idx])

                    prev_slot_label = [slot_idx_to_value[slot_idx][val.argmax().item()] for slot_idx, val in enumerate(outputs['slot_value_logits_list'])]
                else:
                    prev_slot_label = ['none'] * 30
                    prev_id = id_list[idx_tmp]
                    assert dataset.turn_ids[idx_tmp] == 0
                prev_ds_history = dataset.get_prev_ds_string([prev_slot_label], True)

                inputs_raw = dataset.tokenizer(text=[dataset.dialog_history[idx_tmp]],
                                                     text_pair=prev_ds_history,
                                                     truncation=True,
                                                     max_length=dataset.max_length,
                                                     add_special_tokens=False,
                                                     return_token_type_ids=True,
                                                     return_attention_mask=True
                                                     )
                inputs_raw['gate_labels'] = [dataset.gate_label[idx_tmp]]
                inputs_raw['slot_labels'] = [dataset.slot_label[idx_tmp]]

                inputs = dict()
                for k, v in inputs_raw.items():
                    inputs[k] = torch.tensor(v, dtype=torch.long).to(self.args.device)

                outputs = model(**inputs)

                for key, val in outputs.items():

                    if key.endswith('loss'):
                        losses_dict[key].append(val.mean().item())

            if not prediction_loss_only:
                if 'gate_labels' in inputs:
                    gate_label_ids_tmp = inputs["gate_labels"]
                    if 'gate_logits' not in outputs:
                        gate_logits = outputs['domain_gate_logits']
                    else:
                        gate_logits = outputs['gate_logits']
                    assert (gate_label_ids_tmp <= 2).all(), f'my_dst_trainer 1166 individual {gate_label_ids_tmp[torch.where(gate_label_ids_tmp > 2)]}'

                    if gate_label_ids is None:
                        gate_label_ids = gate_label_ids_tmp.detach()
                    else:
                        gate_label_ids = torch.cat((gate_label_ids, gate_label_ids_tmp), dim=0)

                    gate_preds_tmp = gate_logits.argmax(2)
                    # gate_preds_tmp = (gate_logits > 0.5).to(torch.int64).flatten(1)
                    if gate_preds is None:
                        gate_preds = gate_preds_tmp.detach()
                    else:
                        gate_preds = torch.cat((gate_preds, gate_preds_tmp), dim=0)

                if 'slot_labels' in inputs:
                    slot_label_ids_tmp = inputs["slot_labels"]
                    slot_value_logits_list = outputs['slot_value_logits_list']

                    if slot_label_ids is None:
                        slot_label_ids = slot_label_ids_tmp.detach()
                    else:
                        slot_label_ids = torch.cat((slot_label_ids, slot_label_ids_tmp), dim=0)

                    slot_preds_tmp = torch.stack([x.argmax(1) for x in slot_value_logits_list], 1)

                    if slot_preds is None:
                        slot_preds = slot_preds_tmp.detach()
                    else:
                        slot_preds = torch.cat((slot_preds, slot_preds_tmp), dim=0)


                # for i in range(2):
                #     # if (slot_preds_tmp[i] == slot_label_ids_tmp[i]).all():
                #     if not (slot_preds_tmp[i] == slot_label_ids_tmp[i]).all():
                #         domain_list = set()
                #
                #         for slot_idx, slot_name in enumerate(slot_type_list):
                #             if slot_label_ids_tmp[i][slot_idx] != 0:
                #                 domain_list.add(slot_name.split('-')[0])
                #         if len(domain_list) >2:
                #             print('@@', cnt, i, '@@')
                #             cnt_idx_list.append((cnt, i))
                #             print(tokenizer.decode(inputs['input_ids'][i][30:]).split('<pad>')[0])
                #             for slot_idx, slot_name in enumerate(slot_type_list):
                #                 # if slot_preds_tmp[i][slot_idx] != 0:
                #                 tmp_slot_val = slot_idx_to_value[slot_idx]
                #                 print(slot_name, tmp_slot_val[slot_preds_tmp[i][slot_idx].item()],
                #                       tmp_slot_val[slot_label_ids_tmp[i][slot_idx].item()])
                # cnt += 1


        if self.args.local_rank != -1:

            if 'gate_labels' in inputs:
                if gate_label_ids is not None:
                    assert (gate_label_ids <= 2).all(), f'my_dst_trainer 1207. before distributed concat {gate_label_ids[torch.where(gate_label_ids > 2)]}'
                    # logger.debug('before label_id distirbuted concat')
                    gate_label_ids = self.distributed_concat(gate_label_ids, num_total_examples=self.num_examples(dataloader))
                    assert (gate_label_ids <= 2).all(), f'my_dst_trainer 1210. after distributedconcat {gate_label_ids[torch.where(gate_label_ids > 2)]}'
                if gate_preds is not None:
                    # logger.debug('before sop distirbuted concat')
                    gate_preds = self.distributed_concat(gate_preds, num_total_examples=self.num_examples(dataloader))


            if 'slot_labels' in inputs:
                if slot_label_ids is not None:
                    # logger.debug('before label_id distirbuted concat')

                    slot_label_ids = self.distributed_concat(slot_label_ids,
                                                             num_total_examples=self.num_examples(dataloader))

                if slot_preds is not None:
                    # logger.debug('before sop distirbuted concat')
                    slot_preds = self.distributed_concat(slot_preds, num_total_examples=self.num_examples(dataloader))

        # logger.debug('before sumup')
        if 'gate_labels' in inputs:
            assert (gate_label_ids <= 2).all(), f'my_dst_trainer 1229. before cpu numpy {gate_label_ids[torch.where(gate_label_ids > 2)]}'
            if gate_label_ids is not None:
                gate_label_ids = gate_label_ids.cpu().numpy()
            if gate_preds is not None:
                gate_preds = gate_preds.cpu().numpy()
            assert (gate_label_ids <= 2).all(), f'my_dst_trainer 1229. after cpu numpy {gate_label_ids}'

        if 'slot_labels' in inputs:
            if slot_label_ids is not None:
                slot_label_ids = slot_label_ids.cpu().numpy()
            if slot_preds is not None:
                slot_preds = slot_preds.cpu().numpy()

        # torch.cuda.synchronize() # added torch cuda synchronize https://www.facebook.com/groups/PyTorchKR/permalink/1298055203667491/

        if self.compute_metrics is not None:
            metrics = self.compute_metrics(DSTEvalPrediction(
                                                             gate_label_ids=gate_label_ids,
                                                             gate_predictions=gate_preds,
                                                             slot_label_ids=slot_label_ids,
                                                             slot_predictions=slot_preds,
                                                          ))
        else:
            metrics = {}

        for key, val in losses_dict.items():
            metrics[f"{description}_{key}"] = np.mean(val)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith(f"{description}_"):
                metrics[f"{description}_{key}"] = metrics.pop(key)

        return DSTPredictionOutput(
                                   gate_label_ids=gate_label_ids,
                                   gate_predictions=gate_preds,
                                   slot_label_ids=slot_label_ids,
                                   slot_predictions=slot_preds,
                                   metrics=metrics)

    def _prediction_loop(
            self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> DSTPredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """
        # logger.debug('pred loop start')
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        model = self.model
        # logger.debug('before data parallel')
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        logger.debug('after data parallel')
        batch_size = dataloader.batch_size
        if self.is_world_master():
            logger.info("***** Running %s *****", description)
            logger.info("  Num examples = %d", self.num_examples(dataloader))
            logger.info("  Batch size = %d", batch_size)
        # eval_losses: List[float] = []
        losses_dict: Dict[str, list] = collections.defaultdict(list)
        gate_label_ids: torch.Tensor = None
        gate_preds: torch.Tensor = None
        slot_preds: torch.Tensor = None
        slot_label_ids: torch.Tensor = None

        model.eval()

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            past = None

        # from my_tokenization_albert import AlbertTokenizer
        # tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        #
        # # if args.ds_type == 'split':
        # #     domain_special_tokens = [f'[D_{i}]' for i in range(5)] + [f'[S_{i}]' for i in range(17)]
        # # elif args.ds_type == 'merged':
        # domain_special_tokens = [f'[DS_{i}]' for i in range(30)]
        # tokenizer.add_special_tokens({'additional_special_tokens': ['[SEPT]'] + domain_special_tokens})
        # slot_type_list = ['hotel-price range', 'hotel-type', 'hotel-parking', 'hotel-book stay', 'hotel-book day',
        #                   'hotel-book people', 'hotel-area', 'hotel-stars', 'hotel-internet', 'train-destination',
        #                   'train-day',
        #                   'train-departure', 'train-arrive by', 'train-book people', 'train-leave at',
        #                   'attraction-area',
        #                   'restaurant-food', 'restaurant-price range', 'restaurant-area', 'attraction-name',
        #                   'restaurant-name',
        #                   'attraction-type', 'hotel-name', 'taxi-leave at', 'taxi-destination', 'taxi-departure',
        #                   'restaurant-book time', 'restaurant-book day', 'restaurant-book people', 'taxi-arrive by']
        # # print(' ')
        # import pickle

        # with open(f'data/trade_slot_value_dict.pkl', 'rb') as f:
        #     slot_value_dict = pickle.load(f)
        # slot_idx_to_value = {key: {v: k for k, v in tmp.items()} for key, tmp in slot_value_dict.items()}
        #
        # domain_slot_list = dataloader.dataset.domain_slot_list


        cnt = 0
        cnt_idx_list = list()
        for inputs in tqdm(dataloader, desc=description):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.args.device)
            if self.args.past_index >= 0:
                inputs["mems"] = past

            with torch.no_grad():
                outputs = model(**inputs)

                for key, val in outputs.items():

                    if key.endswith('loss'):
                        losses_dict[key].append(val.mean().item())

            if not prediction_loss_only:
                if 'gate_labels' in inputs:
                    gate_label_ids_tmp = inputs["gate_labels"]
                    if 'gate_logits' not in outputs:
                        gate_logits = outputs['domain_gate_logits']
                    else:
                        gate_logits = outputs['gate_logits']
                    assert (gate_label_ids_tmp <= 2).all(), f'my_dst_trainer 1166 individual {gate_label_ids_tmp[torch.where(gate_label_ids_tmp > 2)]}'

                    if gate_label_ids is None:
                        gate_label_ids = gate_label_ids_tmp.detach()
                    else:
                        gate_label_ids = torch.cat((gate_label_ids, gate_label_ids_tmp), dim=0)

                    gate_preds_tmp = gate_logits.argmax(2)
                    # gate_preds_tmp = (gate_logits > 0.5).to(torch.int64).flatten(1)
                    if gate_preds is None:
                        gate_preds = gate_preds_tmp.detach()
                    else:
                        gate_preds = torch.cat((gate_preds, gate_preds_tmp), dim=0)

                if 'slot_labels' in inputs:
                    slot_label_ids_tmp = inputs["slot_labels"]
                    slot_value_logits_list = outputs['slot_value_logits_list']

                    if slot_label_ids is None:
                        slot_label_ids = slot_label_ids_tmp.detach()
                    else:
                        slot_label_ids = torch.cat((slot_label_ids, slot_label_ids_tmp), dim=0)

                    slot_preds_tmp = torch.stack([x.argmax(1) for x in slot_value_logits_list], 1)

                    if slot_preds is None:
                        slot_preds = slot_preds_tmp.detach()
                    else:
                        slot_preds = torch.cat((slot_preds, slot_preds_tmp), dim=0)


                # for i in range(2):
                #     # if (slot_preds_tmp[i] == slot_label_ids_tmp[i]).all():
                #     if not (slot_preds_tmp[i] == slot_label_ids_tmp[i]).all():
                #         domain_list = set()
                #
                #         for slot_idx, slot_name in enumerate(slot_type_list):
                #             if slot_label_ids_tmp[i][slot_idx] != 0:
                #                 domain_list.add(slot_name.split('-')[0])
                #         if len(domain_list) >2:
                #             print('@@', cnt, i, '@@')
                #             cnt_idx_list.append((cnt, i))
                #             print(tokenizer.decode(inputs['input_ids'][i][30:]).split('<pad>')[0])
                #             for slot_idx, slot_name in enumerate(slot_type_list):
                #                 # if slot_preds_tmp[i][slot_idx] != 0:
                #                 tmp_slot_val = slot_idx_to_value[slot_idx]
                #                 print(slot_name, tmp_slot_val[slot_preds_tmp[i][slot_idx].item()],
                #                       tmp_slot_val[slot_label_ids_tmp[i][slot_idx].item()])
                # cnt += 1


        if self.args.local_rank != -1:

            if 'gate_labels' in inputs:
                if gate_label_ids is not None:
                    assert (gate_label_ids <= 2).all(), f'my_dst_trainer 1207. before distributed concat {gate_label_ids[torch.where(gate_label_ids > 2)]}'
                    # logger.debug('before label_id distirbuted concat')
                    gate_label_ids = self.distributed_concat(gate_label_ids, num_total_examples=self.num_examples(dataloader))
                    assert (gate_label_ids <= 2).all(), f'my_dst_trainer 1210. after distributedconcat {gate_label_ids[torch.where(gate_label_ids > 2)]}'
                if gate_preds is not None:
                    # logger.debug('before sop distirbuted concat')
                    gate_preds = self.distributed_concat(gate_preds, num_total_examples=self.num_examples(dataloader))


            if 'slot_labels' in inputs:
                if slot_label_ids is not None:
                    # logger.debug('before label_id distirbuted concat')

                    slot_label_ids = self.distributed_concat(slot_label_ids,
                                                             num_total_examples=self.num_examples(dataloader))

                if slot_preds is not None:
                    # logger.debug('before sop distirbuted concat')
                    slot_preds = self.distributed_concat(slot_preds, num_total_examples=self.num_examples(dataloader))

        # logger.debug('before sumup')
        if 'gate_labels' in inputs:
            assert (gate_label_ids <= 2).all(), f'my_dst_trainer 1229. before cpu numpy {gate_label_ids[torch.where(gate_label_ids > 2)]}'
            if gate_label_ids is not None:
                gate_label_ids = gate_label_ids.cpu().numpy()
            if gate_preds is not None:
                gate_preds = gate_preds.cpu().numpy()
            assert (gate_label_ids <= 2).all(), f'my_dst_trainer 1229. after cpu numpy {gate_label_ids}'

        if 'slot_labels' in inputs:
            if slot_label_ids is not None:
                slot_label_ids = slot_label_ids.cpu().numpy()
            if slot_preds is not None:
                slot_preds = slot_preds.cpu().numpy()

        torch.cuda.synchronize() # added torch cuda synchronize https://www.facebook.com/groups/PyTorchKR/permalink/1298055203667491/

        if self.compute_metrics is not None:
            metrics = self.compute_metrics(DSTEvalPrediction(
                                                             gate_label_ids=gate_label_ids,
                                                             gate_predictions=gate_preds,
                                                             slot_label_ids=slot_label_ids,
                                                             slot_predictions=slot_preds,
                                                          ))
        else:
            metrics = {}

        for key, val in losses_dict.items():
            metrics[f"{description}_{key}"] = np.mean(val)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith(f"{description}_"):
                metrics[f"{description}_{key}"] = metrics.pop(key)

        return DSTPredictionOutput(
                                   gate_label_ids=gate_label_ids,
                                   gate_predictions=gate_preds,
                                   slot_label_ids=slot_label_ids,
                                   slot_predictions=slot_preds,
                                   metrics=metrics)

    def distributed_concat(self, tensor: torch.Tensor, num_total_examples: int) -> torch.Tensor:

        assert self.args.local_rank != -1

        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]

        # logger.info(f'{[torch.unique(x) for x in output_tensors]} unique ## my_dst_trainer 1282')

        torch.distributed.all_gather(output_tensors, tensor)

        # logger.info(f'{[torch.unique(x) for x in output_tensors]} unique ## AFTER ALL_GATHER my_dst_trainer 1286')

        concat = torch.cat(output_tensors, dim=0)

        # logger.info(f'{torch.unique(concat)} unique ## my_dst_trainer 1290 after concat')

        # truncate the dummy elements added by SequentialDistributedSampler
        output = concat[:num_total_examples]

        return output
