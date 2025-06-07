import os
import wandb
import torch
import torch.nn as nn


from torch.utils.data import Sampler
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from transformers import TrainerCallback
from torch.utils.data import DataLoader
from transformers.trainer_utils import seed_worker
from typing import List, Optional, Dict, Union, Any, Tuple
from few_shot_learning_system import SupervisedTrainer, MetaTrainer

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):
    def __init__(
        self,
        model=None,
        tokenizer=None,
        args=None,
        *trainer_args, 
        **trainer_kwargs
    ):
        super().__init__(model=model, tokenizer=tokenizer, args=args, *trainer_args, **trainer_kwargs)     
        self.sl_trainer= SupervisedTrainer(model, args)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()
        
    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()

        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.sl_trainer.forward(inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)
    
        return loss.detach() / self.args.gradient_accumulation_steps

    def prediction_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], 
                        prediction_loss_only: bool,
                        ignore_keys: Optional[List[str]] = None,
                        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        inputs = self._prepare_inputs(inputs)
        loss = None
        with self.compute_loss_context_manager():
            loss = self.sl_trainer.forward(inputs)

        return (loss, None, None)


    def _save_checkpoint(self, model, trial, metrics=None):
        
        if getattr(self.args, 'tune_mm_mlp_adapter', False):

            # Only save Adapter
            ckpt_dir = os.path.join(self.args.output_dir,f"ckpt_{self.state.global_step}")
            if self.model.config.task_head:
                keys_to_match = ['mm_projector', 'lm_head_w', 'vision_resampler']
            else:
                keys_to_match = ['mm_projector', 'vision_resampler']
            
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(ckpt_dir)
                torch.save(weight_to_save, os.path.join(ckpt_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

class MAPD_LLaVA_Trainer(Trainer):
    def __init__(
        self,
        model=None,
        tokenizer=None,
        args=None,
        *trainer_args, 
        **trainer_kwargs
    ):
        super().__init__(model=model, tokenizer=tokenizer, args=args, *trainer_args, **trainer_kwargs)
        self.best_val_loss = float("inf")
        self.meta_trainer= MetaTrainer(model, args)
         
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        opt_model = self.model
        
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            optimizer_kwargs["lr"] = self.args.meta_learning_rate

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # MAML TRAINING LOOP
        supp_inp_set = {}
        qry_inp_set = {}
        
        supp_inp_set["input_ids"] = inputs["supp_input_ids"]
        supp_inp_set["labels"] = inputs["supp_labels"]
        supp_inp_set["attention_mask"] = inputs["supp_attention_mask"]
        supp_inp_set["images"] = inputs["supp_images"]

        qry_inp_set["input_ids"] = inputs["qry_input_ids"]
        qry_inp_set["labels"] = inputs["qry_labels"]
        qry_inp_set["attention_mask"] = inputs["qry_attention_mask"]
        qry_inp_set["images"] = inputs["qry_images"]

        with self.compute_loss_context_manager():
            loss = self.meta_trainer.forward(supp_inp_set, qry_inp_set, self.state.epoch,
                                            use_second_order=self.args.second_order and
                                                            self.state.epoch > self.args.first_order_to_second_order_epoch,
                                            use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization,
                                            num_steps=self.args.number_of_training_steps_per_iter,
                                            training_phase=True, grad_acc_part=self.args.grad_acc_part, inner_checkpointing=self.args.inner_checkpointing)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        
        self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    def prediction_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], 
                        prediction_loss_only: bool,
                        ignore_keys: Optional[List[str]] = None,
                        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        supp_inp_set = {}
        qry_inp_set = {}
        
        inputs = self._prepare_inputs(inputs)
        supp_inp_set["input_ids"] = inputs["supp_input_ids"]
        supp_inp_set["labels"] = inputs["supp_labels"]
        supp_inp_set["attention_mask"] = inputs["supp_attention_mask"]
        supp_inp_set["images"] = [te.to(torch.bfloat16) for te in inputs["supp_images"]]


        qry_inp_set["input_ids"] = inputs["qry_input_ids"]
        qry_inp_set["labels"] = inputs["qry_labels"]
        qry_inp_set["attention_mask"] = inputs["qry_attention_mask"]
        qry_inp_set["images"] = [te.to(torch.bfloat16) for te in inputs["qry_images"]]

        loss = None
        with self.compute_loss_context_manager():
            loss = self.meta_trainer.forward(supp_inp_set, qry_inp_set, self.state.epoch, use_second_order=self.args.second_order and
                                            self.state.epoch > self.args.first_order_to_second_order_epoch,
                                            use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization,
                                            num_steps=self.args.number_of_evaluation_steps_per_iter,
                                            training_phase=False, grad_acc_part=self.args.grad_acc_part, inner_checkpointing=self.args.inner_checkpointing)

        return (loss, None, None)

    def _save_checkpoint(self, model, trial, metrics=None):
        
        if getattr(self.args, 'tune_mm_mlp_adapter', False):

            ckpt_dir = os.path.join(self.args.output_dir,f"ckpt_{self.state.global_step}")
            if self.model.config.task_head:
                keys_to_match = ['mm_projector', 'lm_head_w', 'vision_resampler']
            else:
                keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])
            
            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(ckpt_dir)
                torch.save(weight_to_save, os.path.join(ckpt_dir, f'mm_projector.bin'))
        else:
            super(MAPD_LLaVA_Trainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(MAPD_LLaVA_Trainer, self)._save(output_dir, state_dict)

class SaveModelCallback(TrainerCallback):
    def __init__(self):
        self.last_logged_step = -1
    
    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            trainer = kwargs.get("trainer")
            if trainer is not None:
                model = trainer.model
        ckpt_dir = os.path.join(args.output_dir,f"ckpt_{str(int(state.global_step))}")
        if model.config.task_head:
            keys_to_match = ['mm_projector', 'lm_head_w', 'vision_resampler']
        else:
            keys_to_match = ['mm_projector', 'vision_resampler']

        if getattr(args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(model.named_parameters(), keys_to_match)

        if args.local_rank == 0 or args.local_rank == -1:
            model.config.save_pretrained(ckpt_dir)
            torch.save(weight_to_save, os.path.join(ckpt_dir, f'mm_projector.bin'))
 
class TasksDatasetCallback(TrainerCallback):
    def __init__(self, dataset):
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        train_dataloader = kwargs.get("train_dataloader")
        if state.epoch != 0:
            train_dataloader.dataset.task_creation(args.local_rank)
            if args.custom_sampler:
                train_dataloader.sampler.update_indices(train_dataloader.dataset.dataset_indices)

class ICTDatasetCallback(TrainerCallback):
    def __init__(self, dataset):
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        train_dataloader = kwargs.get("train_dataloader")
        if state.epoch != 0:
            train_dataloader.dataset.ict_ex_creation(args.local_rank)
            if args.custom_sampler:
                train_dataloader.sampler.update_indices(train_dataloader.dataset.dataset_indices)

class ICTZeroShotCallback(TrainerCallback):
    def __init__(self, dataset):
        self.dataset = dataset

    def on_step_end(self, args, state, control, **kwargs):
        train_dataloader = kwargs.get("train_dataloader")
        nz_shots = train_dataloader.dataset.nz_shot
        if args.local_rank == 0:
            wandb.log({"global_step": state.global_step, f"Zero-Shot-Count": nz_shots}) 

class InnerLRCallback(TrainerCallback):
    def __init__(self):      
        self.last_logged_step = -1

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step != self.last_logged_step:
            self.last_logged_step = state.global_step

            model = kwargs.get("model")
            if model is None:
                trainer = kwargs.get("trainer")
                if trainer is not None:
                    model = trainer.model
            
            if model is not None:
                for tup_model in model.inner_loop_optimizer.names_learning_rates_dict.items():
                    for e_step in range(args.number_of_training_steps_per_iter - 2): # Can change to more if necessary
                    # Optionally filter the parameters you care about
                        if tup_model[1].requires_grad and args.local_rank == 0:
                            wandb.log({"global_step": state.global_step, f"inner_lr/{tup_model[0]}_step_{e_step}": tup_model[1][e_step].item()})

        return control
