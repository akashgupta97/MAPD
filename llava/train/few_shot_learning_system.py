import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class SupervisedTrainer(nn.Module):
    def __init__(self, model, args=None):
        super(SupervisedTrainer, self).__init__()
        self.model = model
        self.args = args

    def trainable_parameter_dict(self, params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        return {
            name: param
            for name, param in params
            if "mm_projector" in name
        }
    def forward(self, batch):
        
        names_weights_copy = self.trainable_parameter_dict(self.model.named_parameters())
        idx_to_layer_name_dict = {int(layer_name.split(".")[-1]): layer_name for layer_name in names_weights_copy.keys()}
        outputs = self.model(**batch, fast_weights=names_weights_copy, i2l_dict=idx_to_layer_name_dict)
        
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        return loss    

class MetaTuning(nn.Module):
    def __init__(self, model,tokenizer, args=None):
        super(MetaTuning, self).__init__()
        self.model = model
        self.args = args
        self.tokenizer = tokenizer

        print("Outer Loop parameters")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.shape, param.device, param.requires_grad)
          
    def get_inner_loop_parameter_dict(self, params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """

        return {
            name: param
            for name, param in params
            if param.requires_grad
        }

    def get_across_task_loss_metrics(self, total_losses):
        losses = {'loss': torch.mean(torch.stack(total_losses))}
        return losses


    def apply_inner_loop_update(self, loss, names_weights_copy, use_second_order, current_step_idx):
        """
        Applies an inner loop update given current step's loss, the weights to update, a flag indicating whether to use
        second order derivatives and the current step's index.
        :param loss: Current step's loss with respect to the support set.
        :param names_weights_copy: A dictionary with names to parameters to update.
        :param use_second_order: A boolean flag of whether to use second order derivatives.
        :param current_step_idx: Current step's index.
        :return: A dictionary with the updated weights (name, param)
        """
        
        self.model.zero_grad(params=names_weights_copy)

        grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                    create_graph=use_second_order, allow_unused=True)
        names_grads_copy = dict(zip(names_weights_copy.keys(), grads))

        names_weights_copy = self.model.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                     names_grads_wrt_params_dict=names_grads_copy,
                                                                     num_step=current_step_idx)

        return names_weights_copy

    def finetuning(self, supp_set, use_second_order, num_eval_steps, device, validation, qry_set=None, grad_acc_part=1):
        n_tasks = len(supp_set["input_ids"])
        total_losses = []
        supp_losses = []
        

        for task_id in range(n_tasks):    
            task_losses = []
            supp_task_inputs = {key: val[task_id].to(device) for key, val in supp_set.items()}
            supp_task_inputs["images"] = supp_task_inputs["images"].to(torch.bfloat16)
            
            if grad_acc_part > 1:
                divided_task_inputs = {k:[] for k, v in supp_task_inputs.items()}
                s_shape = supp_task_inputs["images"].shape[0]
                chunk_size = int(s_shape/grad_acc_part)
                for k in divided_task_inputs.keys():
                    for sl in range(0, s_shape, chunk_size):
                        divided_task_inputs[k].append(supp_task_inputs[k][sl:sl+chunk_size])

            if validation:
                qry_task_inputs = {key: val[task_id].to(device) for key, val in qry_set.items()}
                qry_task_inputs["images"] = qry_task_inputs["images"].to(torch.bfloat16)
            
            names_weights_copy = self.get_inner_loop_parameter_dict(self.model.named_parameters())
            
            idx_to_layer_name_dict = {int(layer_name.split(".")[-1]): layer_name for layer_name in names_weights_copy.keys()}
            adapted_weights = []
        
            adapted_weights.append(self.get_inner_loop_parameter_dict(self.model.named_parameters()))

            for num_step in range(num_eval_steps):
                if grad_acc_part > 1:
                    for inn_idx in range(len(divided_task_inputs["input_ids"])):
                        inner_task_inputs = {k: v[inn_idx] for k,v in divided_task_inputs.items()}
                        supp_outputs = self.model(**inner_task_inputs, fast_weights=names_weights_copy, i2l_dict=idx_to_layer_name_dict, use_cache=False)
                        support_loss = supp_outputs["loss"] if isinstance(supp_outputs, dict) else supp_outputs[0]
                        supp_losses.append(support_loss.item())
                        
                        names_weights_copy = self.apply_inner_loop_update(loss=support_loss,
                                                                        names_weights_copy=names_weights_copy,
                                                                        use_second_order=use_second_order,
                                                                        current_step_idx=num_step)
                else:
                    supp_outputs = self.model(**supp_task_inputs, fast_weights=names_weights_copy, i2l_dict=idx_to_layer_name_dict, use_cache=False)
                    support_loss = supp_outputs["loss"] if isinstance(supp_outputs, dict) else supp_outputs[0]
                    supp_losses.append(support_loss.item())
        
                    names_weights_copy = self.apply_inner_loop_update(loss=support_loss,
                                                                    names_weights_copy=names_weights_copy,
                                                                    use_second_order=use_second_order,
                                                                    current_step_idx=num_step)
            
                adapted_weights.append(names_weights_copy)
        
                if validation and num_step == (num_eval_steps - 1):
                    qry_outputs = self.model(**qry_task_inputs, fast_weights=names_weights_copy, i2l_dict=idx_to_layer_name_dict, use_cache=False)
                    qry_loss = qry_outputs["loss"] if isinstance(qry_outputs, dict) else qry_outputs[0]   
                    task_losses.append(qry_loss)
            
            if validation:
                task_losses = torch.sum(torch.stack(task_losses))
                total_losses.append(task_losses)        
        
        
        if validation:
            val_losses = self.get_across_task_loss_metrics(total_losses=total_losses)        
            return val_losses["loss"]
        
        return supp_losses, adapted_weights

class MetaTrainer(nn.Module):
    def __init__(self, model, args=None):
        """
        Initializes a MAML few shot learning system
        """
        super(MetaTrainer, self).__init__()

        self.args = args
        self.device = self.args.device
       
        self.model = model
        if self.args.local_rank == 0:
            print("Inner Loop parameters")
            for key, param in self.model.named_parameters():
                if "names_learning_rates" not in key and param.requires_grad:
                    print(key, param.shape)

            print("Outer Loop parameters")
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(name, param.shape, param.device, param.requires_grad)

    def get_per_step_loss_importance_vector(self, current_epoch):

        loss_weights = np.ones(shape=(self.args.number_of_training_steps_per_iter)) * (
                1.0 / self.args.number_of_training_steps_per_iter)
        decay_rate = 1.0 / self.args.number_of_training_steps_per_iter / self.args.multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / self.args.number_of_training_steps_per_iter
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (current_epoch * (self.args.number_of_training_steps_per_iter - 1) * decay_rate),
            1.0 - ((self.args.number_of_training_steps_per_iter - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights
        
    def get_inner_loop_parameter_dict(self, params):
    
        # return {
        #     name: param
        #     for name, param in params
        #     if param.requires_grad and "names_learning_rates" not in name
        #     and (
        #         not self.args.enable_inner_loop_optimizable_bn_params
        #         and "norm_layer" not in name
        #         or self.args.enable_inner_loop_optimizable_bn_params
        #     )
        # }

        return {
            name: param
            for name, param in params
            if "mm_projector" in name and "names_learning_rates" not in name
        }


    def apply_inner_loop_update(self, loss, names_weights_copy, use_second_order, current_step_idx, acc_grads=None, update=True):
        
        #INNER LOOP GRADIENT ACCUMULATION 
        grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                create_graph=use_second_order, allow_unused=True)
        
        if acc_grads is not None:
            acc_grads = [acc + g for acc, g in zip(acc_grads, grads)]
            names_grads_copy = dict(zip(names_weights_copy.keys(), acc_grads))
        else:
            names_grads_copy = dict(zip(names_weights_copy.keys(), grads))
        
        if update:          
            names_weights_copy = self.model.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                    names_grads_wrt_params_dict=names_grads_copy,
                                                                    num_step=current_step_idx)
        return names_weights_copy

    def get_across_task_loss_metrics(self, total_losses):
        losses = {'loss': torch.mean(torch.stack(total_losses))}
        return losses

    def forward(self, supp_inp_set, qry_inp_set, epoch, use_second_order, use_multi_step_loss_optimization, num_steps, training_phase, grad_acc_part, inner_checkpointing):
        """
        Runs a forward outer loop pass on the batch of tasks using the MAML/++ framework.
        :param supp_inp_set: A data batch containing the support set.
        :param qry_inp_set: A data batch containing the query set.
        :param epoch: Current epoch's index
        :param use_second_order: A boolean saying whether to use second order derivatives.
        :param use_multi_step_loss_optimization: Whether to optimize on the outer loop using just the last step's
        :param num_steps: Number of inner loop steps.
        :param training_phase: Whether this is a training phase (True) or an evaluation phase (False)
        :param grad_acc_part: partitions batch for inner loop gradient accumulation to save memory
        :return: A dictionary with the collected losses of the current outer forward propagation.
        """

        n_tasks = len(supp_inp_set["input_ids"])
        total_losses = []

        for task_id in range(n_tasks):
            
            supp_task_inputs = {key: val[task_id] for key, val in supp_inp_set.items()}
            qry_task_inputs = {key: val[task_id] for key, val in qry_inp_set.items()}
            
            task_losses = []
            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector(current_epoch=epoch)
            names_weights_copy = self.get_inner_loop_parameter_dict(self.model.named_parameters())
            idx_to_layer_name_dict = {int(layer_name.split(".")[-1]): layer_name for layer_name in names_weights_copy.keys() if "lm_head" not in layer_name}
            
            if inner_checkpointing:
                divided_task_inputs = {k:[] for k, v in supp_task_inputs.items()}
                s_shape = supp_task_inputs["images"].shape[0]
                chunk_size = max(1, int(s_shape/grad_acc_part))
                for k in divided_task_inputs.keys():          
                    for sl in range(0, s_shape, chunk_size):
                        divided_task_inputs[k].append(supp_task_inputs[k][sl:sl+chunk_size])
                
                weight_keys = list(names_weights_copy.keys())
                weights = list(names_weights_copy.values())

                # Define a differentiable function for maml inner loop so that torch activation checkpoint() works
                def _maml_internal(inn_idx, *input_tuple):                    
                    names_inner_weights_dict = {weight_keys[i]:input_tuple[i] for i in range(len(weight_keys))}
                    inner_task_inputs = {k: v[inn_idx] for k,v in divided_task_inputs.items()}
          
                    supp_outputs = self.model(
                    **inner_task_inputs, fast_weights=names_inner_weights_dict, i2l_dict=idx_to_layer_name_dict, use_cache=False
                    )
                    support_loss = supp_outputs["loss"] if isinstance(supp_outputs, dict) else supp_outputs[0]
                    outputs = (support_loss)
                    return outputs
                            
                sweights = tuple(weights)

            update=True 
            for num_step in range(num_steps):
            
                if inner_checkpointing:
                    update=False
                    accumulated_grads = [torch.zeros_like(param) for param in names_weights_copy.values()]
                    # DISABLE DEEPSPEED GRADIENT CHECKPOINTING
                    self.model.gradient_checkpointing_disable()

                    # Inner loop checkpointing - Reduces inner loop activation memory
                    # but also adds a bit of overhead when nesting deepspeed gradient checkpointing
                    # and torch.utils.checkpoint. In some cases, not using torch.utils.checkpoint might work
                    # just fine

                    for inn_idx in range(len(divided_task_inputs["input_ids"])):  
                        ckpt_loss = checkpoint(_maml_internal, torch.as_tensor(inn_idx).to(self.device), *sweights, use_reentrant=False)
                    
                        if inn_idx == len(divided_task_inputs["input_ids"]) - 1:
                            update=True
                        
                        with torch.enable_grad():
                            names_weights_copy = self.apply_inner_loop_update(loss=ckpt_loss,
                                                                        names_weights_copy=names_weights_copy,
                                                                        use_second_order=use_second_order,
                                                                        current_step_idx=num_step,
                                                                        acc_grads=accumulated_grads, update=update)
                        sweights = tuple(list(names_weights_copy.values()))
                    
                    # ENABLE DEEPSPEED GRADIENT CHECKPOINTING
                    self.model.gradient_checkpointing_enable()
                else:
                    supp_outputs = self.model(**supp_task_inputs, fast_weights=names_weights_copy, i2l_dict=idx_to_layer_name_dict, use_cache=False)
                    support_loss = supp_outputs["loss"] if isinstance(supp_outputs, dict) else supp_outputs[0]
                    names_weights_copy = self.apply_inner_loop_update(loss=support_loss,
                                                                        names_weights_copy=names_weights_copy,
                                                                        use_second_order=use_second_order,
                                                                        current_step_idx=num_step)
                
                if use_multi_step_loss_optimization and training_phase and epoch < self.args.multi_step_loss_num_epochs:
                    
                    qry_outputs = self.model(**qry_task_inputs, fast_weights=names_weights_copy, i2l_dict=idx_to_layer_name_dict, use_cache=False)
                    qry_loss = qry_outputs["loss"] if isinstance(qry_outputs, dict) else qry_outputs[0]
                
                    task_losses.append(per_step_loss_importance_vectors[num_step] * qry_loss)
                
                elif num_step == (num_steps - 1):
                    qry_outputs = self.model(**qry_task_inputs, fast_weights=names_weights_copy, i2l_dict=idx_to_layer_name_dict, use_cache=False)
                    qry_loss = qry_outputs["loss"] if isinstance(qry_outputs, dict) else qry_outputs[0] 
                    task_losses.append(qry_loss)
            
            task_losses = torch.sum(torch.stack(task_losses))
            total_losses.append(task_losses)
            
        losses = self.get_across_task_loss_metrics(total_losses=total_losses)
        
        return losses["loss"]