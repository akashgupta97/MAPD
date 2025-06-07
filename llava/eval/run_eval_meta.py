import os
import re
import sys
import copy
import torch
import wandb
import random
import requests
import transformers
import numpy as np

from PIL import Image
from io import BytesIO
from transformers import TextStreamer


from PIL import Image
from typing import Dict, Optional
from torch.utils.data import DataLoader
from dataclasses import dataclass, field

from llava.model import *
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.eval.ICL_utils import get_task_instruction
from llava.conversation import conv_templates, SeparatorStyle
from llava.train.few_shot_learning_system import MetaTuning
from llava.train.inner_loop_optimizers import LSLRGradientDescentLearningRule
from llava.data.training_datasets import TestDataArguments, VLICLTestDataset, DataCollatorForVLICLDataset
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

random.seed(0)
torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)

@dataclass
class TestArguments:
    model_path: Optional[str] = field(default="liuhaotian/llava-v1.5-13b")
    version: Optional[str] = field(default="v0")
    model_base: Optional[str] = field(default=None)
    model_vision: Optional[str] = field(default=None)
    device: Optional[str] = field(default="cuda")
    conv_mode: Optional[str] = field(default=None)
    temperature: Optional[float] = field(default=0.2)
    max_new_tokens: Optional[int] = field(default=100)
    load_8bit: bool = field(default=False)
    load_4bit: bool = field(default=False)
    use_flash_attn: bool = field(default=True)
    run_name: Optional[str] = field(default="default_train2")
    project_name: Optional[str] = field(default="default")
    debug: bool = field(default=True)
    conv_mode: Optional[str] = field(default=None)
    finetuning: bool = field(default=True)
    in_context: bool = field(default=True)
    

@dataclass
class TestModelArguments:
    model_max_length: Optional[int] = field(default=512)
    task_learning_rate: Optional[float] = field(default=0.1)
    number_of_evaluation_steps_per_iter: Optional[int] = field(default=5)
    learnable_per_layer_per_step_inner_loop_learning_rate: Optional[bool] = field(default=False)
    enable_inner_loop_optimizable_bn_params: Optional[bool] = field(default=False)
    second_order: Optional[bool] = field(default=True)
    multi_step_loss_num_epochs: Optional[int] = field(default=1)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    extrapolate_lr: bool = field(default=False)
    add_dropout: bool = field(default=False)
    grad_acc_part: int = field(default=1)

    
def exact_match(prediction, answer, dataset):

    if "Answer" in prediction: 
        prediction = prediction.replace("Answer", "")
    if "answer" in prediction: 
        prediction = prediction.replace("answer", "")
    
    prediction = prediction.strip(':')
    prediction = prediction.strip()
    prediction = prediction.strip('\n')
    trunc_index = prediction.find('\n')
    
    if trunc_index <= 0:
        trunc_index = prediction.find('.')
    if trunc_index > 0:
        prediction = prediction[:trunc_index]
    if 'operator_induction' in dataset or 'clevr' in dataset:
        # find the number

        match = re.search(r'\d+', prediction)
        if match:
            prediction = match.group()
        else:
            prediction = ''
    
    if str(prediction).lower() == str(answer).lower():
        score = 1
    else:
        score = 0

    return score

def icl_concat(list_of_convs):
    fin_str = ""
    for d1 in list_of_convs:
        if d1["from"] == "human":
            fin_str +=d1["value"].lstrip("<image>\n")
        if d1["from"] == "gpt":
            fin_str = fin_str + " Answer: " + f"{d1['value']}"
    return fin_str

def make_prompt(prompt, add_prompt, task_ins, chosen_ds):
    if chosen_ds == "open_mi":
        if add_prompt:
            return prompt + f"? {task_ins}"
        else:
            return prompt + "?"
    else:
        if add_prompt:
            return prompt + f" {task_ins}"
        else:
            return prompt

def load_image(image_file):

    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def load_log_dict(log_dict):
    
    qry_ex =  log_dict[0]
    ques = qry_ex["conversations"][0]["value"].split("\n")[1]
    ans = qry_ex["conversations"][1]["value"]
    ques_type = qry_ex["type"]

    new_ques = qry_ex["conversations"][2]["value"]
    new_ans = qry_ex["conversations"][3]["value"]
    ques_img = qry_ex["image"]

    spt_img1 = log_dict[1]["image"]
    spt_img2 = log_dict[-1]["image"]
    conv1  = [d["value"] for d in log_dict[1]["conversations"]]

    spt_conv1 =  ", ".join(conv1)
    #print(spt_conv1)
    conv2 = [d["value"] for d in log_dict[-1]["conversations"]]
    spt_conv2 =  ", ".join(conv2)

    return spt_img1, spt_img2, spt_conv1, spt_conv2, ques, new_ques, ques_img, ans, new_ans, ques_type

def make_supervised_data_module(tokenizer,
                                data_args, model_version) -> Dict:
    """Make dataset and collator for fine-tuning."""
    
    train_dataset = VLICLTestDataset(tokenizer=tokenizer, data_args=data_args, model_version=model_version)
    data_collator = DataCollatorForVLICLDataset(tokenizer)

    return train_dataset, data_collator


def main():
    
    parser = transformers.HfArgumentParser((TestArguments, TestDataArguments, TestModelArguments))
    test_args, data_args, test_model_args = parser.parse_args_into_dataclasses()

    data_args.in_context = test_args.in_context
    query_task_instruction = get_task_instruction(data_args)
    wandb.init(dir="/workspace/storage/wandb", project=test_args.project_name, name=test_args.run_name)
    wandb.config.update(test_args)
    config = wandb.config

    disable_torch_init()
    model_name = get_model_name_from_path(test_args.model_path)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(test_args.model_path, test_args.model_base, test_args.model_vision, model_name, 
                                                                        test_args.load_8bit, test_args.load_4bit, device=test_args.device,use_flash_attn=test_args.use_flash_attn,
                                                                        add_dropout=test_model_args.add_dropout)    
    for name, param in model.named_parameters():
        if "mm_projector" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    def get_inner_loop_parameter_dict(params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        return {
            name: param
            for name, param in params
            if param.requires_grad
            and (
                not test_model_args.enable_inner_loop_optimizable_bn_params
                and "norm_layer" not in name
                or test_model_args.enable_inner_loop_optimizable_bn_params
            )
        }
    
    if test_args.finetuning:
        # Initialize inner loop optimizer class
        model.inner_loop_optimizer = LSLRGradientDescentLearningRule(device=test_args.device,
                                                                        init_learning_rate=test_model_args.task_learning_rate,
                                                                        total_num_inner_loop_steps=test_model_args.number_of_evaluation_steps_per_iter,
                                                                        use_learnable_learning_rates=test_model_args.learnable_per_layer_per_step_inner_loop_learning_rate,
                                                                        extrapolate_lr=test_model_args.extrapolate_lr)
        

        inner_lr_weights=None
        mm_weights = torch.load(test_args.model_path + "/mm_projector.bin", map_location='cpu')
        inner_lr_weights = {k:v for k,v in mm_weights.items() if "inner_loop_optimizer" in k}
        
        model.inner_loop_optimizer.initialise(
            names_weights_dict=get_inner_loop_parameter_dict(params=model.named_parameters()), pretrained_weights_dict=inner_lr_weights)
    
    model.to(torch.bfloat16)
    data_args.is_multimodal = True

    tokenizer.model_max_length = test_model_args.model_max_length
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = test_model_args.mm_use_im_start_end
    data_args.image_processor=image_processor
    
    meta_tune_dataset, meta_collator = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, model_version=test_args.version)
    dataloader = DataLoader(meta_tune_dataset, batch_size=1, collate_fn=meta_collator, num_workers=0)
    meta_tuner = MetaTuning(model, tokenizer, test_model_args)
    

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    elif "qwen" in model_name.lower():
        conv_mode = "qwen_2"
    else:
        conv_mode = "llava_v0"
    
    if test_args.conv_mode is not None and conv_mode != test_args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, test_args.conv_mode, test_args.conv_mode))
    else:
        test_args.conv_mode = conv_mode
    
    
    global_step=0
    corrects = [0] * (test_model_args.number_of_evaluation_steps_per_iter+1)
    n_exs = 0

    predictions_table_test = wandb.Table(columns=["Spt_image1", "Prompt_1", "Answer_1", "Spt_image2", "Prompt_2", "Answer_2", "Question_Image", "Question", "Prediction", "Answer", "Max Rating"])
    for s_idx, batch in enumerate(dataloader):
        
        if test_args.finetuning:
            model.train()
            supp_inp_set = dict(input_ids=batch["input_ids"],
                                labels=batch["labels"],
                                images=batch["images"],
                                attention_mask=batch["attention_mask"])
        
        log_dict = batch["log_dict"][0]
        ques_img = log_dict["image"]
        
        ques = log_dict["question"]
        answer = log_dict["answer"]
        
        supp_tasks = batch["logged_supp_tasks"][0]
        if test_args.finetuning:
            supp_losses, adapted_weight_list = meta_tuner.finetuning(supp_inp_set, use_second_order=test_model_args.second_order,
                                                num_eval_steps=test_model_args.number_of_evaluation_steps_per_iter,
                                                device=test_args.device, validation=False, grad_acc_part=test_model_args.grad_acc_part) 
            
            print(supp_losses) 

        model_params_dict = meta_tuner.get_inner_loop_parameter_dict(model.named_parameters())
        
        open_images = []
        if test_args.in_context:
            ict_supp_tasks = batch["ict_supp_tasks"][0]
            image_list = [ex["image"] for ex in ict_supp_tasks]
        
            for img in image_list:
                open_images.append(img)

        img_o = load_image(os.path.join(data_args.dataDir, ques_img[0]))
        open_images.append(img_o)
        image_tensor = process_images(open_images, image_processor, model.config)
        
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.bfloat16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.bfloat16)
        
        if data_args.dataset != "open_mi":
            spt1 = wandb.Image(supp_tasks[0]["image"])
            if data_args.k_spt > 1:
                spt2 = wandb.Image(supp_tasks[1]["image"])
        else:  
            spt1 = [d1 for d1 in supp_tasks if d1["conversations"][1]["value"] == answer][0]
            spt2 = random.choice([d1 for d1 in supp_tasks if d1["conversations"][1]["value"] != answer])
         
        ques_img_log = wandb.Image(Image.open(os.path.join(data_args.dataDir, ques_img[0])).convert("RGB"))
        
        n_exs += 1
        gen_state = torch.get_rng_state()
        torch.manual_seed(222 + s_idx)
        
        if test_args.finetuning:
            n_iters = len(adapted_weight_list)
        else:
            n_iters=1
        preds = []
        
        curr_scores = []
        for a_step in range(n_iters):
        
            inp = make_prompt(ques, data_args.add_test_prompt, query_task_instruction, data_args.dataset)            
            log_inp = inp
            
            ques_list= []
            if test_args.in_context:

                icl_exs = [icl_concat(ex["conversations"]) for ex in ict_supp_tasks]
                ques_list = copy.deepcopy(icl_exs)
            
            ques_list.append(inp)
            
            conv = conv_templates[test_args.conv_mode].copy()       
            
            fin_inp = ""
            for idx in range(len(ques_list)):
                if model.config.mm_use_im_start_end:
                    fin_inp = fin_inp  + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + ques_list[idx] + '\n'
                else:
                    fin_inp = fin_inp + DEFAULT_IMAGE_TOKEN + '\n ' + ques_list[idx] + '\n'
            
            fin_inp = fin_inp.rstrip("\n")

            conv.append_message(conv.roles[0], fin_inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
            model.eval()
          
            if test_args.finetuning:
                idx_to_layer_name_dict = {int(layer_name.split(".")[-1]): layer_name for layer_name in model_params_dict.keys()}
                usable_weights = adapted_weight_list[a_step]
            else:
                idx_to_layer_name_dict = {int(layer_name.split(".")[-1]): layer_name for layer_name in model_params_dict.keys()}
                usable_weights = model_params_dict
        
            with torch.inference_mode():
                output_ids = meta_tuner.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    temperature=test_args.temperature,
                    max_new_tokens=test_args.max_new_tokens,
                    min_new_tokens=1,
                    streamer=streamer,
                    use_cache=False,
                    stopping_criteria=[stopping_criteria],
                    fast_weights=usable_weights,
                    i2l_dict=idx_to_layer_name_dict)
   
            result = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip("").rstrip(".")
            fin_res = result
            
            preds.append(fin_res)
            curr_ex_score = exact_match(fin_res, answer, data_args.dataset)
            
            corrects[a_step]+=curr_ex_score
            curr_scores.append(curr_ex_score)
            
            wandb.log(
                {
                    f"Test/Max_Accuracy_step": max(corrects) / n_exs,
                },
                step=global_step,
                commit=False
            )
            wandb.log(
                {
                    f"Test/Accuracy_step{a_step}": corrects[a_step] / n_exs,
                },
                step=global_step,
                commit=False
            )
        
        max_acc_match = max(curr_scores)
        max_acc_idx = curr_scores.index(max_acc_match)

        if global_step % 1 == 0:
            if data_args.dataset == "open_mi":
                predictions_table_test.add_data(wandb.Image(spt1["image"]), spt1["conversations"][0]["value"], spt1["conversations"][1]["value"], wandb.Image(spt2["image"]), spt2["conversations"][0]["value"], spt2["conversations"][1]["value"], 
                                                ques_img_log, log_inp, preds[max_acc_idx], answer, 0)
            else:
                if data_args.k_spt > 1:
                    predictions_table_test.add_data(spt1, supp_tasks[0]["conversations"][0]["value"], supp_tasks[0]["conversations"][1]["value"], spt2, supp_tasks[1]["conversations"][0]["value"], supp_tasks[1]["conversations"][1]["value"], 
                                                    ques_img_log, log_inp, preds[max_acc_idx], answer, 0)
                else:
                    predictions_table_test.add_data(spt1, supp_tasks[0]["conversations"][0]["value"], supp_tasks[0]["conversations"][1]["value"], spt1, supp_tasks[0]["conversations"][0]["value"], supp_tasks[0]["conversations"][1]["value"], 
                                                    ques_img_log, log_inp, preds[max_acc_idx], answer, 0)
            
            new_table1 = wandb.Table(columns=predictions_table_test.columns, data=predictions_table_test.data)
            wandb.log({"Test/Predictions": new_table1}, step=global_step, commit=False)

        wandb.log({}, step=global_step, commit=True)

        global_step+=1
        torch.set_rng_state(gen_state)

    print("Max. Accuracy Score: ", max(corrects) / n_exs)

if __name__ == "__main__":
    main()