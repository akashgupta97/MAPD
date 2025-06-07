import os
import copy
import json
import math
import random
import torch
import transformers
import tokenizers

from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from dataclasses import dataclass, field
from torch.utils.data import Dataset
from typing import Dict, Optional, Sequence, List

from llava.model import *
from llava import conversation as conversation_lib
from llava.mm_utils import tokenizer_image_token

from llava.eval.ICL_utils import select_demonstration
from llava.eval.ICL_utils import get_task_instruction, format_answer

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_aspect_ratio: str = 'square'
    k_spt: int =  field(default=None,
                        metadata={"help":"Number of support examples"})
    k_qry: int =  field(default=None,
                        metadata={"help":"Number of query examples"})
    img_size: int =  field(default=None,
                        metadata={"help":"Input image size for resizing if necessary"})
    data_mode: str = field(default=None,
                           metadata={"help": "Define mode for data loading"})
    image_folder: str = field(default=None, metadata={"help": "Path to images/data json"})
    train_datasets: List[str] = field(default_factory=list)
    val_datasets: List[str] = field(default_factory=list)
    remove_instruct: bool = False
    mix_no_shot: bool = True


@dataclass
class TestDataArguments:
    dataDir: str = field(default=None,
                         metadata={"help": "Dataset directory"})
    dataset: str = field(default="open_mi",
                         metadata={"help": "Chosen dataset"})
    n_way: int = field(default=2,
                       metadata={"help": "Number of ways/class in a meta-task"})
    k_spt: int = field(default=1,
                       metadata={"help": "Number of shots per class in a meta-task"})
    
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_aspect_ratio: str = 'square'
    img_size: int =  field(default=None,
                        metadata={"help":"Input image size for resizing if necessary"})
    image_folder: str = field(default=None, metadata={"help": "Path to images/data json"})
    add_fin_prompt: bool = True
    add_test_prompt: bool  = True
    task_description: Optional[str] = field(default="nothing")
    
def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_qwen2(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    
    roles = {"human": "user", "gpt": "assistant"}

    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    im_start, im_end = tokenizer.additional_special_tokens_ids[:2]
    
    unmask_tokens_idx =  [198, im_start, im_end]
    
    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
                   
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    
    if len(input_ids) > 1:
        input_ids = [torch.tensor(inp_id, dtype=torch.long) for inp_id in input_ids]
        targets = [torch.tensor(tgt, dtype=torch.long) for tgt in targets]
    else:
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)
    
    return dict(
        input_ids=input_ids,
        labels=targets, 
    )

def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    
    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = copy.deepcopy(input_ids)

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]   # tokenizing and putting in BOS token
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX
   
    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """

    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("qwen_2"):
        return preprocess_qwen2(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, mode, tokenizer, data_args):
        super(SupervisedDataset, self).__init__()
                
        self.data_args = data_args
        self.data_path =  data_args.data_path
        self.image_folder =  data_args.image_folder
        self.train_datasets = data_args.train_datasets
        self.val_datasets = data_args.val_datasets
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.mode = mode
        self.conv_data = []
        datasets_list = []

        if mode == "val":
            for ds in self.val_datasets:
                with open(os.path.join(self.data_path, self.image_folder, ds, f"{ds}_dataset_val.json")) as json_file:
                    json_data = json.load(json_file)
                copy_json_data = copy.deepcopy(json_data)
                if self.data_args.remove_instruct:
                    copy_json_data = [self.remove_ins(ex, ds) for ex in copy_json_data]
                self.conv_data.extend(copy_json_data)
                temp_list = [ds] * len(copy_json_data)
                datasets_list.extend(temp_list)
        else:
            for ds in self.train_datasets:
                with open(os.path.join(self.data_path, self.image_folder, ds, f"{ds}_dataset.json")) as json_file:
                    json_data = json.load(json_file)
                copy_json_data = copy.deepcopy(json_data)
                if self.data_args.remove_instruct:
                    copy_json_data = [self.remove_ins(ex, ds) for ex in copy_json_data]    
                self.conv_data.extend(copy_json_data)  
                temp_list = [ds] * len(copy_json_data)
                datasets_list.extend(temp_list)
                 
        zipped_batch = list(zip(self.conv_data, datasets_list))
        random.shuffle(zipped_batch)
        self.shuffled_conv_data, self.shuffled_datasets_list = zip(*zipped_batch)
        
        self.shuffled_conv_data = list(self.shuffled_conv_data)
        self.shuffled_datasets_list = list(self.shuffled_datasets_list)
        self.dataset_indices = defaultdict(list)
        
        for d_idx, d_name in enumerate(self.shuffled_datasets_list):
            self.dataset_indices[d_name].append(d_idx)

    def remove_ins(self, conv_ex, ds):
        trim_value_list = conv_ex["conversations"][0]["value"].split("\n")
        if ds in ["basic_qa_geo170k", "mavis_math_metagen", "reasoning_qa_geo170k"]:
            conv_ex["conversations"][0]["value"] = "\n".join(trim_value_list[:1] + trim_value_list[2:])
        elif len(trim_value_list) > 2:
            conv_ex["conversations"][0]["value"] = "\n".join(trim_value_list[:-1])
        return conv_ex
    
    @property
    def lengths(self):
        length_list = []
        for sample in self.conv_data:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.conv_data:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __len__(self):
        return len(self.conv_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        
        src_dict = [self.conv_data[i]]
    
        if "image" in src_dict[0]:
            img_id = src_dict[0]['image']
            processor = self.data_args.image_processor
            image_folder = self.data_args.image_folder
                
            image = Image.open(os.path.join(self.data_path, image_folder, img_id)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            
            src_mul = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in src_dict]),  # Making sure that the DEFAULT_IMAGE_TOKEN is present in sources
                self.data_args)
        else:
            src_mul = copy.deepcopy([e["conversations"] for e in src_dict])

        data_dict = preprocess(
            src_mul,
            self.tokenizer,
            has_image=('image' in src_dict[0]))
    
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        
        if 'image' in src_dict[0]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        
        if self.tokenizer.pad_token_id is None:
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))
        
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
    
        return batch

class MultiTaskDataset(Dataset):
    """Dataset for multi-task fine-tuning."""

    def __init__(self, mode, data_args, local_rank, tokenizer=None, debug_log=False):
        super(MultiTaskDataset, self).__init__()
        self.debug_log=debug_log
        self.data_args = data_args
        self.k_shot = data_args.k_spt  # No. of examples in support 
        self.k_query = data_args.k_qry  # No. of examples in support
        
        self.img_size = data_args.img_size
        self.number_of_tasks = 0
        self.tokenizer = tokenizer
        self.mode = mode
        self.train_datasets = data_args.train_datasets
        self.val_datasets = data_args.val_datasets
        self.data_path = data_args.data_path
        self.image_folder= data_args.image_folder
        self.meta_datasets = []
        self.no_cluster_ds_list = []
        self.local_rank=local_rank
        
        if self.mode == "val":
            self.create_meta_dataset(self.val_datasets, local_rank, val=True)   
        else:
            self.create_meta_dataset(self.train_datasets, local_rank, val=False)

    def remove_ins(self, conv_ex, ds):
        trim_value_list = conv_ex["conversations"][0]["value"].split("\n")
        if ds in ["basic_qa_geo170k", "mavis_math_metagen", "reasoning_qa_geo170k"]:
            conv_ex["conversations"][0]["value"] = "\n".join(trim_value_list[:1] + trim_value_list[2:])
        elif len(trim_value_list) > 2:
            conv_ex["conversations"][0]["value"] = "\n".join(trim_value_list[:-1])
        return conv_ex
    
    def open_datasets(self, dataset, val):
        if val:
            with open(os.path.join(self.data_path, self.image_folder, dataset, f"{dataset}_dataset_val.json"), "r") as json_file:
                conv_data = json.load(json_file)
                count=0
                for a in conv_data:   
                    if self.data_args.remove_instruct:                 
                        self.remove_ins(a, dataset)     
                    if "id" in a:
                        del a["id"]
                    a["s_id"] = count
                    count+=1
        else:
            with open(os.path.join(self.data_path, self.image_folder, dataset, f"{dataset}_dataset.json"), "r") as json_file:
                conv_data = json.load(json_file)
                count=0
                for a in conv_data:   
                    if self.data_args.remove_instruct:                 
                        self.remove_ins(a, dataset)     
                    if "id" in a:
                        del a["id"]
                    a["s_id"] = count
                    count+=1
        return conv_data

    def create_meta_dataset(self, sample_datasets, local_rank, val):

        for dataset in sample_datasets:
            self.meta_datasets.append(dataset)
            self.no_cluster_ds_list.append(self.open_datasets(dataset, val))
        
        self.task_creation(local_rank)

    def task_creation(self, local_rank):
        self.total_tasks = 0
        self.tasks_batch = []
        self.dataset_list = []    
        
        for idx in range(len(self.no_cluster_ds_list)):
            if local_rank == 0:
                    print(f"Creating meta-tasks for {self.meta_datasets[idx].upper()} dataset........")
            self.create_batch_nocluster(local_rank, self.meta_datasets[idx], self.no_cluster_ds_list[idx])
    
        if local_rank == 0:
            print(f"TOTAL META-TASKS CREATED: {self.total_tasks}")
        zipped_batch = list(zip(self.tasks_batch, self.dataset_list))
        random.shuffle(zipped_batch)
        self.shuffled_tasks_batch, self.shuffled_dataset_list = zip(*zipped_batch)
        
        self.shuffled_tasks_batch = list(self.shuffled_tasks_batch)
        
        self.shuffled_dataset_list = list(self.shuffled_dataset_list)
        self.dataset_indices = defaultdict(list)
        
        if self.number_of_tasks == 0:
            self.number_of_tasks = copy.deepcopy(self.total_tasks)
        else:
            if self.number_of_tasks <= self.total_tasks:
                if self.local_rank == 0:
                    print(f"Truncating meta-tasks to {self.number_of_tasks}........")
                self.shuffled_tasks_batch = self.shuffled_tasks_batch[:self.number_of_tasks]
                self.shuffled_dataset_list = self.shuffled_dataset_list[:self.number_of_tasks]
            else:
                if self.local_rank == 0:
                    print(f"Padding meta-tasks to {self.number_of_tasks}........")
                appending_idxs = random.sample(range(len(self.shuffled_tasks_batch)), (self.number_of_tasks - self.total_tasks))
                sampled_tasks_list = [self.shuffled_tasks_batch[i] for i in appending_idxs]
                sampled_dataset_list = [self.shuffled_dataset_list[i] for i in appending_idxs]

                self.shuffled_tasks_batch.extend(sampled_tasks_list)
                self.shuffled_dataset_list.extend(sampled_dataset_list)

        for d_idx, d_name in enumerate(self.shuffled_dataset_list):
           self.dataset_indices[d_name].append(d_idx)


    def create_batch_nocluster(self, local_rank, dataset, ds_conv_data):
        n_tasks = 0
        edit_conv_data = copy.deepcopy(ds_conv_data)
        total_num_exs = len(ds_conv_data)
        global_count = 0
        K = self.k_shot + self.k_query
        if local_rank == 0:
            pbar = tqdm(total=total_num_exs, desc="Progress", unit="step")
        while global_count < total_num_exs:
            
            if len(edit_conv_data) < K:
            
                residual = K - len(edit_conv_data)
                rem_seen_ids = {elem['s_id'] for elem in edit_conv_data}
                sliced_conv_data = [x for x in ds_conv_data if x["s_id"] not in rem_seen_ids]
                extra_conv_data = random.sample(sliced_conv_data, residual)
                edit_conv_data += extra_conv_data
                sampled_elements = random.sample(edit_conv_data, K)
                if local_rank==0: pbar.update(residual)
                global_count+=residual
            
            else:
            
                sampled_elements = random.sample(edit_conv_data, K)  
                seen_ids = {elem['s_id'] for elem in sampled_elements}
                edit_conv_data = [x for x in edit_conv_data if x["s_id"] not in seen_ids]
                if local_rank==0: pbar.update(K)
                global_count+=K
            
            fin_llava_cls = copy.deepcopy(sampled_elements)   
            self.tasks_batch.append(fin_llava_cls)
            self.dataset_list.append(dataset)
            n_tasks+=1
            
        if local_rank==0:
            print("No. of meta-tasks created - ", n_tasks)
            self.total_tasks += n_tasks

    def __len__(self):
        return len(self.shuffled_tasks_batch)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        
        src_tasks = self.shuffled_tasks_batch[i]
        
        if "image" in src_tasks[0]:
            image_files = [conv_dict["image"] for conv_dict in src_tasks]
            processor = self.data_args.image_processor

            image_folder = self.data_args.image_folder
                
            images = [Image.open(os.path.join(self.data_args.data_path, image_folder, img_id)).convert('RGB')
                        for img_id in image_files]

            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                
                exp_images = [expand2square(s, tuple(int(x*255) for x in processor.image_mean)) for s in images]      
                proc_images = [processor.preprocess(s, return_tensors='pt')['pixel_values'][0] for s in exp_images]      
            else:
                proc_images = [processor.preprocess(s, return_tensors='pt')['pixel_values'][0] for s in images]

            src_mul = preprocess_multimodal(
                    copy.deepcopy([cv["conversations"] for cv in src_tasks]),
                    self.data_args)

        else:
            src_mul = copy.deepcopy([cv["conversations"] for cv in src_tasks])


        data_dict = preprocess(
                src_mul,
                self.tokenizer,
                has_image=('image' in src_tasks[0]))
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"],
                            labels=data_dict["labels"])

        if "image" in src_tasks[0]:
            data_dict['images'] = proc_images
        return data_dict

@dataclass
class DataCollatorForMultiTaskDataset(object):
    """Collate examples for multi-task fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        input_ids, labels, \
        images = tuple([instance[key] for instance in instances]
                                for key in ("input_ids", "labels",
                                            "images"))

        input_ids_flatten = [ips for bl in input_ids for ips in bl]
        labels_flatten = [lbs for bl in labels for lbs in bl]
        images_flatten = [img for bl in images for img in bl]

        if self.tokenizer.pad_token_id is None:
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids_flatten,
                                                batch_first=True,
                                                padding_value=self.tokenizer.pad_token_id)                    
        
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels_flatten,
                                                    batch_first=True,
                                                    padding_value=IGNORE_INDEX)
        
        
        input_ids_padded = input_ids_padded[:, :self.tokenizer.model_max_length]
        labels_padded = labels_padded[:, :self.tokenizer.model_max_length]
        
        batch = dict(
            input_ids=input_ids_padded,
            labels=labels_padded,
            attention_mask=input_ids_padded.ne(self.tokenizer.pad_token_id),
        )
        batch['images'] = torch.stack(images_flatten)

        return batch

class ICTDataset(Dataset):
    """Dataset for in-context finetuning"""

    def __init__(self, mode, tokenizer, data_args, local_rank):
        super(ICTDataset, self).__init__()
                
        #rank0_print("Formatting inputs...Skip in lazy mode")
        self.data_args = data_args
        self.data_path =  data_args.data_path
        self.k_shot = data_args.k_spt
        self.image_folder =  data_args.image_folder
        self.train_datasets = data_args.train_datasets # Assuming single dataset support for now
        self.val_datasets = data_args.val_datasets
        self.tokenizer = tokenizer
        self.data_args = data_args
        
        #Adding dummy token for easy partition
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<boc>", "<eoc>"]})
        self.eff_con_len = tokenizer.model_max_length - ((data_args.k_spt + 1) * data_args.prefix_length)
        self.mode = mode
        self.ict_datasets = []
        self.ict_ds_list = []
        self.number_of_tasks = 0
        self.local_rank=local_rank
        self.nz_shot = 0
        if mode == "val":
            self.create_ict_dataset(self.val_datasets, local_rank, val=True)
        else:
            self.create_ict_dataset(self.train_datasets, local_rank, val=False)
    
    def remove_ins(self, conv_ex, ds):
        trim_value_list = conv_ex["conversations"][0]["value"].split("\n")
        if ds in ["basic_qa_geo170k", "mavis_math_metagen", "reasoning_qa_geo170k"]:
            conv_ex["conversations"][0]["value"] = "\n".join(trim_value_list[:1] + trim_value_list[2:])
        elif len(trim_value_list) > 2:
            conv_ex["conversations"][0]["value"] = "\n".join(trim_value_list[:-1])
        return conv_ex
        
    def open_datasets(self, dataset, val):
        if val:
            with open(os.path.join(self.data_path, self.image_folder, dataset, f"{dataset}_dataset_val.json"), "r") as json_file:
                conv_data = json.load(json_file)
                count=0
                for a in conv_data:   
                    if self.data_args.remove_instruct:                 
                        self.remove_ins(a, dataset)     
                    if "id" in a:
                        del a["id"]
                    a["s_id"] = count
                    count+=1
        else:
            with open(os.path.join(self.data_path, self.image_folder, dataset, f"{dataset}_dataset.json"), "r") as json_file:
                conv_data = json.load(json_file)
                count=0
                for a in conv_data:   
                    if self.data_args.remove_instruct:                 
                        self.remove_ins(a, dataset)     
                    if "id" in a:
                        del a["id"]
                    a["s_id"] = count
                    count+=1
        return conv_data
    
    def create_ict_dataset(self, sample_datasets, local_rank, val):
        for dataset in sample_datasets:
                self.ict_datasets.append(dataset)
                self.ict_ds_list.append(self.open_datasets(dataset, val))
        
        self.ict_ex_creation(local_rank)

    def ict_ex_creation(self, local_rank):
        self.total_tasks = 0
        self.ict_batch = []
        self.dataset_list = []

        for idx in range(len(self.ict_ds_list)):
            if local_rank == 0:
                print(f"Creating meta-tasks for {self.ict_datasets[idx].upper()} dataset........")
            self.create_batch_ict(local_rank, self.ict_datasets[idx], self.ict_ds_list[idx])

        if local_rank == 0:
            print(f"TOTAL META-TASKS CREATED: {self.total_tasks}")
        zipped_batch = list(zip(self.ict_batch, self.dataset_list))
        random.shuffle(zipped_batch)
        self.shuffled_ict_batch, self.shuffled_dataset_list = zip(*zipped_batch)
    
        self.shuffled_ict_batch = list(self.shuffled_ict_batch)
        self.shuffled_dataset_list = list(self.shuffled_dataset_list)
        self.dataset_indices = defaultdict(list)

        if self.number_of_tasks == 0:
            self.number_of_tasks = copy.deepcopy(self.total_tasks)
        else:
            if self.number_of_tasks <= self.total_tasks:
                if self.local_rank == 0:
                    print(f"Truncating meta-tasks to {self.number_of_tasks}........")
                self.shuffled_ict_batch = self.shuffled_ict_batch[:self.number_of_tasks]
                self.shuffled_dataset_list = self.shuffled_dataset_list[:self.number_of_tasks]
            else:
                if self.local_rank == 0:
                    print(f"Padding meta-tasks to {self.number_of_tasks}........")
                appending_idxs = random.sample(range(len(self.shuffled_ict_batch)), (self.number_of_tasks - self.total_tasks))
                sampled_supp_list = [self.shuffled_ict_batch[i] for i in appending_idxs]
                sampled_dataset_list = [self.shuffled_dataset_list[i] for i in appending_idxs]

                self.shuffled_ict_batch.extend(sampled_supp_list)
                self.shuffled_dataset_list.extend(sampled_dataset_list)

        for d_idx, d_name in enumerate(self.shuffled_dataset_list):
           self.dataset_indices[d_name].append(d_idx)
        
    
    def create_batch_ict(self, local_rank, dataset, ds_conv_data):
        n_tasks = 0
        edit_conv_data = copy.deepcopy(ds_conv_data)
        total_num_exs = len(ds_conv_data)
        global_count = 0
        inner_batch = 30
        K = (self.k_shot+1)*inner_batch   # inner_batch for faster creation of ICT tasks
        if local_rank == 0:
            pbar = tqdm(total=total_num_exs, desc="Progress", unit="step")
        while global_count < total_num_exs:
            if len(edit_conv_data) < K:
                
                residual = K - len(edit_conv_data)
                rem_seen_ids = {elem['s_id'] for elem in edit_conv_data}
                sliced_conv_data = [x for x in ds_conv_data if x["s_id"] not in rem_seen_ids]
                extra_conv_data = random.sample(sliced_conv_data, residual)
                edit_conv_data += extra_conv_data
                sampled_elements = random.sample(edit_conv_data, K)
                if local_rank==0: pbar.update(residual)
                global_count+=residual
            
            else:

                sampled_elements = random.sample(edit_conv_data, K)   
                seen_ids = {elem['s_id'] for elem in sampled_elements}
                edit_conv_data = [x for x in edit_conv_data if x["s_id"] not in seen_ids]
                if local_rank==0: pbar.update(K)
                global_count+=K
            
            fin_llava_cls = copy.deepcopy(sampled_elements)    
            fin_llava_cls_subs = [fin_llava_cls[i:i+(self.k_shot+1)] for i in range(0, len(fin_llava_cls), (self.k_shot+1))]
            
            for subl in fin_llava_cls_subs:
                fin_dict = {}
                has_img_ind = False
                if "image" in subl[0]:
                    fin_dict["image"] = [d1["image"] for d1 in subl]
                    has_img_ind = True
                fin_dict["conversations"] = self.ict_concat(copy.deepcopy([d1["conversations"] for d1 in subl]), has_img_ind)
                         
                self.ict_batch.append(fin_dict)
                self.dataset_list.append(dataset)
                n_tasks+=1
           
        if local_rank==0:
            print("No. of tasks created - ", n_tasks)
            self.total_tasks += n_tasks

    def ict_concat(self, convos, has_img_ind):
        if has_img_ind:
            mod_convos = preprocess_multimodal(convos, self.data_args)
        else:
            mod_convos = convos

        temp_ict_dict = {}
        temp_ict_dict["from"] = "human"
        ict_dicts = [d1 for sub in mod_convos[:-1] for d1 in sub]
        
        if has_img_ind:
            temp_ict_dict["value"]= " ".join(
                            ("Answer: " + d["value"]) if d["from"] == "gpt" else ("\n" + d["value"])
                            for d in ict_dicts
                            )[1:]
        else:

            temp_ict_dict["value"]= "<boc>" + " ".join(
                            ("Answer: " + d["value"]) if d["from"] == "gpt" else ("\n" + d["value"])
                            for d in ict_dicts
                            )[1:] + "<eoc>"
            
        query_dict = mod_convos[-1]
        temp_ict_dict["value"] += "\n" + query_dict[0]["value"]
        concat_convs = [temp_ict_dict]
        concat_convs.extend(query_dict[1:])
        
        return concat_convs        
    
    def process_image(self, processor, image_folder, img_id):
        image = Image.open(os.path.join(self.data_path, image_folder, img_id)).convert('RGB')
        if self.data_args.image_aspect_ratio == 'pad':
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        return image

    def weighted_reduce_int(self, text_tokens, cap, alpha=-0.5):
        n1 = len(text_tokens)
        if any(x <= 0 for x in text_tokens):
            raise ValueError("All numbers must be positive.")
        if cap < n1:
            raise ValueError("Cap is too small: each number must be at least 1.")
          
        # Compute weights using the transformation function
        weights = [x ** alpha for x in text_tokens]
        total_weight = sum(weights)
        
        # Compute the ideal (floating point) allocation for each number
        float_allocations = [min(text_tokens[idx], cap * weights[idx] / total_weight) for idx in range(len(weights))] 
        trimmed_allocations = [float_allocations[idx] for idx in range(len(float_allocations)) if float_allocations[idx] != text_tokens[idx]]
        trimmed_text_tokens = [text_tokens[idx] for idx in range(len(float_allocations)) if float_allocations[idx] != text_tokens[idx]]
        trimmed_idxs = [idx for idx in range(len(float_allocations)) if float_allocations[idx] != text_tokens[idx]]
        
        # Convert to integer allocations by flooring (ensuring a minimum of 1)
        int_allocations = [max(1, int(math.floor(val))) for val in float_allocations]
        allocated = sum(int_allocations)

        # Distribute the remaining units using the largest remainder method
        remainders = [trimmed_text_tokens[idx] - trimmed_allocations[idx] for idx in range(len(trimmed_allocations))]
        diff = cap - allocated
        
        indices_sorted = [a for b, a in sorted(zip(remainders, trimmed_idxs))]
        
        while diff > 0:
            for i in indices_sorted:
                if diff <= 0:
                    break
                # Check if there's room to add more tokens
                if int_allocations[i] < text_tokens[i]:
                    int_allocations[i] += 1
                    diff -= 1    

        # Final check
        if sum(int_allocations) != cap:
            raise RuntimeError("Allocation error: sum of allocations does not equal cap.")     
        return int_allocations

    def __len__(self):
        return len(self.shuffled_ict_batch)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        
        src_dict = [self.shuffled_ict_batch[i]]
        if "image" in src_dict[0]:
            img_id = src_dict[0]['image']
            processor = self.data_args.image_processor
            image_folder = self.data_args.image_folder
            if type(img_id) is list:
                image = [self.process_image(processor, image_folder, f) for f in img_id]
                assert all(x is not None and x.shape == image[0].shape for x in image)
            else:
                image = self.process_image(processor, image_folder, img_id)
        
        src_mul = copy.deepcopy([e["conversations"] for e in src_dict])
    
        data_dict = preprocess(
            src_mul,
            self.tokenizer,
            has_image=('image' in src_dict[0]))
        
        if "image" in src_dict[0]:
 
            img_token_idxs = torch.where(data_dict["input_ids"][0] == IMAGE_TOKEN_INDEX)[0].tolist()
            text_token_lens = []
            for i in range(len(img_token_idxs)-1): 
                text_token_lens.append(img_token_idxs[i+1] - img_token_idxs[i]-1)
            input_len = data_dict["input_ids"][0].shape[0]
            input_sub_query = input_len - img_token_idxs[-1] -1

            some_pad = 5
            cap_number = self.eff_con_len - img_token_idxs[0] - (input_sub_query) - some_pad
            if sum(text_token_lens) > cap_number:
                if cap_number <= 100:
                    if self.data_args.mix_no_shot:
                        # WARNING: Mixes 0-shot examples in training as limited by context length.
                        
                        self.nz_shot+=1
                        image = [image[-1]]
                        data_dict["input_ids"][0] = torch.cat([data_dict["input_ids"][0][:img_token_idxs[0]], data_dict["input_ids"][0][img_token_idxs[-1]:]], dim=0)
                        data_dict["labels"][0] = torch.cat([data_dict["labels"][0][:img_token_idxs[0]], data_dict["labels"][0][img_token_idxs[-1]:]], dim=0)
                    else:
                        raise ValueError("Query too long for current context length")
                
                else:
                    mod_text_token_lens = self.weighted_reduce_int(text_token_lens, cap_number)
                    mod_text_token_lens = [img_token_idxs[0]-1] + mod_text_token_lens + [input_sub_query]
                    img_token_idxs = [0] + img_token_idxs
                    inp_id_chunks = [data_dict["input_ids"][0][start : start + length + 1] for start, length in zip(img_token_idxs, mod_text_token_lens)]
                    lab_chunks = [data_dict["labels"][0][start : start + length + 1] for start, length in zip(img_token_idxs, mod_text_token_lens)]
                    data_dict["input_ids"][0] = torch.cat(inp_id_chunks)
                    data_dict["labels"][0] = torch.cat(lab_chunks)
                    trimmed_img_token_idxs =  torch.where(data_dict["input_ids"][0][:self.tokenizer.model_max_length] == IMAGE_TOKEN_INDEX)[0].tolist() 
                    assert len(trimmed_img_token_idxs) == (self.k_shot + 1)
        
        else:
            no_img_eff_con_len = self.eff_con_len + ((self.data_args.k_spt + 1) * self.data_args.prefix_length)
            input_len = data_dict["input_ids"][0].shape[0]
            st_token_idx = torch.where(data_dict["input_ids"][0] == self.tokenizer.convert_tokens_to_ids("<boc>"))[0].tolist()[0]
            ed_token_idx = torch.where(data_dict["input_ids"][0] == self.tokenizer.convert_tokens_to_ids("<eoc>"))[0].tolist()[0]
            inp_sub_query = input_len - (ed_token_idx + 1)
            some_pad = 5
            cap_number = no_img_eff_con_len - inp_sub_query - some_pad - st_token_idx
            ict_text_tokens = ed_token_idx - st_token_idx - 1
            
            if ict_text_tokens > cap_number:
                fin_slice_len = cap_number
            else:
                fin_slice_len = ed_token_idx

            data_dict["input_ids"] = torch.cat([data_dict["input_ids"][0][:st_token_idx], data_dict["input_ids"][0][st_token_idx+1:fin_slice_len], data_dict["input_ids"][0][ed_token_idx+1:]]).unsqueeze(0)
            data_dict["labels"] = torch.cat([data_dict["labels"][0][:st_token_idx], data_dict["labels"][0][st_token_idx+1:fin_slice_len], data_dict["labels"][0][ed_token_idx+1:]]).unsqueeze(0)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        
        if 'image' in src_dict[0]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])

        return data_dict

@dataclass
class DataCollatorForICTDataset(object):
    """Collate examples for in-context fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        
        if self.tokenizer.pad_token_id is None:
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))
        
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if type(instances[0]["image"]) is list: 
                images = [im for im_list in images for im in im_list]

            batch['images'] = torch.stack(images)
           
        return batch

class MAPDDataset(Dataset):
    """
    Dataset for MAPD
    """

    def __init__(self, mode, data_args, local_rank, tokenizer=None, debug_log=False):
        super(MAPDDataset, self).__init__()
        self.debug_log=debug_log
        self.data_args = data_args
        self.k_shot = data_args.k_spt  # No. of support examples
        self.k_query = data_args.k_qry  # No. of query examples
        
        self.img_size = data_args.img_size  # resize to
        self.number_of_tasks = 0
 
        self.tokenizer = tokenizer
        self.mode = mode
        self.train_datasets = data_args.train_datasets
        self.val_datasets = data_args.val_datasets
        self.data_path = data_args.data_path
        self.image_folder= data_args.image_folder
        self.meta_datasets = []
        self.no_cluster_ds_list = []
        self.local_rank=local_rank
        
        if self.mode == "val":
            self.create_meta_dataset(self.val_datasets, local_rank, val=True)   
        else:
            self.create_meta_dataset(self.train_datasets, local_rank, val=False)

    def remove_ins(self, conv_ex, ds):
        trim_value_list = conv_ex["conversations"][0]["value"].split("\n")
        if ds in ["basic_qa_geo170k", "mavis_math_metagen", "reasoning_qa_geo170k"]:
            conv_ex["conversations"][0]["value"] = "\n".join(trim_value_list[:1] + trim_value_list[2:])
        elif len(trim_value_list) > 2:
            conv_ex["conversations"][0]["value"] = "\n".join(trim_value_list[:-1])
        return conv_ex

    def open_datasets(self, dataset, val):
        if val:
            with open(os.path.join(self.data_path, self.image_folder, dataset, f"{dataset}_dataset_val.json"), "r") as json_file:
                conv_data = json.load(json_file)
                count=0
                for a in conv_data:   
                    if self.data_args.remove_instruct:                 
                        self.remove_ins(a, dataset)     
                    if "id" in a:
                        del a["id"]
                    a["s_id"] = count
                    count+=1
        else:
            with open(os.path.join(self.data_path, self.image_folder, dataset, f"{dataset}_dataset.json"), "r") as json_file:
                conv_data = json.load(json_file)
                count=0
                for a in conv_data:   
                    if self.data_args.remove_instruct:                 
                        self.remove_ins(a, dataset)     
                    if "id" in a:
                        del a["id"]
                    a["s_id"] = count
                    count+=1
        return conv_data

    def create_meta_dataset(self, sample_datasets, local_rank, val):

        for dataset in sample_datasets:
            self.meta_datasets.append(dataset)
            self.no_cluster_ds_list.append(self.open_datasets(dataset, val))
              
        self.task_creation(local_rank)
        

    def task_creation(self, local_rank):
        self.total_tasks = 0
        self.support_x_batch = []
        self.query_x_batch = []
        self.dataset_list = []
    
        for idx in range(len(self.no_cluster_ds_list)):
            if local_rank == 0:
                    print(f"Creating meta-tasks for {self.meta_datasets[idx].upper()} dataset........")
            self.create_batch_nocluster(local_rank, self.meta_datasets[idx], self.no_cluster_ds_list[idx])
        
        if local_rank == 0:
            print(f"TOTAL META-TASKS CREATED: {self.total_tasks}")
        zipped_batch = list(zip(self.support_x_batch, self.query_x_batch, self.dataset_list))
        random.shuffle(zipped_batch)
        self.shuffled_support_x_batch, self.shuffled_query_x_batch, self.shuffled_dataset_list = zip(*zipped_batch)
        
        self.shuffled_support_x_batch = list(self.shuffled_support_x_batch)
        self.shuffled_query_x_batch = list(self.shuffled_query_x_batch)
        self.shuffled_dataset_list = list(self.shuffled_dataset_list)
        self.dataset_indices = defaultdict(list)
        
        if self.number_of_tasks == 0:
            self.number_of_tasks = copy.deepcopy(self.total_tasks)
        else:
            if self.number_of_tasks <= self.total_tasks:
                if self.local_rank == 0:
                    print(f"Truncating meta-tasks to {self.number_of_tasks}........")
                self.shuffled_support_x_batch = self.shuffled_support_x_batch[:self.number_of_tasks]
                self.shuffled_query_x_batch = self.shuffled_query_x_batch[:self.number_of_tasks]
                self.shuffled_dataset_list = self.shuffled_dataset_list[:self.number_of_tasks]
            else:
                if self.local_rank == 0:
                    print(f"Padding meta-tasks to {self.number_of_tasks}........")
                appending_idxs = random.sample(range(len(self.shuffled_support_x_batch)), (self.number_of_tasks - self.total_tasks))
                sampled_supp_list = [self.shuffled_support_x_batch[i] for i in appending_idxs]
                sampled_query_list = [self.shuffled_query_x_batch[i] for i in appending_idxs]
                sampled_dataset_list = [self.shuffled_dataset_list[i] for i in appending_idxs]

                self.shuffled_support_x_batch.extend(sampled_supp_list)
                self.shuffled_query_x_batch.extend(sampled_query_list)
                self.shuffled_dataset_list.extend(sampled_dataset_list)

        for d_idx, d_name in enumerate(self.shuffled_dataset_list):
           self.dataset_indices[d_name].append(d_idx)


    def create_batch_nocluster(self, local_rank, dataset, ds_conv_data):
        n_tasks = 0
        edit_conv_data = copy.deepcopy(ds_conv_data)
        total_num_exs = len(ds_conv_data)
        global_count = 0
        K = self.k_shot + self.k_query
        if local_rank == 0:
            pbar = tqdm(total=total_num_exs, desc="Progress", unit="step")
        while global_count < total_num_exs:
            
            if len(edit_conv_data) < K:
                residual = K - len(edit_conv_data)
                rem_seen_ids = {elem['s_id'] for elem in edit_conv_data}
                sliced_conv_data = [x for x in ds_conv_data if x["s_id"] not in rem_seen_ids]
                extra_conv_data = random.sample(sliced_conv_data, residual)
                edit_conv_data += extra_conv_data
                sampled_elements = random.sample(edit_conv_data, K)
                if local_rank==0: pbar.update(residual)
                global_count+=residual
            else:
                sampled_elements = random.sample(edit_conv_data, K)
                
                seen_ids = {elem['s_id'] for elem in sampled_elements}
                edit_conv_data = [x for x in edit_conv_data if x["s_id"] not in seen_ids]
                if local_rank==0: pbar.update(K)
                global_count+=K
            
            fin_llava_cls = copy.deepcopy(sampled_elements)
            temp_supp_x = fin_llava_cls[:self.k_shot]
            temp_query_x = fin_llava_cls[self.k_shot:]
            
            support_x = []
            query_x = []
                    
            support_x.append(temp_supp_x)
            query_x.append(temp_query_x)

            self.support_x_batch.append(support_x)
            self.query_x_batch.append(query_x)
            self.dataset_list.append(dataset)
            n_tasks+=1
            
        if local_rank==0:
            print("No. of tasks created - ", n_tasks)
            self.total_tasks += n_tasks
        

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
       
        supp_src_tasks = self.shuffled_support_x_batch[i]
        qry_src_tasks = self.shuffled_query_x_batch[i]
    
        processor = self.data_args.image_processor
        
        if "image" in self.shuffled_support_x_batch[i][0][0]:

            supp_image_files = [conv_dict["image"] for task_list in supp_src_tasks for conv_dict in task_list]
            qry_image_files = [conv_dict["image"] for task_list in qry_src_tasks for conv_dict in task_list]
            
            image_folder = self.data_args.image_folder
                
            supp_images = [Image.open(os.path.join(self.data_args.data_path, image_folder, img_id)).convert('RGB')
                        for img_id in supp_image_files]

            qry_images = [Image.open(os.path.join(self.data_args.data_path, image_folder, img_id)).convert('RGB')
                        for img_id in qry_image_files]
            
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                exp_supp_images = [expand2square(s, tuple(int(x*255) for x in processor.image_mean)) for s in supp_images]
                exp_qry_images = [expand2square(q, tuple(int(x*255) for x in processor.image_mean)) for q in qry_images]
                
                proc_supp_images = [processor.preprocess(s, return_tensors='pt')['pixel_values'][0] for s in exp_supp_images]
                proc_qry_images = [processor.preprocess(q, return_tensors='pt')['pixel_values'][0] for q in exp_qry_images]      
            else:
                proc_supp_images = [processor.preprocess(s, return_tensors='pt')['pixel_values'][0] for s in supp_images]
                proc_qry_images = [processor.preprocess(q, return_tensors='pt')['pixel_values'][0] for q in qry_images]
            
            
            supp_sources = [preprocess_multimodal(
                copy.deepcopy([cv["conversations"] for cv in task_list]),
                self.data_args) for task_list in supp_src_tasks]
            
            qry_sources = [preprocess_multimodal(
                copy.deepcopy([cv["conversations"] for cv in task_list]),
                self.data_args) for task_list in qry_src_tasks]
            
        else:
            supp_sources = [copy.deepcopy([cv["conversations"] for cv in task_list]) for task_list in supp_src_tasks]
            qry_sources = [copy.deepcopy([cv["conversations"] for cv in task_list]) for task_list in qry_src_tasks]
        
        supp_data_dict = [preprocess(
            task_list,
            self.tokenizer,
            has_image=('image' in supp_src_tasks[0][0])) for task_list in supp_sources]
        
        qry_data_dict = [preprocess(
            task_list,
            self.tokenizer,
            has_image=('image' in qry_src_tasks[0][0])) for task_list in qry_sources]
        
        supp_data_dict_inps = [inps for task_list in supp_data_dict for inps in task_list["input_ids"]]
        qry_data_dict_inps = [inps for task_list in qry_data_dict for inps in task_list["input_ids"]]

        supp_data_dict_labs = [labs for task_list in supp_data_dict for labs in task_list["labels"]]
        qry_data_dict_labs = [labs for task_list in qry_data_dict for labs in task_list["labels"]]

        if isinstance(i, int):
            data_dict = dict(supp_input_ids=supp_data_dict_inps,
                            supp_labels=supp_data_dict_labs,
                            qry_input_ids=qry_data_dict_inps,
                            qry_labels=qry_data_dict_labs)

        # image exist in the data
        if 'image' in supp_src_tasks[0][0]:
            data_dict['supp_images'] = torch.stack(proc_supp_images)
            data_dict['qry_images'] = torch.stack(proc_qry_images)
        else:
            # image does not exist in the data, but the model is multimodal
            crop_size = processor.crop_size     
            data_dict['supp_images'] = torch.zeros(len(supp_src_tasks[0]), 3, crop_size['height'], crop_size['width'])
            data_dict['qry_images'] = torch.zeros(len(qry_src_tasks[0]), 3, crop_size['height'], crop_size['width'])
        
        return data_dict

    def __len__(self):
        return len(self.shuffled_support_x_batch)
    
@dataclass
class DataCollatorForMAPDDataset(object):
    """Collate examples for MAPD."""
    tokenizer: Optional[transformers.PreTrainedTokenizer] = None

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        supp_input_ids, supp_labels, qry_input_ids, \
        qry_labels, supp_images, qry_images = tuple([instance[key] for instance in instances]
                                                    for key in ("supp_input_ids", "supp_labels",
                                                                "qry_input_ids", "qry_labels",
                                                                "supp_images", "qry_images"))

        if self.tokenizer.pad_token_id is None:
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.


        supp_input_ids_padded = [torch.nn.utils.rnn.pad_sequence(sublist,
                                                batch_first=True,
                                                padding_value=self.tokenizer.pad_token_id)
                                                for sublist in supp_input_ids]                       
        
        supp_labels_padded = [torch.nn.utils.rnn.pad_sequence(sublist,
                                                    batch_first=True,
                                                    padding_value=IGNORE_INDEX)
                                                    for sublist in supp_labels]
        
        qry_input_ids_padded = [torch.nn.utils.rnn.pad_sequence(sublist,
                                                batch_first=True,
                                                padding_value=self.tokenizer.pad_token_id)
                                                for sublist in qry_input_ids]                       
        
        qry_labels_padded = [torch.nn.utils.rnn.pad_sequence(sublist,
                                                    batch_first=True,
                                                    padding_value=IGNORE_INDEX)
                                                    for sublist in qry_labels]
        
        supp_input_ids_padded = [s[:, :self.tokenizer.model_max_length] for s in supp_input_ids_padded]
        supp_labels_padded = [s[:, :self.tokenizer.model_max_length] for s in supp_labels_padded]
        qry_input_ids_padded = [q[:, :self.tokenizer.model_max_length] for q in qry_input_ids_padded]
        qry_labels_padded = [q[:, :self.tokenizer.model_max_length] for q in qry_labels_padded]
        
        batch = dict(
            supp_input_ids=supp_input_ids_padded,
            supp_labels=supp_labels_padded,
            supp_attention_mask=[l.ne(self.tokenizer.pad_token_id) for l in supp_input_ids_padded],
            qry_input_ids=qry_input_ids_padded,
            qry_labels=qry_labels_padded,
            qry_attention_mask=[l.ne(self.tokenizer.pad_token_id) for l in qry_input_ids_padded],
        )
        batch['supp_images'] = supp_images
        batch['qry_images'] = qry_images
        
        return batch

class VLICLTestDataset(Dataset):
    def __init__(self, tokenizer, data_args, model_version):
        self.data_args = data_args
        self.img_size = data_args.img_size  # resize to
        self.tokenizer = tokenizer
        self.data_dir = data_args.dataDir  # image path
        self.chosen_ds = data_args.dataset
        self.k_shot = data_args.k_spt
        self.task_instruction = get_task_instruction(data_args)
        query_file = os.path.join(self.data_dir, data_args.dataset, 'query.json')
        support_file = os.path.join(self.data_dir, data_args.dataset, 'support.json')

        with open(query_file, 'r') as f:
            self.query_meta = json.load(f)
        with open(support_file, 'r') as f:
            self.support_meta = json.load(f)
        
        self.add_fin_prompt = data_args.add_fin_prompt
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_version]
    
    def load_image(self, img_ids, root_path):
        if isinstance(img_ids, str):
            img_ids = [img_ids]
        images = []
        image_paths = []
        for img_id in img_ids:
            image_path = os.path.join(root_path, img_id)
            image = Image.open(image_path).convert('RGB')
            images.append(image)
            image_paths.append(image_path)
        
        return images, image_paths

    def supp_ict_concat(self, convos, has_img_ind):
        
        if has_img_ind:
            mod_convos = preprocess_multimodal(convos, self.data_args)
        else:
            mod_convos = convos
        
        for subl in mod_convos[1:]:
            subl[0]["value"] = "\n" + subl[0]["value"]

        ict_dicts = [d1 for sub in mod_convos for d1 in sub]

        return ict_dicts        

    def __convert_llava(self, ques_list, ans_list, img_id_list):
        
        fin_img_conv_dicts = []
        
        for i in range(len(img_id_list)):
            temp_dict = {"image": "", "conversations": []}
            temp_conversation_dict_human = {"from": "human", "value": ""}
            temp_conversation_dict_gpt = {"from": "gpt", "value": ""}
            
            temp_dict["image"] = img_id_list[i]
            temp_conversation_dict_gpt["value"] = ans_list[i]
            if self.chosen_ds == "open_mi":
                temp_conversation_dict_human["value"] = self.make_prompt(f"<image>\n{ques_list[i]}?")
            else:
                temp_conversation_dict_human["value"] = self.make_prompt(f"<image>\n{ques_list[i]}")
            temp_conv_list = [temp_conversation_dict_human, temp_conversation_dict_gpt]
            temp_dict["conversations"] = temp_conv_list
            fin_img_conv_dicts.append(temp_dict)

        return fin_img_conv_dicts
    
    def make_prompt(self, prompt):
        if self.add_fin_prompt:
            return prompt + f" {self.task_instruction}"
        else:
            return prompt

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        
        n_shot_support = select_demonstration(self.support_meta, self.k_shot, self.chosen_ds, self.data_args.n_way, self.query_meta[i])
        
        images = []
        ques_list = []
        ans_list = []
        logged_images = []
        logged_ans = []
        logged_ques = []
        img_paths = []

        for k in range(len(n_shot_support)):
            for image_path in n_shot_support[k]['image']:
                
                img_paths.append(os.path.join(self.data_dir, image_path))
                images.append(Image.open(os.path.join(self.data_dir, image_path)).convert("RGB"))

                ques_list.append(n_shot_support[k]['question'])
                revised_ans = format_answer(n_shot_support[k]['answer'], self.chosen_ds, self.query_meta[i])
                ans_list.append(revised_ans)
                
                if self.chosen_ds == "open_mi":
                    if revised_ans not in logged_ans:
                        logged_ans.append(revised_ans)  
                        logged_images.append(Image.open(os.path.join(self.data_dir, image_path)).convert("RGB"))
                        logged_ques.append(n_shot_support[k]['question'])

 
        if self.chosen_ds != "open_mi":
            chosen_indices = random.sample(range(len(ans_list)), min(self.k_shot, 2))
            logged_img_paths = [img_paths[i] for i in chosen_indices]
            logged_ans = [ans_list[i] for i in chosen_indices]
            logged_images = [images[i] for i in chosen_indices]
            logged_ques = [ques_list[i] for i in chosen_indices]
            
        supp_src_tasks = self.__convert_llava(ques_list, ans_list, images)
                
        ict_supp_tasks = {}
        if self.data_args.in_context:
            ict_supp_tasks = copy.deepcopy(supp_src_tasks)
            
        if self.chosen_ds == "open_mi":
            logged_supp_tasks = self.__convert_llava(logged_ques, logged_ans, logged_images)
        else:
            logged_supp_tasks = self.__convert_llava(logged_ques, logged_ans, logged_img_paths)
        
        supp_images = [conv_dict["image"] for conv_dict in supp_src_tasks]
        processor = self.data_args.image_processor
        
        if self.data_args.image_aspect_ratio == 'pad':
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            exp_supp_images = expand2square(supp_images, tuple(int(x*255) for x in processor.image_mean))
            proc_supp_images = processor.preprocess(exp_supp_images, return_tensors='pt')['pixel_values']             
        else:
            proc_supp_images = processor.preprocess(supp_images, return_tensors='pt')['pixel_values']   
            
        supp_sources = preprocess_multimodal(
            copy.deepcopy([cv["conversations"] for cv in supp_src_tasks]),  # Making sure that the DEFAULT_IMAGE_TOKEN is present in sources
            self.data_args)

        supp_data_dict = preprocess(
            supp_sources,
            self.tokenizer,
            has_image=True)

        
        supp_data_dict_inps = supp_data_dict["input_ids"]
        supp_data_dict_labs = supp_data_dict["labels"]

        
        if isinstance(i, int):
            data_dict = dict(supp_input_ids=supp_data_dict_inps,
                             supp_labels=supp_data_dict_labs,
                             log_dict=self.query_meta[i],
                             logged_supp_tasks=logged_supp_tasks,
                             ict_supp_tasks=ict_supp_tasks,
                             )

        # image exist in the data
        data_dict['supp_images'] = proc_supp_images
        
        return data_dict

    def __len__(self):
        return len(self.query_meta)
    
@dataclass
class DataCollatorForVLICLDataset(object):

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        supp_input_ids, supp_labels, supp_images, \
        log_dict, logged_supp_tasks, ict_supp_tasks = tuple([instance[key] for instance in instances]
                                        for key in ("supp_input_ids", "supp_labels", "supp_images",
                                                    "log_dict", "logged_supp_tasks", "ict_supp_tasks"))
        
        supp_input_ids_padded = [torch.nn.utils.rnn.pad_sequence(sublist,
                                                 batch_first=True,
                                                 padding_value=self.tokenizer.pad_token_id)
                                                 for sublist in supp_input_ids]                       
        
        supp_labels_padded = [torch.nn.utils.rnn.pad_sequence(sublist,
                                                    batch_first=True,
                                                    padding_value=IGNORE_INDEX)
                                                    for sublist in supp_labels]
        

        supp_input_ids_padded = [s[:, :self.tokenizer.model_max_length] for s in supp_input_ids_padded]
        supp_labels_padded = [s[:, :self.tokenizer.model_max_length] for s in supp_labels_padded]
        
        batch = dict(
            input_ids=supp_input_ids_padded,
            labels=supp_labels_padded,
            attention_mask=[l.ne(self.tokenizer.pad_token_id) for l in supp_input_ids_padded],
            log_dict=log_dict,
            logged_supp_tasks=logged_supp_tasks,
            ict_supp_tasks=ict_supp_tasks,
            )
        
        batch['images'] = supp_images

        return batch