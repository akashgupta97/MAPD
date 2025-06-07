<h1 align="center">Meta-Adaptive Prompt Distillation for Few-Shot Visual Question Answering</h1>
<p align="center">
  <a href="https://scholar.google.com/citations?user=ypzuEjgAAAAJ&hl=en&authuser=1">Akash Gupta</a>,
  <a href="https://homepages.inf.ed.ac.uk/amos">Amos Storkey</a>,
  <a href="https://homepages.inf.ed.ac.uk/mlap">Mirella Lapata</a>,
</p>

___

This repository contains the code for the paper Meta-Adaptive Prompt Distillation for Few-Shot Visual Question Answering or MAPD, a meta-learning approach for inducing few-shot capabilties in the LMMs using a fixed set of soft prompts that are distilled from task-relevant image features and can be adapted at test time using as low as 1 example. Visit our 📃 paper on arxiv - [Link](dummy_link). This code is based on the LLaVA repository - [Link](https://github.com/haotian-liu/LLaVA/tree/main) and below we list out steps for running training and evaluation for MAPD and other prompt distillation approaches.

## Install dependencies

```Shell
conda create -n MAPD python=3.10 -y
conda activate MAPD
pip install -e .
pip install -e ".[train]"
pip install accelerate==0.21.0
pip install peft==0.13.2
pip install flash-attn==2.6.3 --no-build-isolation
```

## Data Preparation

The current implementation of MAPD uses the LLaVA pretraining and finetuning datasets. We refer to the publicly available release of these datasets on HuggingFace.

### Pretraining

We use the LCS-558K subset (also used in LLaVA v1.5 pretraining) of the LAION/CC/SBU dataset filtered with a more balanced concept coverage. [Link](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)

The final datasets directory should be structured in the following way:

```
datasets/
|-- LLaVA-Pretrain
|   `-- Image_data
|       |-- blip
```


### Finetuning

For finetuning, we use the LLaVA v1.5 finetuning data mixture, which can be downloaded from here - [Link](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)

We remove the ShareGPT-40K dataset from the llava_v1_5_mix665k.json as we do not use unimodal text-only data.

We further include 3 datasets in our finetuning from the LLaVA-OneVision data mixture designed to solve mathematical question answering tasks - MAVIS_math_metagen, TabMWP_Cauldron, geo170k(qa). These can be downloaded from here. [Link](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data)
(NOTE: geo170k is further divided (basic, reasoning) based on the instructions depending upon if it requires reasoning.)

The final datasets directory should be structured in the following way:

```
datasets/
|-- LLaVA-Instruct
|   |-- Image_data
|   |   |-- aokvqa
|   |   |-- basic_qa_geo170k
|   |   |-- coco
|   |   |-- complex_res
|   |   |-- conv
|   |   |-- det
|   |   |-- gqa
|   |   |-- mavis_math_metagen
|   |   |-- ocr_vqa
|   |   |-- okvqa
|   |   |-- reasoning_qa_geo170k
|   |   |-- refcoco
|   |   |-- sharegpt
|   |   |-- tabmwp_cauldron
|   |   |-- textvqa
|   |   |-- vg
|   |   `-- vqav2
```

Each dataset has its own conversation json split file which is needed for meta-task creation and we perform the split in the following way - MAPD and all other prompt distillation approaches require separating all the datasets so as to create meta-tasks for training. In the LLaVA v1.5 mixture, we simply separate all the datasets by either searching for the available dataset keyword names or based on the task instructions as provided in Table 8 in the paper - *Improved baselines with Visual Instruction Tuning* ([Link](https://arxiv.org/pdf/2310.03744)) in the above conversation data (llava_v1_5_mix665k.json)

The images should be placed in the respective folders inside each dataset directory based on their image paths.


## Model Training

### Compute Requirements

The current model training pipeline uses 4 H200 GPUs with a 143GB VRAM per GPU with different gradient accumulation steps for different prompt distillation approaches as mentioned in Appendix A.1.3

### Pretraining

Please run the below command to start model pretraining

```Shell
bash scripts/v1_5/pretrain_qwen_sl.sh
```

### Finetuning

Please run the below command to start model finetuning

**MAPD**

```Shell
bash scripts/v1_5/finetune_qwen_mapd.sh
```

**Multi-Task**

```Shell
bash scripts/v1_5/finetune_qwen_mltasks.sh
```

**NoMetaTask**

```Shell
bash scripts/v1_5/finetune_qwen_sl.sh
```

**ICT**

```Shell
bash scripts/v1_5/finetune_qwen_ict.sh
```

**ModelAvg**

This uses the same script as NoMetaTask but we finetune the attention-mapper separately on each dataset in our finetuning data mixture and then compute a weighted average of parameters.

The MAML code is borrowed from the implementation of Antoniou et al. - *How to Train Your MAML* ([Link](https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/tree/master)) and our modified version can be found in the file ```MAPD/llava/train/few_shot_learning_system.py```.

## Model Evaluation

Our model evaluation involves both in-context learning (ICL) and finetuning-based (FT) adaptation. We provide our evaluation script in ```MAPD/llava/eval/run_eval_meta.py```.

To run evaluation, please run the following bash file that runs the evaluation script
```Shell
bash llava/eval/run_eval_meta.sh
```

In this bash script, For FT set ```--finetuning True```, ICL ```--in-context True```

We use the VL-ICL benchmark for our few-shot evaluation - *VL-ICL BENCH: THE DEVIL IN THE DETAILS OF MULTIMODAL IN-CONTEXT LEARNING* ([Link](https://arxiv.org/pdf/2403.13164)).


## Citation


