# Content

This repo contains the minimal necessary code to finetune and infere model.

## Quick start

### Installation

Before anything please install the necessary library:

It's recommanded to use a virtual environement:

```bash
python -m ./venv
./venv/bin/activate
```

1. Install unsloth for fast training

To install unlsoth please look at the original [repo](https://github.com/unslothai/unsloth?tab=readme-ov-file#pip-installation)

2. Install the necessary other library

```bash
pip install -r requirements.txt
```

### Repo organisation

1. Finetuning

finetune.py can finetune model using unsloth

For more detail see the section finetuning bellow

2. Inference

infere.py can infere a model via the class

For more detail see the section inference bellow

3. RAG

rag.py can be used as a memory retriever for the model

For more detail see the section rag bellow

## Finetuning

The dataset should be a Dataset object provided by the datasets library.

It should have the 2 splits train and test.

To save the model finetuned you can call `save_model_inference` and put the number of bits for quantization in parameter from [3,4,6,8]

If the saving crash you can quantize the model manually.

If it crashes you should have the llama.cpp repo cloned at the root of the project.

You can run the following command to quantize the model:

```bash
python llama.cpp/convert-hf-to-gguf.py ./finetuned --outfile ./finetuned-unsloth.BF16.gguf --outtype bf16
llama.cpp/llama-quantize ./finetuned-unsloth.BF16.gguf ./finetuned-unsloth.Q6_K.gguf q6_k 12
```



### DPO

To use DPO, add use_dpo=True when calling `prepare_for_training`

The necessary columns for DPO are prompt, chosen and rejected.

You can rename the columns by calling:

```python
dataset["train"].rename_columns(
    {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
)
```

### SFT

To use SFT, add use_dpo=False when calling `prepare_for_training`

The necessary column for SFT is text.

This colums must contain the text that the model should generate


## Inference

### Usual inference

You can use the path to the file saved from the finetuned one.

You can change the system prompt for better result.

## RAG

You can use the RAG model to retrieve information from a pdf for example.

The memory only accept text for now.

It compute the similarity between the text in prompt and the memory retrieved.

It outputs the original text without any modification.

## Test

You can test all the functionnalities by running the test.py file. with the command:

```bash
python test.py --model_path "path_to_model" --dataset_path "path_to_dataset" --file_path "path_to_file" --prompt "Principe fondamental de la dynamique"
```


## TODO

- [ ] Add LoRA support to inference
- [x] add RAG

