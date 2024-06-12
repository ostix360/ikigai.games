from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, DPOTrainer
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from typing import Union, Literal

# if using jupyter notebook DPOTrainer need to be patched
# from unsloth import PatchDPOTrainer
# PatchDPOTrainer()


class TrainerProcessor:
    """

    """

    def __init__(
        self,
        dataset: Dataset,
        max_seq_length:int = 2048,
        model_name_or_path:str = "unsloth/llama-3-8b-bnb-4bit",
        ) -> None:
        """

        """
        self.max_seq_length = max_seq_length 
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name_or_path,
            max_seq_length = max_seq_length,
            dtype = None,
            load_in_4bit = True,
        )
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = 32,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 64,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            max_seq_length = max_seq_length,
        )
        self.dataset = dataset

    def set_dataset(self, dataset: Dataset):
        """
        Set the dataset for the model.

        Parameters:
        - dataset (Dataset): The dataset to be set.

        Returns:
        - None

        """
        self.dataset = dataset

    def prepare_for_training(
            self,
            outuput_dir: str,
            use_dpo: bool,
            batch_size: int = 8,
            num_epoch: int = 1,
        ) -> Union[Trainer, TrainingArguments]:
        """

        """
        args = TrainingArguments(
            per_device_train_batch_size = batch_size,
            warmup_ratio = 0.05,
            num_train_epochs = num_epoch,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            output_dir = outuput_dir,
        )
        if use_dpo:
            trainer = DPOTrainer(
                model = self.model,
                ref_model = None,
                args = args,
                beta = 0.1,
                train_dataset = self.dataset["train"],
                eval_dataset = self.dataset["test"],
                tokenizer = self.tokenizer,
                max_length = self.max_seq_length,
            )
        else:
            trainer = SFTTrainer(
                model = self.model,
                train_dataset = self.dataset["train"],
                eval_dataset = self.dataset["test"],
                dataset_text_field = "text",
                max_seq_length = self.max_seq_length,
                tokenizer = self.tokenizer,
                dataset_num_proc = 2,
                packing = False,
                args = args,
            )
        return trainer, args

    def save_model_for_inference(self, output_dir: str, bits:Literal[3,4,6,8] = 4):
        """
        
        """
        if bits == 8:
            quant = "q8_0"
        else:
            quant = f"q{bits}_k"
        self.model.save_pretrained_gguf(output_dir, self.tokenizer, quantization_method = quant)
