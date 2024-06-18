from typing import Literal

from datasets import DatasetDict
from transformers import TrainingArguments
from trl import SFTTrainer, DPOTrainer, ORPOConfig, ORPOTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

# if using jupyter notebook DPOTrainer need to be patched
# from unsloth import PatchDPOTrainer
# PatchDPOTrainer()


class TrainerProcessor:
    """
    This class is used to process and train a model using a given dataset.

    Attributes:
        max_seq_length (int): Maximum sequence length for the model.
        model (FastLanguageModel): The model to be trained.
        tokenizer (Tokenizer): The tokenizer for the model.
        dataset (DatasetDict): The dataset for training the model.
    """

    def __init__(
            self,
            max_seq_length: int = 2048,
            model_name_or_path: str = "unsloth/zephyr-sft-bnb-4bit",
    ) -> None:
        """
        Initializes the TrainerProcessor with the given parameters.

        Args:
            max_seq_length (int, optional): Maximum sequence length for the model. Defaults to 2048.
            model_name_or_path (str, optional): The name or path of the model. Defaults to "unsloth/zephyr-sft-bnb-4bit".
        """
        self.max_seq_length = max_seq_length
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name_or_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        self.dataset = None
        self.max_seq_length = max_seq_length

    def set_dataset(self, dataset: DatasetDict):
        """
        Set the dataset for the model.

        Args:
            dataset (Dataset): The dataset to be set.
        """
        self.dataset = dataset
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj", ],
            lora_alpha=32,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing=True,  # True or "unsloth" for very long context
            max_seq_length=self.max_seq_length,
        )

    def prepare_for_training(
            self,
            outuput_dir: str,
            use_orpo: bool,
            batch_size: int = 8,
            num_epoch: int = 1,
    ) -> tuple[DPOTrainer | SFTTrainer, TrainingArguments]:
        """
        Prepares the model for training.

        Args:
            outuput_dir (str): The directory where the output will be saved.
            use_orpo (bool): Whether to use ORPO for training.
            batch_size (int, optional): The batch size for training. Defaults to 8.
            num_epoch (int, optional): The number of epochs for training. Defaults to 1.

        Returns:
            tuple[DPOTrainer | SFTTrainer, TrainingArguments]: The trainer and training arguments.

        """
        args = TrainingArguments(
            per_device_train_batch_size=batch_size,
            warmup_ratio=0.05,
            num_train_epochs=num_epoch,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=50,
            optim="adamw_8bit",
            output_dir=outuput_dir,
        )
        if use_orpo:
            args = ORPOConfig(
                per_device_train_batch_size=batch_size,
                warmup_ratio=0.05,
                num_train_epochs=num_epoch,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                learning_rate=3e-5,
                logging_steps=50,
                optim="adamw_8bit",
                output_dir=outuput_dir,
                max_length=self.max_seq_length,
                max_prompt_length=self.max_seq_length // 2,
                max_completion_length=self.max_seq_length // 2,
                beta=0.1,
            )
            trainer = ORPOTrainer(
                model=self.model,
                args=args,
                train_dataset=self.dataset["train"],
                eval_dataset=self.dataset["test"],
                tokenizer=self.tokenizer,
            )
        else:
            trainer = SFTTrainer(
                model=self.model,
                train_dataset=self.dataset["train"],
                eval_dataset=self.dataset["test"],
                dataset_text_field="text",
                max_seq_length=self.max_seq_length,
                tokenizer=self.tokenizer,
                packing=False,
                args=args,
            )
        return trainer, args

    def save_model_for_inference(self, output_dir: str, bits: Literal[3, 4, 6, 8] = 4):
        """
        Saves the model for inference.

        Args:
            output_dir (str): The directory where the model will be saved.
            bits (Literal[3, 4, 6, 8], optional): The number of bits for quantization. Defaults to 4.
        """
        if bits == 8:
            quant = "q8_0"
        else:
            quant = f"q{bits}_k"
        self.model.save_pretrained_gguf(output_dir, self.tokenizer, quantization_method=quant)
