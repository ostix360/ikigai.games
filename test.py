from document import Document
import pdfplumber

# from .infere import Inference
# from .finetune import TrainerProcessor
from rag import Memory
from datasets import load_dataset
from argparse import ArgumentParser

INSTRUCT = {"role": "system",
                 "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user."}

def test_inference(model_path):
    """
    Test the inference of a model using the given model path.

    Args:
        model_path (str): The path to the model file.

    Returns:
        None

    """
    inf = Inference(model_path)
    response = inf.generate_response("Make a story about Joe and his dog.")

    print(response)

def test_finetune(dataset_path):
    """
    Fine-tunes a model using the provided dataset.

    Args:
        dataset_path (str): The path to the dataset.

    Returns:
        None
    """
    trainer_processor = TrainerProcessor(None)
    dataset = load_dataset(dataset_path)

    def process_data(example):
        """
        Preprocesses examples from the dataset.

        Args:
            example (str): The examples to be processed.

        Returns:
            dict: The processed examples.
        """
        return { 
            "text" :
                trainer_processor.tokenizer.tokenizer.apply_chat_template(
                    [INSTRUCT, {"role": "user", "content": example}], add_generation_prompt=True
                )
        }
    
    dataset = dataset.map(process_data)
    dataset = dataset.train_test_split(test_size=0.05)
    trainer_processor.set_dataset(dataset)
    trainer, _ = trainer_processor.prepare_for_training("train", use_dpo=False)
    trainer.train()
    trainer_processor.save_model_for_inference(output_dir="finetuned",bits=6)

def test_rag(file_path, prompt):
    """
    Extracts text content from a PDF file, saves it to a database, and retrieves relevant memories based on a prompt.

    Args:
        file_path (str): The path to the PDF file.
        prompt (str): The prompt used to retrieve relevant memories.

    Returns:
        None
    """
    memory = Memory("database")
    memory.create_collection("physics")
    # with pdfplumber.open(file_path) as pdf:
    #     content = "\n".join([page.extract_text() for page in pdf.pages])
    # document = Document(content, file_path.split("/")[-1].split(".")[0])
    # memory.save_documents_to_db(collection_name="physics", documents=document)
    extract = memory.get_memories(prompt, collection_name="physics")
    print(extract)

if __name__ == "__main__":
    parser = ArgumentParser(description="Test the different components of the project.")
    parser.add_argument("--model_path", type=str, help="The path to the model to be tested during inference.")
    parser.add_argument("--dataset_path", type=str, help="The path to the dataset to be tested during fine-tuning.")
    parser.add_argument("--file_path", type=str, help="The path to the pdf to be tested during RAG.")
    parser.add_argument("--prompt", type=str, help="The prompt to be tested during RAG.")
    args = parser.parse_args()

    if args.model_path:
        test_inference(args.model_path)
    if args.dataset_path:
        test_finetune(args.dataset_path)
    if args.file_path and args.prompt:
        test_rag(args.file_path, args.prompt)
