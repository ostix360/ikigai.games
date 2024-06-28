from argparse import ArgumentParser

from data.document import Document

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
    from infere import Inference
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
    from utils import get_datasets, apply_chat_template
    from finetune import TrainerProcessor

    trainer_processor = TrainerProcessor(max_seq_length=2048)
    if dataset_path == "default":
        dataset_path = "HuggingFaceH4/ultrafeedback_binarized"
    """
    To use the utility function, the dataset must be have the following columns:
    - prompt: The prompt of the user.
    - chosen: The correct / better response to the prompt.
    - rejected: The incorrect / worse response to the prompt.
    """
    raw_datasets = get_datasets(
        {dataset_path: 0.02},  # 2% sampled
        splits=["train_prefs"],  # if you have splits in the dataset
    )
    column_names = list(raw_datasets["train"].features)

    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": trainer_processor.tokenizer, "task": "dpo"},
        num_proc=12,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    dataset = raw_datasets.rename_columns(
        {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"})
    dataset = dataset["train"].train_test_split(test_size=0.05)  # Not necessary for already split dataset
    trainer_processor.set_dataset(dataset)
    trainer, _ = trainer_processor.prepare_for_training("train", use_orpo=True, batch_size=1)
    trainer.train()
    trainer_processor.save_model_for_inference(output_dir="finetuned", bits=6)


def test_save_model():
    """
    Save the model for inference.

    Args:

    Returns: None
    """
    from finetune import TrainerProcessor
    import os
    os.environ['PATH'] += ':' + "/home/ostix/.virtualenvs/ikigai/bin"
    trainer_processor = TrainerProcessor(max_seq_length=2048, model_name_or_path="finetuned")

    trainer_processor.save_model_for_inference(output_dir="finetuned", bits=6)


def test_rag(file_path, prompt):
    """
    Extracts text content from a PDF file, saves it to a database, and retrieves relevant memories based on a prompt.

    Args:
        file_path (str): The path to the PDF file.
        prompt (str): The prompt used to retrieve relevant memories.

    Returns:
        None
    """
    from rag import Memory
    import pdfplumber
    import nltk
    nltk.download('punkt')

    memory = Memory(db_path="database", chunk_size=512)  # create a memory object
    memory.create_collection("physics")  # create a collection
    with pdfplumber.open(file_path) as pdf:  # read the pdf file
        content = "\n".join([page.extract_text() for page in pdf.pages])
    document = Document(content, file_path.split("/")[-1].split(".")[0])  # create a document object
    memory.save_documents_to_db(collection_name="physics", documents=document)  # save the document to the collection
    extract = memory.get_memories(prompt, collection_name="physics", limit=8, )  # retrieve relevant memories
    # save the memories to a file
    with open("memories.txt", "w") as f:
        f.write(str(extract))
    print(extract)


if __name__ == "__main__":
    parser = ArgumentParser(description="Test the different components of the project.")
    parser.add_argument("--model_path", type=str, help="The path to the model to be tested during inference.")
    parser.add_argument("--dataset_path", type=str, help="The path to the dataset to be tested during fine-tuning.")
    parser.add_argument("--file_path", type=str, help="The path to the pdf to be tested during RAG.")
    parser.add_argument("--prompt", type=str, help="The prompt to be tested during RAG.")
    parser.add_argument("--save_model", type=bool, help="Save the model for inference.")
    args = parser.parse_args()

    if args.model_path:
        print("Testing inference")
        test_inference(args.model_path)
        print("inference test succeed")
    if args.dataset_path:
        print("Testing fine tuning")
        test_finetune(args.dataset_path)
        print("fine-tuning succeed")
    if args.file_path and args.prompt:
        test_rag(args.file_path, args.prompt)
        print("RAG test succeed")
    if args.save_model:
        test_save_model()
