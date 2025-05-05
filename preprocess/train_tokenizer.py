from transformers import AutoTokenizer 
from datasets import load_from_disk
import loguru

def prepare_tokenizer(model_name_or_path):
    """
    Load the tokenizer from the specified model name or path.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return tokenizer

def prepare_dataset(dataset_path):
    """
    Load the dataset from the specified path.
    """
    dataset = load_from_disk(dataset_path)
    training_corpus = (
        dataset[i : i + 1000]["text"]
        for i in range(0, len(dataset), 1000)
    )
    return training_corpus

def train_tokenizer(tokenizer, dataset, output_dir):
    training_corpus = prepare_dataset(dataset)
    loguru.logger.info("training corpus loaded")
    # Train the tokenizer on the training corpus
    loguru.logger.info("training tokenizer")
    new_tokenizer = tokenizer.train_new_from_iterator(training_corpus, 
                                                      vocab_size=128_000,  # to prevent oov
                                                    )
    loguru.logger.info("tokenizer trained")
    # Save the tokenizer to the specified output directory
    new_tokenizer.save_pretrained(output_dir)
    loguru.logger.info(f"tokenizer saved to {output_dir}")

if __name__ == "__main__":
    # Specify the model name or path and the dataset path
    model_name_or_path = "unsloth/Llama-3.2-1B-unsloth-bnb-4bit"
    dataset_path = "/workspace/CAS4133/data/tokenizer_train"
    output_dir = "data/korean_tokenizer"

    # Prepare the tokenizer and dataset
    tokenizer = prepare_tokenizer(model_name_or_path)
    loguru.logger.info("tokenizer loaded")
    
    # Train the tokenizer
    train_tokenizer(tokenizer, dataset_path, output_dir)

