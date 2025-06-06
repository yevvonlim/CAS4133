from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from unsloth import FastLanguageModel
import torch
from utils import set_seed
from datasets import load_from_disk
import loguru
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_dataset():
    
    # Load the dataset
    train_dataset = load_from_disk("/workspace/CAS4133/data/train_multi_lang")
    # Split into train and test sets
    test_dataset = load_from_disk("data/ko_wiki_dataset/test")

    return train_dataset, test_dataset


def prepare():
    set_seed(1)
    train_dataset, test_dataset = load_dataset()

    max_seq_length = 1024
    dtype = None # None for auto detection. Bfloat16 for Ampere+ GPUs.
    load_in_4bit = True # Use 4bit quantization to reduce memory usage.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Llama-3.2-1B-unsloth-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    tokenizer = AutoTokenizer.from_pretrained("/workspace/CAS4133/data/korean_tokenizer_new")
     
    model.resize_token_embeddings(
        new_num_tokens=len(tokenizer),
        pad_to_multiple_of=64,         
        mean_resizing=True
    )
    model.config.bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    model.config.eos_token_id = tokenizer.convert_tokens_to_ids("<|end_of_text|>")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.tie_weights() 
    model = FastLanguageModel.get_peft_model(
        model,
        r = 512,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "embed_tokens", "lm_head",], # Add for continual pretraining
        lora_alpha = 32,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = False, # Uses 30% less VRAM
        random_state = 3407,
        use_rslora = True,   # For rank stabilized LoRA
        loftq_config = None,
    )
    

    trainer = UnslothTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = test_dataset, # Make sure to use original test_dataset for eval
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 48,

        args = UnslothTrainingArguments(
            packing = True, # Use packing to increase gradient signal
            logging_steps = 50,
            output_dir = "outputs-new-tokenizer-multi",
            report_to = "wandb", # Use wandb for better logging or Set to None
            
            max_steps = 2500, # DO NOT EXCEED 2500 steps for this assignment
            
            ###### DO NOT CHANGE ######
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 8,
            warmup_steps = 10,
            warmup_ratio = 0.1,
            num_train_epochs = 1,

            learning_rate = 5e-5,
            embedding_learning_rate = 1e-5, # Select a 2 to 10x smaller learning rate for the embedding matrices
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,        
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            ###### DO NOT CHANGE ######
        
        ),
    )
    return trainer, model, tokenizer

def main():
    # Load dataset
    trainer, model, tokenizer = prepare()

    # Train the model
    trainer.train()
    trainer.save_model("outputs/llama3-2-1b-ye")
    tokenizer.save_pretrained("outputs/llama3-2-1b-ye")
    loguru.logger.info("Model and tokenizer saved to outputs/llama3-2-1b-ye")

    # Evaluate the model
    eval_results = trainer.evaluate()
    eval_loss = eval_results["eval_loss"]

    perplexity = torch.exp(torch.tensor(eval_loss))
    loguru.logger.info(f"Perplexity: {perplexity:.2f}")
    loguru.logger.info(f"Eval loss: {eval_loss:.4f}")
    
if __name__ == "__main__":
    main()
   