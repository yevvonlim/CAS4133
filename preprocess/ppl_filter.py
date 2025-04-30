#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import loguru
import gc
from utils import set_seed

def compute_ppl(dataloader, model, tokenizer, device):
    """
    Compute perplexity for each batch in the DataLoader.
    Moves inputs to the given device and runs the model in eval mode.
    """
    model.eval()
    all_ppl = []
    torch.cuda.empty_cache()
    
    for i, batch in enumerate(tqdm(dataloader, desc="Computing PPL")):
        # Tokenize inputs (truncate/pad to 512 tokens)
        enc = tokenizer(
            batch["text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        # Move inputs to the target device (GPU 0 for DataParallel)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.inference_mode():
            outputs = model(**enc, labels=enc["input_ids"])
            # Loss is averaged over tokens and GPUs
            ppl_value = torch.exp(outputs.loss.detach().cpu().mean()).item()

        if i < 5:
            loguru.logger.info(f"Batch {i}: PPL = {ppl_value}")
        # Record one value per example in the batch
        all_ppl.extend([ppl_value] * len(batch["text"]))

            
    return all_ppl

def collate_fn(batch):
    """
    Collate function that collects the 'text' field into a list.
    """
    return {"text": [ex["text"] for ex in batch]}

def main():
    # Ensure reproducibility
    set_seed(1)

    # # Use all GPUs if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loguru.logger.info(f"Using device: {device}")

    # # Load model & tokenizer
    # model_name = "Qwen/Qwen3-4B"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.float16,  # FP16 for memory & speed
    # ).to(device)

    # # Wrap in DataParallel to leverage all GPUs
    # if torch.cuda.device_count() > 1:
    #     loguru.logger.info(f"Found {torch.cuda.device_count()} GPUs → enabling DataParallel")
    #     model = torch.nn.DataParallel(model)
    # model.eval()

    # Load dataset
    # dataset = load_from_disk("data/ko_wiki_dataset/train")
    # dataloader = DataLoader(dataset, batch_size=8*8*2, collate_fn=collate_fn)

    # # Compute perplexities
    # ppl_list = compute_ppl(dataloader, model, tokenizer, device)

    # # Release model GPU memory
    # model.to("cpu")
    # del model
    # torch.cuda.empty_cache()
    # gc.collect()

    # Attach PPL column and save
    # dataset = dataset.add_column("ppl", ppl_list)
    # dataset.save_to_disk("data/ko_wiki_dataset/train_with_ppl")
    dataset = load_from_disk("data/ko_wiki_dataset/train_with_ppl")
    # loguru.logger.info("✅ Saved train_with_ppl")

    # Filter out high-perplexity samples and save
    before = len(dataset)
    filtered = dataset.filter(lambda x: x["ppl"] < 500, num_proc=8)
    after = len(filtered)
    loguru.logger.info(f"Filtered {before} → {after} samples (PPL < 500)")
    filtered.save_to_disk("data/ko_wiki_dataset/train_filtered")
    loguru.logger.info("✅ Saved train_filtered")

if __name__ == "__main__":
    main()