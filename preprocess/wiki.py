from datasets import load_dataset
import argparse
import os


def build_prompt_fn(prompt_template, eos_token):
    """
    Create a function that formats examples into prompt strings with the EOS token appended.
    """
    def _fn(examples):
        titles = examples["title"]
        texts  = examples["text"]
        formatted = []
        for title, text in zip(titles, texts):
            formatted.append(prompt_template.format(title, text) + eos_token)
        return {"text": formatted}
    return _fn


def main():
    parser = argparse.ArgumentParser(
        description="Filter and preprocess English/Japanese Wikipedia articles, format prompts, and save to disk"
    )
    parser.add_argument(
        "--output_en",
        required=True,
        help="Directory to save processed English dataset (e.g., ./wiki_en_40k)"
    )
    parser.add_argument(
        "--output_ja",
        required=True,
        help="Directory to save processed Japanese dataset (e.g., ./wiki_ja_20k)"
    )
    parser.add_argument(
        "--date_en",
        default="20220301",
        help="Dump date for English Wikipedia (YYYYMMDD). Must match available dumps."
    )
    parser.add_argument(
        "--date_ja",
        default="20231101",
        help="Dump date for Japanese Wikipedia (YYYYMMDD). Must match available dumps."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for dataset shuffling (default: 42)"
    )
    args = parser.parse_args()

    configs = {
        "en": {
            "num_samples": 40000,
            "prompt": "# Wikipedia article\n### Title: {}\n\n### Article:\n{}",
            "output_dir": args.output_en,
            "config_name": f"{args.date_en}.en",
        },
        "ja": {
            "num_samples": 20000,
            "prompt": "# ウィキペディア記事\n### タイトル: {}\n\n### 記事:\n{}",
            "output_dir": args.output_ja,
            "config_name": f"{args.date_ja}.ja",
        },
    }

    # Load tokenizer once to get the EOS token
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    eos_token = tokenizer.eos_token or ""

    for lang, cfg in configs.items():
        out_dir = cfg["output_dir"]
        if os.path.exists(out_dir):
            continue  # Skip if the output directory already exists
        os.makedirs(out_dir, exist_ok=False)

        # Load the specified Wikipedia dump configuration
        ds = load_dataset("wikimedia/wikipedia", cfg["config_name"], split="train", num_proc=128)

        # Shuffle and select the desired number of samples
        ds = ds.shuffle(seed=args.seed).select(range(cfg["num_samples"]))

        # Format examples using the prompt template and EOS token
        prompt_fn = build_prompt_fn(cfg["prompt"], eos_token)
        ds = ds.map(prompt_fn, batched=True, remove_columns=["title", "text"])

        # Save the processed dataset to disk
        ds.save_to_disk(out_dir)
        print(f"✅ [{lang}] Saved {cfg['num_samples']} samples (config: {cfg['config_name']}) to {out_dir}")

if __name__ == "__main__":
    main()
