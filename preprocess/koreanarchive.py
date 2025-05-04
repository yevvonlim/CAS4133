from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
import pandas as pd
from transformers import AutoTokenizer

# 1. 원본 HF 데이터셋 로드
ds = load_dataset("sionic-ai/korean-archive-dataset", "baseline")
#    └ splits: "queries", "corpus"
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-unsloth-bnb-4bit")
# 2. qrels 읽어오기
qrels_path = hf_hub_download(
    repo_id="sionic-ai/korean-archive-dataset",
    filename="qrels_korean_archive_strategyqa.csv",
    repo_type="dataset",
)
qrels = pd.read_csv(qrels_path)  # columns: query_id, corpus_id, score

# 3. ID → text 매핑 생성
query_map  = {ex["_id"]: ex["text"] for ex in ds["queries"]}
corpus_map = {ex["_id"]: ex["text"] for ex in ds["corpus"]}

# 4. qrels 순회하며 text 합치기
paired_texts = []
for _, row in qrels.iterrows():
    qid = row["query_id"]
    cid = row["corpus_id"]
    q_text = query_map.get(qid, "")
    c_text = corpus_map.get(cid, "")
    if q_text and c_text:
        # 원하는 방식으로 구분자(여기선 한 줄바꿈) 추가 가능
        combined = f"""질문에 대답하기 위한 문서 리스트들을 나열하세요. \n
질문: {q_text} \n
문서 리스트: {c_text}""" + tokenizer.eos_token
        paired_texts.append(combined)

# 5. 최종 Dataset 생성
final_ds = Dataset.from_dict({"text": paired_texts})

# check
print(f"Total samples: {len(final_ds)}")
# sample
print(final_ds[0])
final_ds.save_to_disk("data/korean-archive-dataset")
