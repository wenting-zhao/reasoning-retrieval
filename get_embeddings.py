from datasets import load_dataset, Dataset
from tqdm import tqdm
from openai import OpenAI
from utils import get_embedding, extract_equations
import sys


ds = load_dataset('gsm8k', 'main', split='train')

new_ds = []
for d in tqdm(ds):
    try:
        eqs = extract_equations(d['answer'])
        d['embedding'] = get_embedding(eqs)
        #d['embedding'] = get_embedding(f"Q: {d['question']}\nA: {d['answer']}")
        #d['embedding'] = get_embedding(d['question'])
        new_ds.append(d)
    except Exception as e:
        print(d["question"])
        print(e)
        print("="*100)
new_ds = Dataset.from_list(new_ds)
new_ds.to_json(f"embeddings/embedding_eqs.json")
