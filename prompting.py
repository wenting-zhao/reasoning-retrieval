import argparse
import re
from tqdm import tqdm
from datasets import load_dataset
from utils import query_chat, get_embedding, extract_answer_and_chain, extract_equations
from utils import load_model_and_tokenizer
import nltk.data
import faiss
import numpy as np

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", action="store_true", help="whether to retrieve")
    parser.add_argument("--option", type=str, default="question", help="what to base the retrieval on")
    parser.add_argument("--num", type=int, default=8, help="number of examples to use in the context")
    parser.add_argument("--seed", type=int, default=42, help="seed for shuffling the train set")
    parser.add_argument("--datastore", type=str, help="number of examples to use in the context")
    parser.add_argument("--model", type=str, help="what model to use")
    args = parser.parse_args()
    model, tok = load_model_and_tokenizer(args.model)
    ds_retrival = load_dataset("gsm8k", 'main', split='train')
    demonstrations = ds_retrival.shuffle(seed=args.seed).select(range(args.num))
    if args.retrieval:
        datastore = load_dataset("json", data_files=args.datastore, split='train')
        d = len(datastore['embedding'][0])
        index = faiss.IndexFlatL2(d)
        index.add(np.array(datastore['embedding']))
        print(f"added {index.ntotal} embeddings to index")
    ds_test = load_dataset("gsm8k", 'main', split='test')
    acc = 0
    outs = []
    old_outs = []
    retrieved = []
    sys_msg = {"role": "system", "content": "Your task is to solve a primary school level math problem. You should provide both a chain of reasoning and a final answer. Each reasoning step should involve only one arithmetic operation, which should be enclosed within double angle brackets, following this format: <<48/2=24>>. Every step must be presented on a separate line. The final answer should be preceded by ####, for instance, #### 72."}
    for idx, one in tqdm(enumerate(ds_test), total=len(ds_test)):
        messages = [sys_msg]
        for q, a in zip(demonstrations['question'], demonstrations['answer']):
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": one['question']})
        out = query_chat(messages, model, tok)
        if args.retrieval:
            if args.option == "cot":
                query = get_embedding(f"Q: {one['question']}\nA: {out}")
            elif args.option == "question":
                query = get_embedding(one['question'])
            elif args.option == "eqs":
                text = extract_equations(out)
                while text == '':
                    out = query_chat(messages, model, tok)
                    text = extract_equations(out)
                query = get_embedding(text)
            else:
                raise NotImplementedError(f"{args.option} is not implemented")
            old_outs.append(out)
            if query is not None:
                query = np.array(query).reshape(1, -1)
                _, knns = index.search(query, args.num)
                questions = [datastore['question'][i] for i in knns[0]]
                answers = [datastore['answer'][i] for i in knns[0]]
                messages = [sys_msg]
                for q, a in zip(questions, answers):
                    messages.append({"role": "user", "content": q})
                    messages.append({"role": "assistant", "content": a})
                messages.append({"role": "user", "content": one['question']})
                out = query_chat(messages, model, tok)
                retrieved.append([(q, a) for q, a in zip(questions, answers)])
            else:
                retrieved.append([])
        old_pred, _ = extract_answer_and_chain(old_outs[-1])
        pred, _ = extract_answer_and_chain(out)
        label, _ = extract_answer_and_chain(one['answer'])
        res = pred == label
        if res:
            acc += 1
        print(res)
        print(old_outs[-1], '|', old_pred)
        print(out, '|', pred)
        print(one['answer'], '|', label)
        print("="*100)
        outs.append(out)
    print(acc/len(ds_test))
    ds_test = ds_test.add_column(name='out', column=outs)
    if args.retrieval:
        ds_test = ds_test.add_column(name='original', column=old_outs)
    name = f"{args.model}-{args.num}"
    if args.retrieval:
        ds_test = ds_test.add_column(name='retrieval', column=retrieved)
        name += f'-retrieval-{args.option}'
    name += '.json'
    name = 'outputs/'+name
    ds_test.to_json(name)

if __name__ == '__main__':
    main()
