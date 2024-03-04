import sys
import time
import numpy as np
import torch
import os
import openai
import re
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def get_embedding(text, model="text-embedding-3-large"):
    error = True
    while error:
        try:
            out = client.embeddings.create(input = [text], model=model).data[0].embedding
            error = False
        except openai._exceptions.OpenAIError as e:
            print(type(e), e)
            time.sleep(1)
    return out

def query_chat(messages, model, tokenizer=None, temperature=1, max_tokens=512):
    if isinstance(model, str):
        error = True
        while error:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                output = response.choices[0].message.content.strip()
                error = False
            except openai._exceptions.OpenAIError as e:
                if 'context_length_exceeded' in str(e):
                    if len(messages) > 1:
                        messages = messages[-1:]
                    else:
                        messages[-1]['content'] = messages[-1]['content'][int(0.9*len(messages[-1]['content'])):]
                time.sleep(5)
                print(type(e), e)
    else:
        if 'system' not in tokenizer.default_chat_template:
            messages = messages[1:]
        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
        attn_mask = torch.ones_like(tokenized_chat, device=device)
        with torch.no_grad():
            outputs = model.generate(tokenized_chat, attention_mask=attn_mask, max_new_tokens=max_tokens, temperature=temperature, do_sample=True)
        output = tokenizer.decode(outputs[0][len(tokenized_chat[0]):], clean_up_tokenization_spaces=True, skip_special_tokens=True).strip()
    return output

def load_model_and_tokenizer(name):
    if "gpt" not in name:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(name)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        model = name
        tokenizer = ""
    return model, tokenizer

def extract_answer_and_chain(response):
    answer_pattern = re.compile(r"####.*")
    match = answer_pattern.search(response)
    if not match:
        return None, None
    match = match.group(0).replace(',', '')
    try:
        answer = re.search(r'-?[0-9.]+', match).group(0)
    except Exception:
        return None, None
    answer = [x for x in answer.split('.') if x != '']
    if len(answer) == 1:
        answer = float(answer[0])
    elif len(answer) == 2:
        answer = float(f'{answer[0]}.{answer[1]}')
    else:
        answer = 0
    reasoning = response.split("####")[0]
    reasoning = [one.strip() for one in reasoning.split('\n')]
    reasoning = [one for one in reasoning if one != ""]
    return answer, reasoning

def check_response(predict_response, actual_result):
    cur_answer, cur_chain = extract_answer_and_chain(predict_response)
    real_answer, real_chain = extract_answer_and_chain(actual_result)
    if cur_chain is None or len(cur_chain) == 0:
        print("!!!!!EMPTY CHAIN!!!!!", cur_chain, predict_response)
    else:
        for one in cur_chain:
            if not check_equation(one):
                print("!!!!!REASONING ERROR!!!!!")
                return False
    answer_correct = cur_answer == real_answer
    if not answer_correct:
        print("!!!!!ANSWER ERROR!!!!!")
    return answer_correct

def check_answer(predict_response, actual_result):
    cur_answer, _ = extract_answer_and_chain(predict_response)
    real_answer, _ = extract_answer_and_chain(actual_result)
    return cur_answer == real_answer

def extract_equations(text):
    match = re.findall(r"<<.*>>", text)
    return '\n'.join(match)
