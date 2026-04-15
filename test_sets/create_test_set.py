from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
from transformers import set_seed
from tqdm import tqdm

set_seed(42)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")

dataset = load_dataset("HuggingFaceFW/fineweb-edu", "CC-MAIN-2025-18",split="train", streaming=True).shuffle()

validation_set = pd.DataFrame(columns=list(dataset.column_names))
test_set = pd.DataFrame(columns=list(dataset.column_names))


i = 0
j = 0
k = 0
for text in tqdm(dataset):
    if j<1000:
        tokenized_text = tokenizer.encode(text["text"])
        if len(tokenized_text)>=1000:
            new_tokenized_text = tokenized_text[:1001]
            untokenized_text = tokenizer.decode(new_tokenized_text)
            text_df = pd.DataFrame(text,index=[j])
            text_df["text"] = untokenized_text
            validation_set = pd.concat([validation_set,text_df])
            j+=1
    elif k<1000:
        tokenized_text = tokenizer.encode(text["text"])
        if len(tokenized_text)>=1000:
            new_tokenized_text = tokenized_text[:1001]
            untokenized_text = tokenizer.decode(new_tokenized_text)
            text_df = pd.DataFrame(text,index=[j])
            text_df["text"] = untokenized_text
            test_set = pd.concat([test_set,text_df])
            k+=1
    if j>=1000 and k>=1000:
        break
    i+=1
    if i>=10**6:
        break
    
validation_set.to_parquet("validation_set.parquet")
test_set.to_parquet("test_set.parquet")