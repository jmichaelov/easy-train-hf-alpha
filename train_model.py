from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig,DataCollatorForLanguageModeling,Trainer,TrainingArguments
from transformers import set_seed
import yaml
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Trains a model given pre-set parameters')

    parser.add_argument('--config_file', '-c', type=str, default=None,
                        help='Path to a config YAML file. Allows user to pre-specify all arguments for convenience/documentation.')
    parser.add_argument('--seed_num', '-s', type=int, default=None,
                        help='Seed number')    
    args = parser.parse_args()
    return args

args = parse_args()


assert args.config_file is not None, "No training config specified."

with open(args.config_file) as f:
    config = yaml.safe_load(f)

if args.seed_num is not None:
    config["seed_num"] = args.seed_num


assert config["name"] is not None, "No name specified in config."
assert config["seed_num"] is not None, "No seed_num specified in config."
assert config["output_directory"] is not None, "No output_directory specified in config."
assert config["model_config_path"] is not None, "No model_config_path specified in config."
assert config["tokenizer_path"] is not None, "No tokenizer_path specified in config."
assert config["train_type"] is not None, "No train_type specified in config."
assert config["validation_type"] is not None, "No validation_type specified in config."
if config["train_type"]=="hf":
    assert config["hf_train"] is not None, "HF train set specified, but no specific hf_train specified in config."
    assert config["hf_train_config"] is not None, "HF train set specified, but no specific hf_train_config specified in config."
    assert config["hf_train_split"] is not None, "HF train set specified, but no specific hf_train_split specified in config."
elif config["train_type"]=="local":
    assert config["local_train_path"] is not None, "Local train set specified, but no specific local_train_path specified in config."
    assert config["local_train_files"] is not None, "Local train set specified, but no specific local_train_files specified in config."
else:
    raise ValueError("Invalid training set type.")
if config["validation_type"]=="hf":
    assert config["hf_validation"] is not None, "HF validation set specified, but no specific hf_validation specified in config."
    assert config["hf_validation_config"] is not None, "HF validation set specified, but no specific hf_validation_config specified in config."
    assert config["hf_validation_split"] is not None, "HF validation set specified, but no specific hf_validation_split specified in config."
elif config["validation_type"]=="local":
    assert config["local_validation_path"] is not None, "Local validation set specified, but no specific local_validation_path specified in config."
    assert config["local_validation_files"] is not None, "Local validation set specified, but no specific local_validation_files specified in config."
else:
    raise ValueError("Invalid validation set type.")
assert config["batch_size"] is not None, "No batch_size specified in config."
assert config["gradient_accumulation"] is not None, "No gradient_accumulation specified in config."
assert config["model_save_step"] is not None, "No model_save_step specified in config."
assert config["maximum_save_steps"] is not None, "No maximum_save_steps specified in config."
assert config["max_context_length"] is not None, "No max_context_length specified in config."
assert config["lr_scheduler"] is not None, "No lr_scheduler specified in config."


set_seed(config["seed_num"])

tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"],add_bos_token=True)

def tokenize_train(element):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=config["max_context_length"],
        return_overflowing_tokens=True,
        return_length=True,
        add_special_tokens=True
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == config["max_context_length"]:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

def tokenize_val(element):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=config["max_context_length"],
        return_overflowing_tokens=True,
        return_length=True,
        add_special_tokens=True
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        input_batch.append(input_ids)
    return {"input_ids": input_batch}


if config["train_type"]=="hf":
    ds_train = load_dataset(config["hf_train"], config["hf_train_config"], split=config["hf_train_split"],streaming=True)
elif config["train_type"]=="local":
    ds_train = load_dataset(path=config["local_train_path"],data_files={"train":config["local_train_files"]})
    ds_train = ds_train["train"]

if config["validation_type"]=="hf":
    ds_val = load_dataset(config["hf_validation"], config["hf_validation_config"], split=config["hf_validation_split"],streaming=True)
elif config["validation_type"]=="local":
    ds_val = load_dataset(path=config["local_validation_path"],data_files={"train":config["local_validation_files"]})
    ds_val = ds_val["train"]


tokenized_datasets = DatasetDict(
    {
        "train": ds_train.shuffle(seed=config["seed_num"]).map(tokenize_train, batched=True, remove_columns=ds_train.column_names),
        "val": ds_val.map(tokenize_val, batched=True, remove_columns=ds_val.column_names),
    }
)


tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"


data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False,seed=config["seed_num"])


model_config = AutoConfig.from_pretrained(
    config["model_config_path"],
    vocab_size=len(tokenizer),
    n_ctx=config["max_context_length"],
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id = tokenizer.pad_token_id
)


if hasattr(model_config,"torch_dtype"):
    config["model_dtype"] = model_config.torch_dtype
if hasattr(model_config,"dtype"):
    config["model_dtype"] = model_config.dtype
else:
    config["model_dtype"] = "bfloat16"


def create_model():
    return AutoModelForCausalLM.from_config(model_config,dtype=config["model_dtype"])

args = TrainingArguments(
    output_dir= f"{config['output_directory']}/{config['name']}_seed{config['seed_num']}",
    per_device_train_batch_size=config["batch_size"],
    per_device_eval_batch_size=config["batch_size"],
    eval_strategy="steps",
    logging_steps=config["model_save_step"],
    eval_steps=config["model_save_step"],
    gradient_accumulation_steps=config["gradient_accumulation"],
    weight_decay=0.1,
    warmup_steps=100,
    lr_scheduler_type=config["lr_scheduler"],
    learning_rate=6e-4,
    logging_strategy="steps",
    save_strategy="steps",
    save_steps=config["model_save_step"],
    max_steps=config["maximum_save_steps"],
    fp16=(config["model_dtype"]=="float16"),
    bf16=(config["model_dtype"]=="bfloat16"),
    push_to_hub=False,
    save_only_model=True,
    seed = config["seed_num"]
)

trainer = Trainer(
    model_init=create_model,
    processing_class=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["val"],
)

trainer.train()