from transformers import T5Tokenizer
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("wisenut-nlp-team/KoT5-base")
import preprocess
from datasets import load_dataset
import torch

# 각 split 불러오기
data_files = {
    "train": "train.jsonl",
    "validation": "valid.jsonl",
    "test": "test.jsonl"
}
datasets = load_dataset("json", data_files=data_files)

# 각 입력 문장의 토큰 길이를 확인해보기
lengths = [len(tokenizer.encode(text)) for text in datasets["train"]["input"]]
print(f"최대 길이: {max(lengths)}")
print(f"평균 길이: {sum(lengths) / len(lengths):.2f}")
print(f"상위 10개 길이: {sorted(lengths, reverse=True)[:10]}")


# 전처리 함수
def preprocess_function(examples):
    inputs = examples["input"]
    targets = examples["target"]

    # 입력 토큰화
    model_inputs = tokenizer(
        inputs,
        max_length=64,
        padding="max_length",
        truncation=True,
    )

    # 타겟 토큰화
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=64,
            padding="max_length",
            truncation=True,
        )

    # pad_token_id → -100으로 마스킹
    labels_ids = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels_ids
    return model_inputs

# 확인용 코드 ---------

sample_batch = datasets["train"][:4]  # 예: 4개 샘플 batch
print(sample_batch["input"][0])  # 이 input_ids가 어떤 문장에서 나왔는지
processed = preprocess_function(sample_batch)
#print(processed)
for k, v in processed.items():
    print(f"{k}: {v[:10]}...")
# ------

# map을 통해 전체 split에 전처리 적용
tokenized_datasets = datasets.map(preprocess_function, batched=True)

# torch tensor 형식으로 설정
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments

model = AutoModelForSeq2SeqLM.from_pretrained("wisenut-nlp-team/KoT5")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset = tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

trainer.train()

model.save_pretrained("/dst/code/results")
tokenizer.save_pretrained("/dst/code/results")

# 전처리된 데이터셋 저장
tokenized_datasets.save_to_disk("./tokenized_datasets")
