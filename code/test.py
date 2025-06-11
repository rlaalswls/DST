from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer

from datasets import load_from_disk
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

def myDST(input_text, model_dir="/dst/code/results"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=64,
        num_beams=1,
        do_sample=False
    )
    
    print("output_ids:", outputs)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"result: {result}")
    return result


# 1. 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("/dst/code/results")
model = T5ForConditionalGeneration.from_pretrained("/dst/code/results")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 2. 데이터 로드
tokenized_datasets = load_from_disk("/dst/code/tokenized_datasets")
test_dataset = tokenized_datasets["test"]

# 3. BLEU 계산 준비
smoothie = SmoothingFunction().method4
bleu_scores = []

# Accuracy 계산 준비
correct_predictions = 0

# 4. 테스트셋에 대해 generate 및 BLEU, Accuracy 계산
for i in range(len(test_dataset)):
    sample = test_dataset[i]

    input_ids = sample['input_ids'].unsqueeze(0).to(device)
    attention_mask = sample['attention_mask'].unsqueeze(0).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=64,
            num_beams=5,
            do_sample=False
        )

    pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    label_ids = [id for id in sample['labels'] if id != -100]
    label = tokenizer.decode(label_ids, skip_special_tokens=True)

    # BLEU 계산
    bleu = sentence_bleu([label.split()], pred.split(), smoothing_function=smoothie)
    bleu_scores.append(bleu)

    # Accuracy 계산
    if pred.strip() == label.strip():
        correct_predictions += 1

    if (i + 1) % 50 == 0 or i == len(test_dataset) - 1:
        avg_bleu = np.mean(bleu_scores)
        accuracy = correct_predictions / (i + 1)
        print(f"Processed {i + 1}/{len(test_dataset)} | Avg BLEU: {avg_bleu:.4f} | Accuracy: {accuracy:.4f}")

# 5. 전체 BLEU 및 Accuracy 출력
final_bleu = np.mean(bleu_scores)
final_accuracy = correct_predictions / len(test_dataset)

print(f"\n전체 Average BLEU score: {final_bleu:.4f}")
print(f"전체 Accuracy: {final_accuracy:.4f}")
