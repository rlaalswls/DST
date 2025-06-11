import argparse
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch

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
    
    print(input_text)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"result: {result}")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="방언 → 표준어 변환기")
    parser.add_argument("--input_text", type=str, required=True, help="방언 문장")

    args = parser.parse_args()
    myDST(args.input_text, model_dir="/dst/code/results")
