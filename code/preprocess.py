import json
import os
import random
import re
from sklearn.model_selection import train_test_split

# 전처리 함수
def extract_second_option_or_remove(text):
    if "#" in text or "&" in text or "@" in text:
        return None

    def replace(match):
        options = re.split(r'/|\\', match.group(1))
        return options[1] if len(options) > 1 else options[0]

    text = re.sub(r'\(([^)]+)\)', replace, text)
    return re.sub(r'\s+', ' ', text).strip()

# 경로 설정
dialect_folders = {
    "jeju": "/dst/jeju",
    "gyeongsang": "/dst/gyeongs",
    "jeolla": "/dst/jeolla",
}

dialect_pairs = {
    "jeju": [],
    "gyeongsang": [],
    "jeolla": [],
}

MAX_PAIRS = 60000 #상한선
duplicate_count = 0

# 데이터 수집
for domain_name, input_dir in dialect_folders.items():
    for file_name in os.listdir(input_dir):
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(input_dir, file_name)
        with open(file_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)

        for utt in data.get("utterance", []):
            # 수집 개수 제한
            if len(dialect_pairs[domain_name]) >= MAX_PAIRS:
                break

            dialect = extract_second_option_or_remove(utt.get("dialect_form", "").strip())
            standard = extract_second_option_or_remove(utt.get("standard_form", "").strip())

            if dialect and standard:
                domain_free_input = dialect.strip()
                if domain_free_input == standard:
                    duplicate_count += 1
                else:
                    input_text = f"{domain_name}: {dialect}"
                    target_text = standard
                    dialect_pairs[domain_name].append({
                        "input": input_text,
                        "target": target_text
                    })

# 전체 합치기
all_pairs = []
for domain, pairs in dialect_pairs.items():
    all_pairs.extend(pairs)

print(f"총 수집된 데이터: {len(all_pairs)}쌍")
#print(f"도메인 태그 제거 후 input == target인 경우: {duplicate_count}쌍")

# 8:1:1 비율 계산
train_size = int(len(all_pairs) * 0.8)
valid_size = int(train_size * 0.125)  # valid:test = 1:1이므로 0.8*0.125 = 0.1
test_size = valid_size

# 데이터 셔플
random.shuffle(all_pairs)
train_data = all_pairs[:train_size]
valid_data = all_pairs[train_size:train_size + valid_size]
test_data = all_pairs[train_size + valid_size:train_size + valid_size + test_size]

# 저장 함수
def save_jsonl(path, data):
    with open(path, 'w', encoding='utf-8-sig') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

save_jsonl("train.jsonl", train_data)
save_jsonl("valid.jsonl", valid_data)
save_jsonl("test.jsonl", test_data)

print(f"train: {len(train_data)}개, valid: {len(valid_data)}개, test: {len(test_data)}개 저장 완료")
