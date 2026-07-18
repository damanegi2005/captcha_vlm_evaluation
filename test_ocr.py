#마지막엔 결과를 하나의 df로 통합, excel로 저장.
#문자열 accuracy
import os, base64, pandas as pd
from openai import OpenAI
import time

client = OpenAI(api_key="")

def encode(path): #이미지파일을 base64로 변환.
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def solve(image, prompt):

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{
            "role":"user",
            "content":[
                {"type":"text","text":prompt}, #텍스트 프롬프트와 이미지 입력
                {
                    "type":"image_url",
                    "image_url":{
                        "url":f"data:image/png;base64,{encode(image)}"
                    }
                }
            ]
        }],
        max_tokens=20
    )

    return response.choices[0].message.content.strip()

def char_accuracy(pred, gt): #문자별 맞는지 

    correct_chars = 0

    for p, g in zip(pred, gt): #zip: 같은 위치의 문자끼리 묶음
        if p == g:
            correct_chars += 1

    return correct_chars / len(gt)

results = []

for folder in os.listdir("1.OCR"):

    folder_path = os.path.join(
        "1.OCR",
        folder
    )

    if not os.path.isdir(folder_path):
        continue

    files = os.listdir(folder_path)

    image = next(
        os.path.join(folder_path, f)
        for f in files
        if f.endswith((".png", ".jpg", ".jpeg"))
    )

    txt = next(
        os.path.join(folder_path, f)
        for f in files
        if f.endswith(".txt")
    )

    with open(
        txt,
        "r",
        encoding="utf-8"
    ) as f:

        lines = f.readlines()

    prompt = (
        lines[1].strip()
        + "\nOutput only the exact text."
    )

    answer = lines[2].strip()

    # STEP 1

    start1 = time.time()

    gpt1 = solve(
        image,
        prompt
    )

    latency1 = time.time() - start1

    correct1 = int(
        gpt1.strip()
        ==
        answer.strip()
    )

    char_acc1 = char_accuracy(
        gpt1.strip(),
        answer.strip()
    )

    # STEP 2. self verification prompting

    step2_prompt = prompt + """ 
Carefully identify all letters and numbers.
Check ambiguous characters before answering.
Verify the final text.
Output only the final answer.
"""

    start2 = time.time()

    gpt2 = solve(
        image,
        step2_prompt
    )

    latency2 = time.time() - start2

    correct2 = int(
        gpt2.strip()
        ==
        answer.strip()
    )

    char_acc2 = char_accuracy(
        gpt2.strip(),
        answer.strip()
    )

    char_improvement = (
        char_acc2
        -
        char_acc1
    )

    print("\n----------------")
    print("Folder :", folder)

    print("Answer :", answer)

    print("\nSTEP 1")
    print("GPT :", gpt1)
    print("Correct :", correct1)
    print("Character Accuracy :", round(char_acc1,3))
    print("Latency :", round(latency1,2),"sec")

    print("\nSTEP 2")
    print("GPT :", gpt2)
    print("Correct :", correct2)
    print("Character Accuracy :", round(char_acc2,3))
    print("Latency :", round(latency2,2),"sec")

    print(
        "\nCharacter Improvement :",
        round(char_improvement,3)
    )

    results.append({

        "folder": folder,
        "answer": answer,

        "gpt_step1": gpt1,
        "correct_step1": correct1,
        "char_acc_step1": round(char_acc1,3),
        "latency_step1": round(latency1,2),

        "gpt_step2": gpt2,
        "correct_step2": correct2,
        "char_acc_step2": round(char_acc2,3),
        "latency_step2": round(latency2,2),

        "char_improvement":
            round(char_improvement,3)

    })

df = pd.DataFrame(results)

step1_acc = df["correct_step1"].mean() #각샘플의 exact match 평균계산.
step2_acc = df["correct_step2"].mean()

step1_char = df["char_acc_step1"].mean()
step2_char = df["char_acc_step2"].mean()

print("\n====================")

print(
    "Step1 Accuracy :",
    round(step1_acc,3)
)

print(
    "Step2 Accuracy :",
    round(step2_acc,3)
)

print(
    "Accuracy Improvement :",
    round(
        step2_acc-step1_acc,
        3
    )
)

print()

print(
    "Step1 Character Accuracy :",
    round(step1_char,3)
)

print(
    "Step2 Character Accuracy :",
    round(step2_char,3)
)

print(
    "Character Accuracy Improvement :",
    round(
        step2_char-step1_char,
        3
    )
)

df.to_excel(
    "ocr_results.xlsx",
    index=False
)

print(
    "\nResults saved to ocr_results.xlsx"
)