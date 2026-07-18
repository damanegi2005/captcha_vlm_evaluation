#f1
import os, base64, pandas as pd
from openai import OpenAI
import time 

client = OpenAI(api_key="")

def encode(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def solve(image, prompt):

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{
            "role":"user",
            "content":[
                {"type":"text","text":prompt},
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

def normalize(x): #단순 문자열 비교가 아닌 쉼표 기준으로 번호 분리하고 공백 제거 후 숫자순으로 정렬

    nums = [i.strip() for i in x.split(",")]
    nums = sorted(nums, key=int)

    return nums

def metrics(pred, gt): #평가방식

    pred = set(normalize(pred)) #번호를 집합으로 변환.
    gt = set(normalize(gt))

    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)

    precision = ( #실제 고른것중 맞은것
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = ( #실제 정답중 맞은것 
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = ( #precisi과 recall 의 조화평균
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return tp, fp, fn, precision, recall, f1

results = []

for folder in os.listdir("2.Object_Recognition"):

    folder_path = os.path.join(
        "2.Object_Recognition",
        folder
    )

    if not os.path.isdir(folder_path):
        continue

    files = os.listdir(folder_path)

    image = next(
        os.path.join(folder_path,f)
        for f in files
        if f.endswith(
            (".png",".jpg",".jpeg")
        )
    )

    txt = next(
        os.path.join(folder_path,f)
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
        + "\n\nThe grid is indexed from left to right and top to bottom."
        + "\nThe indices range from 0 to 8."
        + "\nOutput only the matching cell number(s)."
        + "\nIf multiple cells are correct, separate them with commas."
        + "\nExample: 1,4,7"
        + "\nDo not explain."
    )

    answer = lines[2].strip()

    # STEP 1

    start1 = time.time()

    gpt1 = solve(
        image,
        prompt
    )

    latency1 = (
        time.time()
        -
        start1
    )

    tp1, fp1, fn1, precision1, recall1, f1_1 = metrics(
        gpt1,
        answer
    )

    correct1 = int(
        normalize(gpt1)
        ==
        normalize(answer)
    )

    # STEP 2

    step2_prompt = ( #추론단계 명시. (객체탐지, 그리드 매핑, 결과검증)
    lines[1].strip()
    + "\n\nThe grid is a 3x3 matrix indexed from left to right and top to bottom."
    + "\nThe indices range from 0 to 8 as follows:"
    + "\n0,1,2"
    + "\n3,4,5"
    + "\n6,7,8"
    + "\nFirst identify all target objects."
    + "\nMap each object to its corresponding grid cell."
    + "\nVerify every selected cell carefully."
    + "\nOutput only the matching cell number(s)."
    + "\nIf multiple cells are correct, separate them with commas."
    + "\nExample: 1,4,7"
    + "\nDo not explain."
)

    start2 = time.time()

    gpt2 = solve(
        image,
        step2_prompt
    )

    latency2 = (
        time.time()
        -
        start2
    )

    tp2, fp2, fn2, precision2, recall2, f1_2 = metrics(
        gpt2,
        answer
    )

    correct2 = int( #동일하면 1
        normalize(gpt2)
        ==
        normalize(answer)
    )

    f1_improvement = (
        f1_2
        -
        f1_1
    )

    print("\n----------------")
    print("Folder :", folder)

    print("\nSTEP 1")
    print("GPT :", gpt1)
    print("Correct :", correct1)
    print("TP :", tp1)
    print("FP :", fp1)
    print("FN :", fn1)
    print("Precision :", round(precision1,3))
    print("Recall :", round(recall1,3))
    print("F1 :", round(f1_1,3))
    print("Latency :", round(latency1,2),"sec")

    print("\nSTEP 2")
    print("GPT :", gpt2)
    print("Correct :", correct2)
    print("TP :", tp2)
    print("FP :", fp2)
    print("FN :", fn2)
    print("Precision :", round(precision2,3))
    print("Recall :", round(recall2,3))
    print("F1 :", round(f1_2,3))
    print("Latency :", round(latency2,2),"sec")

    print(
        "\nF1 Improvement :",
        round(f1_improvement,3)
    )

    results.append({

        "folder": folder,
        "answer": answer,

        "gpt_step1": gpt1,
        "correct_step1": correct1,
        "tp_step1": tp1,
        "fp_step1": fp1,
        "fn_step1": fn1,
        "precision_step1": round(precision1,3),
        "recall_step1": round(recall1,3),
        "f1_step1": round(f1_1,3),
        "latency_step1": round(latency1,2),

        "gpt_step2": gpt2,
        "correct_step2": correct2,
        "tp_step2": tp2,
        "fp_step2": fp2,
        "fn_step2": fn2,
        "precision_step2": round(precision2,3),
        "recall_step2": round(recall2,3),
        "f1_step2": round(f1_2,3),
        "latency_step2": round(latency2,2),

        "f1_improvement":
            round(f1_improvement,3)

    })

df = pd.DataFrame(results)

step1_acc = df["correct_step1"].mean()
step2_acc = df["correct_step2"].mean()

step1_f1 = df["f1_step1"].mean()
step2_f1 = df["f1_step2"].mean()

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
    "Step1 F1 :",
    round(step1_f1,3)
)

print(
    "Step2 F1 :",
    round(step2_f1,3)
)

print(
    "F1 Improvement :",
    round(
        step2_f1-step1_f1,
        3
    )
)

df.to_excel(
    "object_results.xlsx",
    index=False
)

print(
    "\nResults saved to object_results.xlsx"
)
    