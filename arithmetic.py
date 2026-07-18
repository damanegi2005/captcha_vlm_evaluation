import os, base64, pandas as pd
from openai import OpenAI
import time

client = OpenAI(api_key="")

def encode(path):
    with open(path,"rb") as f:
        return base64.b64encode(f.read()).decode()

def solve(image,prompt):

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
        max_tokens=30
    )

    return response.choices[0].message.content.strip()

results=[]

for folder in os.listdir("7.Arithmetic"):

    folder_path=os.path.join(
        "7.Arithmetic",
        folder
    )

    if not os.path.isdir(folder_path):
        continue

    files=os.listdir(folder_path)

    image=next(
        os.path.join(folder_path,f)
        for f in files
        if f.endswith(
            (".png",".jpg",".jpeg")
        )
    )

    txt=next(
        os.path.join(folder_path,f)
        for f in files
        if f.endswith(".txt")
    )

    with open(
        txt,
        "r",
        encoding="utf-8"
    ) as f:

        lines=f.readlines()

    prompt=(
        lines[1].strip()
        + "\nOutput only the answer."
        + "\nIf there are multiple numbers, separate them with commas."
        + "\nDo not use spaces."
        + "\nSort in descending order."
        + "\nExample: 23,6,5"
        + "\nDo not explain."
    )

    answer=lines[2].strip()

    # STEP 1

    start1=time.time()

    gpt1=solve(
        image,
        prompt
    )

    latency1=time.time()-start1

    correct1=int(
        gpt1.strip()
        ==
        answer.strip()
    )

    # STEP 2

    step2_prompt=prompt+"""
Identify all numbers and symbols.
Solve the problem step by step.
Verify the calculation carefully.
Output only the final answer.
"""

    start2=time.time()

    gpt2=solve(
        image,
        step2_prompt
    )

    latency2=time.time()-start2

    correct2=int(
        gpt2.strip()
        ==
        answer.strip()
    )

    improvement=(
        correct2
        -
        correct1
    )

    print("\n----------------")
    print("Folder :",folder)

    print("\nSTEP 1")
    print("GPT :",gpt1)
    print("Correct :",correct1)
    print("Latency :",round(latency1,2))

    print("\nSTEP 2")
    print("GPT :",gpt2)
    print("Correct :",correct2)
    print("Latency :",round(latency2,2))

    print(
        "Accuracy Improvement :",
        improvement
    )

    results.append({

        "folder":folder,
        "answer":answer,

        "gpt_step1":gpt1,
        "correct_step1":correct1,
        "latency_step1":round(latency1,2),

        "gpt_step2":gpt2,
        "correct_step2":correct2,
        "latency_step2":round(latency2,2),

        "improvement":improvement

    })

df=pd.DataFrame(results)

step1_acc=df["correct_step1"].mean()
step2_acc=df["correct_step2"].mean()

print("\n====================")
print("Step1 Accuracy :",round(step1_acc,3))
print("Step2 Accuracy :",round(step2_acc,3))
print(
    "Accuracy Improvement :",
    round(
        step2_acc-step1_acc,
        3
    )
)

df.to_excel(
    "arithmetic_results.xlsx",
    index=False
)

print(
    "\nResults saved to arithmetic_results.xlsx"
)