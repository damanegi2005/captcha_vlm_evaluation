import pandas as pd
from openai import OpenAI
import time

client = OpenAI(api_key="")

df = pd.read_csv("6.Common_sense_Reasoning/logical_qa_pairs.csv")

results = []

for i in range(len(df)): #df는 행갯수. 전체문제개수.

    question = str(df.iloc[i]["question"]) #i번째행의 question열 가져온다,
    answer = str(df.iloc[i]["answer"]).strip().lower()

    if answer == "true":
        answer = "yes"
    elif answer == "false":
        answer = "no"

    # STEP 1

    start1 = time.time()

    response1 = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{
            "role":"user",
            "content":
            question +
            "\n\nAnswer only Yes or No."
        }]
    )

    latency1 = time.time() - start1

    gpt1 = (
        response1
        .choices[0]
        .message
        .content
        .strip()
        .lower()
    )

    correct1 = int(
        gpt1 == answer
    )

    # STEP 2

    step2_prompt = (
        question
        + "\n\nRead the question carefully."
        + "\nConsider all conditions before answering."
        + "\nVerify your reasoning."
        + "\nAnswer only Yes or No."
    )

    start2 = time.time()

    response2 = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{
            "role":"user",
            "content": step2_prompt
        }]
    )

    latency2 = time.time() - start2

    gpt2 = (
        response2
        .choices[0]
        .message
        .content
        .strip()
        .lower()
    )

    correct2 = int(
        gpt2 == answer
    )

    improvement = (
        correct2
        -
        correct1
    )

    print("\n----------------")
    print("Question :", question)

    print("\nSTEP 1")
    print("GPT :", gpt1)
    print("Correct :", correct1)
    print("Latency :", round(latency1,2))

    print("\nSTEP 2")
    print("GPT :", gpt2)
    print("Correct :", correct2)
    print("Latency :", round(latency2,2))

    print(
        "Improvement :",
        improvement
    )

    results.append({

        "question": question,
        "answer": answer,

        "gpt_step1": gpt1,
        "correct_step1": correct1,
        "latency_step1": round(latency1,2),

        "gpt_step2": gpt2,
        "correct_step2": correct2,
        "latency_step2": round(latency2,2),

        "improvement": improvement
    })

df_result = pd.DataFrame(results)

step1_acc = df_result["correct_step1"].mean()
step2_acc = df_result["correct_step2"].mean()

print("\n====================")
print("Step1 Accuracy :", round(step1_acc,3))
print("Step2 Accuracy :", round(step2_acc,3))
print("Accuracy Improvement :", round(step2_acc-step1_acc,3))

df_result.to_csv(
    "common_sense_results.csv",
    index=False
)

print("\nResults saved to common_sense_results.csv")