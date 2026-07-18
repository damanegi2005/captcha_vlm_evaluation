#폴더명이 숫자로 시작하면 bounding box(exact match, iou, latency기록), 아니면 문자열 정확도(accuracy, latency).
#px:prediction, gx:ground truth
#iou: intersection over union, 겹치는 영역 넓이/합집합 영역 넓이
import os, base64, pandas as pd, re
from openai import OpenAI
import time 
import re

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
        max_tokens=50
    )

    return response.choices[0].message.content.strip()

def iou(pred, gt): #여러개의 bounding box를 처리할 수 있도록 수정. pred, gt는 문자열로 입력됨. 

    pred_nums = list(map(
        float,
        re.findall(r"-?\d+\.?\d*", pred)
    ))

    gt_nums = list(map(
        float,
        re.findall(r"-?\d+\.?\d*", gt)
    ))

    if len(pred_nums) < 4 or len(gt_nums) < 4:
        return -1

    pred_boxes = [ #아까와달리 박스 여러개 처리함. 박스 여러개 생김.
        pred_nums[i:i+4]
        for i in range(0, len(pred_nums), 4)
        if len(pred_nums[i:i+4]) == 4
    ]

    gt_boxes = [
        gt_nums[i:i+4]
        for i in range(0, len(gt_nums), 4)
        if len(gt_nums[i:i+4]) == 4
    ]

    n = min(len(pred_boxes), len(gt_boxes))

    if n == 0:
        return -1

    total_iou = 0

    for i in range(n): #박스 여러개니까 iou 여러개 계산.

        px1, py1, px2, py2 = pred_boxes[i]
        gx1, gy1, gx2, gy2 = gt_boxes[i]

        px1, px2 = min(px1, px2), max(px1, px2) #좌표 순서 수정.
        py1, py2 = min(py1, py2), max(py1, py2)

        gx1, gx2 = min(gx1, gx2), max(gx1, gx2)
        gy1, gy2 = min(gy1, gy2), max(gy1, gy2)

        inter_x1 = max(px1, gx1)
        inter_y1 = max(py1, gy1)
        inter_x2 = min(px2, gx2)
        inter_y2 = min(py2, gy2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)

        inter_area = inter_w * inter_h

        pred_area = (px2 - px1) * (py2 - py1)
        gt_area = (gx2 - gx1) * (gy2 - gy1)

        union_area = pred_area + gt_area - inter_area

        box_iou = (
            inter_area / union_area
            if union_area > 0
            else 0
        )

        total_iou += box_iou #계속 더해.

    return total_iou / n #평균 반환

results = []

for folder in os.listdir("4. click order"):

    folder_path = os.path.join(
        "4. click order",
        folder
    )

    if not os.path.isdir(folder_path):
        continue

    files = os.listdir(folder_path)

    image = next(
        os.path.join(folder_path,f)
        for f in files
        if f.endswith((".png",".jpg",".jpeg"))
    )

    txt = next(
        os.path.join(folder_path,f)
        for f in files
        if f.endswith(".txt")
    )

    with open(txt,"r",encoding="utf-8") as f:
        lines = f.readlines()

    # ==========================
    # TYPE 1 : Coordinate
    # ==========================

    if folder[0].isdigit():

        prompt = (
            lines[1].strip()
            + "\n\nOutput only the bounding box coordinates."
            + "\nFormat:"
            + "\n(x1,y1,x2,y2),(x1,y1,x2,y2)"
            + "\nDo not explain."
        )

        answer = lines[3].strip()

        # STEP 1

        start1 = time.time()

        gpt1 = solve(
            image,
            prompt
        )

        latency1 = time.time() - start1

        iou1 = iou(
            gpt1,
            answer
        )

        correct1 = int(
            gpt1.strip()
            ==
            answer.strip()
        )

        # STEP 2

        step2_prompt = prompt + """
Identify all candidate targets first.
Determine the correct order carefully.
Compare shapes, brightness, contrast, and outlines.
Verify the final order before answering.
Output only the final answer.
"""

        start2 = time.time()

        gpt2 = solve(
            image,
            step2_prompt
        )

        latency2 = time.time() - start2

        iou2 = iou(
            gpt2,
            answer
        )

        correct2 = int(
            gpt2.strip()
            ==
            answer.strip()
        )

        iou_improvement = (
            iou2 - iou1
            if iou1 != -1 and iou2 != -1
            else None
        )

        print("\n----------------")
        print("Folder :", folder)

        print("\nSTEP 1")
        print("GPT :", gpt1)
        print("IOU :", iou1)
        print("Correct :", correct1)

        print("\nSTEP 2")
        print("GPT :", gpt2)
        print("IOU :", iou2)
        print("Correct :", correct2)

        print(
            "IOU Improvement :",
            iou_improvement
        )

        results.append({
            "folder": folder,
            "type": "coordinate",

            "answer": answer,

            "gpt_step1": gpt1,
            "correct_step1": correct1,
            "iou_step1": iou1,
            "latency_step1": round(latency1,2),

            "gpt_step2": gpt2,
            "correct_step2": correct2,
            "iou_step2": iou2,
            "latency_step2": round(latency2,2),

            "iou_improvement":
                iou_improvement
        })

    # ==========================
    # TYPE 2 : String
    # ==========================

    else:

        prompt = (
            "Read the characters from left to right."
            "\nOutput only the final string."
            "\nNo spaces."
            "\nDo not explain."
        )

        answer = lines[0].strip()

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

        # STEP 2

        step2_prompt = prompt + """
Identify each character carefully.
Read them strictly from left to right.
Check ambiguous characters before answering.
Output only the final string.
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

        improvement = (
            correct2
            -
            correct1
        )

        print("\n----------------")
        print("Folder :", folder)

        print("\nSTEP 1")
        print("GPT :", gpt1)
        print("Correct :", correct1)

        print("\nSTEP 2")
        print("GPT :", gpt2)
        print("Correct :", correct2)

        print(
            "Improvement :",
            improvement
        )

        results.append({
            "folder": folder,
            "type": "string",

            "answer": answer,

            "gpt_step1": gpt1,
            "correct_step1": correct1,
            "latency_step1": round(latency1,2),

            "gpt_step2": gpt2,
            "correct_step2": correct2,
            "latency_step2": round(latency2,2),

            "improvement":
                improvement
        })

df = pd.DataFrame(results)

df.to_excel(
    "click_order_results.xlsx",
    index=False
)

print(
    "\nResults saved to click_order_results.xlsx"
)