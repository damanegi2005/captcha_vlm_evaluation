#문자선택유형->accuracy, 좌표예측유형->iou
import os, base64, pandas as pd, re
from openai import OpenAI
import time 

client = OpenAI(api_key="")
def encode(path):
    with open(path,"rb") as f: return base64.b64encode(f.read()).decode()

def solve(image,prompt):
    response = client.chat.completions.create(model="gpt-4.1-mini",messages=[{"role":"user","content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":f"data:image/png;base64,{encode(image)}"}}]}],max_tokens=50)
    return response.choices[0].message.content.strip()

def iou(pred, gt): #예측한, 정답. intersection over union. bounding box

    p = list(map(float, re.findall(r"-?\d+\.?\d*", pred))) #문자열 속 숫자 추출 후 float로 변환 
    g = list(map(float, re.findall(r"-?\d+\.?\d*", gt)))

    if len(p) < 4 or len(g) < 4: #네개 좌표 확실?
        return -1

    px1, py1, px2, py2 = p[:4] #각각 변수에 저장
    gx1, gy1, gx2, gy2 = g[:4]

    inter_x1 = max(px1, gx1) #겹치는 좌표 계산
    inter_y1 = max(py1, gy1)
    inter_x2 = min(px2, gx2)
    inter_y2 = min(py2, gy2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1) #겹치는 영역 넓이

    pred_area = abs((px2 - px1) * (py2 - py1)) #각각 박스넓이
    gt_area = abs((gx2 - gx1) * (gy2 - gy1))

    union_area = pred_area + gt_area - inter_area #합집합

    return inter_area / union_area if union_area > 0 else 0 #완전겹치면 1

results=[]

for folder in os.listdir("5.Rule_based_filtering"):

    folder_path=os.path.join("5.Rule_based_filtering",folder)

    if not os.path.isdir(folder_path):
        continue

    files=os.listdir(folder_path)

    image=next(os.path.join(folder_path,f) for f in files if f.endswith((".png",".jpg",".jpeg")))
    txt=next(os.path.join(folder_path,f) for f in files if f.endswith(".txt"))

    with open(txt,"r",encoding="utf-8") as f:
        lines=f.readlines()

    # =========================
    # A TYPE
    # =========================

    if folder.startswith("A"):

        prompt=lines[1].strip()+"\nOutput only the bounding box coordinates.\nFormat:(x1,y1,x2,y2)\nDo not explain."

        answer=lines[2].strip()

        start1=time.time()
        gpt1=solve(image,prompt)
        latency1=time.time()-start1

        iou1=iou(gpt1,answer)

        step2_prompt=prompt+"\nIdentify all candidate targets first.\nApply the rule carefully.\nVerify the selected target.\nOutput only the final answer."

        start2=time.time()
        gpt2=solve(image,step2_prompt)
        latency2=time.time()-start2

        iou2=iou(gpt2,answer)

        iou_improvement=(iou2-iou1) if iou1!=-1 and iou2!=-1 else None

        print("\n----------------")
        print("Folder :",folder)

        print("\nSTEP 1")
        print("GPT :",gpt1)
        print("IOU :",iou1)
        print("Latency :",round(latency1,2))

        print("\nSTEP 2")
        print("GPT :",gpt2)
        print("IOU :",iou2)
        print("Latency :",round(latency2,2))

        print("IOU Improvement :",iou_improvement)

        results.append({
            "folder":folder,
            "type":"coordinate",
            "answer":answer,

            "gpt_step1":gpt1,
            "iou_step1":iou1,
            "latency_step1":round(latency1,2),

            "gpt_step2":gpt2,
            "iou_step2":iou2,
            "latency_step2":round(latency2,2),

            "iou_improvement":iou_improvement
        })

    # =========================
    # V TYPE
    # =========================

    elif folder.startswith("v"):

        prompt=lines[0].strip()+"\nOutput only the letters.\nSeparate multiple answers with commas.\nSort alphabetically.\nExample: A,O\nDo not explain."

        answer=lines[2].strip()

        start1=time.time()
        gpt1=solve(image,prompt)
        latency1=time.time()-start1

        correct1=int(gpt1.strip()==answer.strip())

        step2_prompt=prompt+"\nIdentify all letters first.\nDetermine which letters satisfy the rule.\nVerify the selected letters carefully.\nOutput only the final answer."

        start2=time.time()
        gpt2=solve(image,step2_prompt)
        latency2=time.time()-start2

        correct2=int(gpt2.strip()==answer.strip())

        improvement=correct2-correct1

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

        print("Accuracy Improvement :",improvement)

        results.append({
            "folder":folder,
            "type":"letters",
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

letter_df=df[df["type"]=="letters"] #df을 유형별로 분리. 각각 df 생성
coord_df=df[df["type"]=="coordinate"]

if len(letter_df)>0: #문자면 걍 accurcay 사용
    step1_acc=letter_df["correct_step1"].mean()
    step2_acc=letter_df["correct_step2"].mean()

    print("\n====================")
    print("Step1 Accuracy :",round(step1_acc,3))
    print("Step2 Accuracy :",round(step2_acc,3))
    print("Accuracy Improvement :",round(step2_acc-step1_acc,3))

if len(coord_df)>0: #iou 평균 사용.
    iou1_avg=coord_df["iou_step1"].mean()
    iou2_avg=coord_df["iou_step2"].mean()

    print("\n====================")
    print("Step1 IOU :",round(iou1_avg,3))
    print("Step2 IOU :",round(iou2_avg,3))
    print("IOU Improvement :",round(iou2_avg-iou1_avg,3))

df.to_excel("rule_based_results.xlsx",index=False)

print("\nResults saved to rule_based_results.xlsx")