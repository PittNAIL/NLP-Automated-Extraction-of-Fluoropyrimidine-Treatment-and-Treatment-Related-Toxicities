import os
import pandas as pd
import re
from ollama import Client
from sklearn.metrics import classification_report
import requests

# Check Ollama GPU status
def check_ollama_gpu():
    try:
        response = requests.get("http://localhost:11434/api/ps")
        if response.status_code == 200:
            models = response.json().get('models', [])
            print("Ollama running models:")
            for model in models:
                print(f"  - {model.get('name', 'Unknown')}")
        
        # Check if GPU is being used by looking at system info
        response = requests.get("http://localhost:11434/api/version")
        if response.status_code == 200:
            print(f"Ollama server is running")

        else:
            print("Ollama server not responding")
    except Exception as e:
        print(f"Error checking Ollama status: {e}")

# Add this at the beginning of your script
print("Checking Ollama GPU status...")
check_ollama_gpu()

TAGS = [
    'handfootpreventative',
    'cartoxarrhythmia',
    'cartoxheartfailure',
    'cartoxvalvularcomplications',
    'drugsofinterest'
]
SPLIT_DIR = r"D:\\Github\\MedSDoH\\data\\train_cape\\split_datasets"
MODEL_DIR = "llama_zero_shot_new_prompts_examples_val"
os.makedirs(MODEL_DIR, exist_ok=True)
client = Client(host="http://localhost:11434")
MODEL = "llama3.1:8b" 

def zero_shot_llama(tag):
    if tag == 'handfootpreventative':
        val_df = pd.read_csv(os.path.join(SPLIT_DIR, tag, 'val.csv'))
        prompt0 = f"""You are given a sentence from a clinical text, if that sentence contains any information related to the use of topical creams, respond with yes and explain why. 
        If not, respond with no and explain why. Always start your response with yes or no. These words are examples of topical creams :  
        cream, urea cream, Aqua Care, Nutraplus, Vanamide, Carbamide, Elaqua XX, Lanaphilic, Ureaphil, Carbamide, Utterly smooth, Udderly smooth  
        cream, lotion, gel, ointment, salve, solution, suspension uridine triacetate (Vistogard). 
        If these words are mentioned in the sentence, respond with yes and explain why.
        If these words are not mentioned in the sentence, respond with no and explain why."""
        y_true = val_df["target"].tolist()
        y_pred = []
        llm_outputs = []
        index=0
        for sentence in val_df["sentence"]:
            print(index)
            index+=1
            response = client.chat(model=MODEL, messages=[
                {"role": "user", "content": "Here is the sentence:"+sentence+prompt0}
                ],options={"temperature": 1})
            llm_outputs.append(response['message']['content'].strip())
            print(response['message']['content'].strip())
            if re.search(r'\byes\b', response['message']['content'].strip(), re.IGNORECASE):
                y_pred.append(1)
            elif re.search(r'\bno\b', response['message']['content'].strip(), re.IGNORECASE):
                y_pred.append(0)
            else:
                y_pred.append(0)
            print(y_pred[-1])

        val_df["llm_output"] = llm_outputs
        val_df["y_pred"] = y_pred
        val_df.to_csv(os.path.join(MODEL_DIR, f"{tag}_llama_zero_shot_run_1.csv"), index=False)
        print(classification_report(y_true, y_pred, digits=4))

        with open(os.path.join(MODEL_DIR, f'{tag}_llama_zero_shotreport.txt'), 'a') as f:
            f.write(classification_report(y_true, y_pred, digits=4))
            f.write("\n\n")
        
    elif tag == 'cartoxarrhythmia':
        prompt1 = f"""You are given a sentence from a clinical text, if that sentence contains any information related to instances of arrhythmias, respond with yes and explain why. 
        If not, respond with no and explain why. These words are examples of arrhythmias:  
        cardiac arrhythmia, dysrhythmia, irregular heartbeat, heart rhythm disturbance, cardiac rhythm disorder  
        Afib, A fib, Atrial flutter, Auricular flutter, A-flutter, AF, auricular fibrillation  
        VF, ventricular fibrillation, cardiac arrest due to VF  
        ventricular tachycardia (v tach), tachycardia;ventricular  
        flutter atrial, fibrillation atrial, flutter auricular, heart arrhythmia.
        If these words are mentioned in the sentence, respond with yes and explain why.
        If these words are not mentioned in the sentence, respond with no and explain why."""
        val_df = pd.read_csv(os.path.join(SPLIT_DIR, tag, 'val.csv'))
        y_true = val_df["target"].tolist()
        y_pred = []
        llm_outputs = []
        index=0
        for sentence in val_df["sentence"]:
            print(index)
            index+=1
            response = client.chat(model=MODEL, messages=[
                {"role": "user", "content": "Here is the sentence:"+sentence+prompt1}
                ],options={"temperature": 1})
            llm_outputs.append(response['message']['content'].strip())
            print(response['message']['content'].strip())
            if re.search(r'\byes\b', response['message']['content'].strip(), re.IGNORECASE):
                y_pred.append(1)
            elif re.search(r'\bno\b', response['message']['content'].strip(), re.IGNORECASE):
                y_pred.append(0)
            else:
                y_pred.append(0)
            print(y_pred[-1])

        val_df["llm_output"] = llm_outputs
        val_df["y_pred"] = y_pred
        val_df.to_csv(os.path.join(MODEL_DIR, f"{tag}_llama_zero_shot_run_4.csv"), index=False)
        print(classification_report(y_true, y_pred, digits=4))

        with open(os.path.join(MODEL_DIR, f'{tag}_llama_zero_shotreport.txt'), 'a') as f:
            f.write(classification_report(y_true, y_pred, digits=4))
            f.write("\n\n")

    elif tag == 'cartoxheartfailure':
        prompt2 = f"""You are given a sentence from a clinical text, if that sentence contains any information related to instances of heart failure, respond with yes and explain why. 
        If not, respond with no and explain why. These words being the signs and evidence::  
        HF, cardiac failure, heart insufficiency, myocardial failure, cardiac insufficiency  
        bilateral leg edema, swelling, dropsy, hydrops, oedema, fluid overload  
        reduced ejection fraction (EF or LVEF), reduced LV function  
        cardiogenic shock, heart shock, cardiovascular collapse, HF exacerbation.
        If these words are mentioned in the sentence, respond with yes and explain why.
        If these words are not mentioned in the sentence, respond with no and explain why."""
        val_df = pd.read_csv(os.path.join(SPLIT_DIR, tag, 'val.csv'))
        y_true = val_df["target"].tolist()
        y_pred = []
        llm_outputs = []
        index=0
        for sentence in val_df["sentence"]:
            print(index)
            index+=1
            response = client.chat(model=MODEL, messages=[
                {"role": "user", "content": "Here is the sentence:"+sentence+prompt2}
                ],options={"temperature": 1})
            llm_outputs.append(response['message']['content'].strip())
            print(response['message']['content'].strip())
            if re.search(r'\byes\b', response['message']['content'].strip(), re.IGNORECASE):
                y_pred.append(1)
            elif re.search(r'\bno\b', response['message']['content'].strip(), re.IGNORECASE):
                y_pred.append(0)
            else:
                y_pred.append(0)
            print(y_pred[-1])

        val_df["llm_output"] = llm_outputs
        val_df["y_pred"] = y_pred
        val_df.to_csv(os.path.join(MODEL_DIR, f"{tag}_llama_zero_shot_run_4.csv"), index=False)
        print(classification_report(y_true, y_pred, digits=4))

        with open(os.path.join(MODEL_DIR, f'{tag}_llama_zero_shotreport.txt'), 'a') as f:
            f.write(classification_report(y_true, y_pred, digits=4))
            f.write("\n\n")
    elif tag == 'cartoxvalvularcomplications':
        prompt3 = f"""You are given a sentence, if that sentence contains any information related to instances of valvular complications, respond with yes and explain why. 
        If not, respond with no and explain why. These words being the signs and evidence:  
        TR, tricuspid insufficiency, incompetence, right AV valve regurgitation  
        AR, aortic regurgitation, aortic incompetence, aortic valve insufficiency  
        valve disorder, AV valve abnormality, valvular dysfunction.
        If these words are mentioned in the sentence, respond with yes and explain why.
        If these words are not mentioned in the sentence, respond with no and explain why. """
        val_df = pd.read_csv(os.path.join(SPLIT_DIR, tag, 'val.csv'))
        y_true = val_df["target"].tolist()
        y_pred = []
        llm_outputs = []
        index=0
        for sentence in val_df["sentence"]:
            print(index)
            index+=1
            response = client.chat(model=MODEL, messages=[
                {"role": "user", "content": "Here is the sentence:"+sentence+prompt3}
                ],options={"temperature": 1})
            llm_outputs.append(response['message']['content'].strip())
            print(response['message']['content'].strip())
            if re.search(r'\byes\b', response['message']['content'].strip(), re.IGNORECASE):
                y_pred.append(1)
            elif re.search(r'\bno\b', response['message']['content'].strip(), re.IGNORECASE):
                y_pred.append(0)
            else:
                y_pred.append(0)
            print(y_pred[-1])

        val_df["llm_output"] = llm_outputs
        val_df["y_pred"] = y_pred
        val_df.to_csv(os.path.join(MODEL_DIR, f"{tag}_llama_zero_shot_run_4.csv"), index=False)
        print(classification_report(y_true, y_pred, digits=4))

        with open(os.path.join(MODEL_DIR, f'{tag}_llama_zero_shotreport.txt'), 'a') as f:
            f.write(classification_report(y_true, y_pred, digits=4))
            f.write("\n\n")
    elif tag == 'drugsofinterest':
        prompt4 = f"""You are given a sentence, if that sentence directly or indirectly mentions capecitabine, which contains 5-FU, or directly or indirectly mentions other drugs that also contains 5-FU, respond with yes and explain why. 
        If not, respond with no and explain why. These words being the signs and evidence:  
        Capecitabine, Xeloda, Xitabin  
        5-FU, 5-Fluorouracil, Fluoro Uracil, Adrucil, Carac, Flurablastin  
        CAPOX, CAPIRI, CAPEOX, CAPEMONO, FOLFOX, FOLFIRI, FOLFOXIRI, MFOLFOX, AIO, De Gramont regimen, XELOX, XELIRI, FOLFIRINOX.
        If these words are mentioned in the sentence, respond with yes and explain why.
        If these words are not mentioned in the sentence, respond with no and explain why."""
        val_df = pd.read_csv(os.path.join(SPLIT_DIR, tag, 'val.csv'))
        y_true = val_df["target"].tolist()
        y_pred = []
        llm_outputs = []
        index=0
        for sentence in val_df["sentence"]:
            print(index)
            index+=1
            response = client.chat(model=MODEL, messages=[
                {"role": "user", "content": "Here is the sentence:"+sentence+prompt4}
                ],options={"temperature": 1})
            llm_outputs.append(response['message']['content'].strip())
            print(response['message']['content'].strip())
            if re.search(r'\byes\b', response['message']['content'].strip(), re.IGNORECASE):
                y_pred.append(1)
            elif re.search(r'\bno\b', response['message']['content'].strip(), re.IGNORECASE):
                y_pred.append(0)
            else:
                y_pred.append(0)
            print(y_pred[-1])

        val_df["llm_output"] = llm_outputs
        val_df["y_pred"] = y_pred
        val_df.to_csv(os.path.join(MODEL_DIR, f"{tag}_llama_zero_shot_run_1.csv"), index=False)
        print(classification_report(y_true, y_pred, digits=4))

        with open(os.path.join(MODEL_DIR, f'{tag}_llama_zero_shotreport.txt'), 'a') as f:
            f.write(classification_report(y_true, y_pred, digits=4))
            f.write("\n\n")
if __name__ == "__main__":
    zero_shot_llama(TAGS[2])
    zero_shot_llama(TAGS[3])
    zero_shot_llama(TAGS[4])
