import os
import pandas as pd
import re
from ollama import Client
from sklearn.metrics import classification_report

TAGS = [
    'handfootpreventative',
    'cartoxarrhythmia',
    'cartoxheartfailure',
    'cartoxvalvularcomplications',
    'drugsofinterest'
]
SPLIT_DIR = r"D:\\Github\\MedSDoH\\data\\train_cape\\split_datasets"
# MODEL_DIR = "llama_few_shot_new_prompts_train"
MODEL_DIR = "llama_few_shot_new_prompts_val"
os.makedirs(MODEL_DIR, exist_ok=True)
client = Client(host="http://localhost:11434")
MODEL = "llama3.1:8b" 

def few_shot_llama(tag):
    if tag == 'handfootpreventative':
        val_df = pd.read_csv(os.path.join(SPLIT_DIR, tag, 'val.csv'))
        prompt0 = f"""You are given a sentence from a clinical text, if that sentence contains any information related to the use of topical creams, respond with yes and explain why. 
        If not, respond with no and explain why. Always start your response with yes or no. Analyze step by step. These words are examples of topical creams :  
        cream, urea cream, Aqua Care, Nutraplus, Vanamide, Carbamide, Elaqua XX, Lanaphilic, Ureaphil, Carbamide, Utterly smooth, Udderly smooth  
        cream, lotion, gel, ointment, salve, solution, suspension uridine triacetate (Vistogard). 
        If these words are mentioned in the sentence, respond with yes and explain why.
        If these words are not mentioned in the sentence, respond with no and explain why."""
        sentence1 = """multivitamin (DAILY MULTI-VITAMIN) oral tablet Take 1 tablet by mouth daily   
        ondansetron (ZOFRAN-ODT) 4 mg oral disintegrating tablet Take 1 tablet by mouth every 8 hours as needed for nausea or vomiting (Emesis)   
        ondansetron HCl (ZOFRAN, AS HYDROCHLORIDE,) 4 mg oral tablet Take 1 tablet by mouth every 8 hours   
        OxyCODONE (DAZIDOX) 10 mg oral tablet Take 1 tablet by mouth every 6 hours as needed for pain   
        oxyCODONE (OXYCONTIN) 20 mg oral tablet,oral only,ext.rel.12 hr Take 1 tablet by mouth every 12 hours   
        pramoxine-hydrocortisone 1-1 % rect cream Insert rectally 2 times a day    
        rosuvastatin (CRESTOR) 20 mg oral tablet Take 1 tablet by mouth daily   sertraline (ZOLOFT) 100 mg oral tablet 2 a day   
        triamcinolone (KENALOG) 0.1 % topical cream Apply to affected area 2 times a day   
        valsartan (DIOVAN) 40 mg oral tablet Take 1 tablet by mouth daily 1/2 tab a  Day       G
        IANT EAGLE #0043 - [ADDRESS] - [ADDRESS] AT   [ADDRESS]  Phone: [ALPHANUMERICID] Fax: [ALPHANUMERICID]        
        REVIEW OF SYSTEMS     Constitutional: Negative  Skin: Negative  HEENT: Negative"""
        answer1 = """Reasoning:
                    Step 1: Read the sentence and identify medication descriptions.
                    Step 2: Look for any mention of topical dosage forms such as "cream", "lotion", "gel", etc.
                    Step 3: The sentence includes “triamcinolone (KENALOG) 0.1 % topical cream Apply to affected area 2 times a day”.
                    Step 4: "Topical cream" directly matches the keyword "cream", which is included in the list of recognized topical formulations.
                    Step 5: Therefore, the sentence contains information related to the use of a topical cream.

                    Answer: Yes. The sentence explicitly mentions a topical cream, satisfying the criteria for identifying topical cream usage."""
        sentence2 = """Cardiovascular:      Rate and Rhythm: Normal rate and regular rhythm"""
        answer2 = """Reasoning:  
                    Step 1: Read the sentence carefully.  
                    Step 2: Check whether it mentions any medication, treatment, or formulation.  
                    Step 3: The sentence says “Cardiovascular: Rate and Rhythm: Normal rate and regular rhythm”.  
                    Step 4: There are no mentions of creams, lotions, gels, or any topical formulations.  
                    Step 5: This sentence focuses on cardiovascular examination findings, not medications.
                    Answer:  
                    No. The sentence does not mention any topical creams and is unrelated to topical treatment."""
        sentence3 = """His weight is stable for the most part."""
        answer3 = """Reasoning:  
                    Step 1: Read the sentence carefully.  
                    Step 2: Identify whether the sentence includes any medication, particularly topical forms like cream, lotion, etc.  
                    Step 3: The sentence says “His weight is stable for the most part”.  
                    Step 4: There is no mention of any medications or topical formulations.  
                    Step 5: The sentence is about body weight status, not treatment.

                    Answer: 
                    No. The sentence does not mention any topical creams and is unrelated to topical treatment."""
        
        y_true = val_df["target"].tolist()
        y_pred = []
        llm_outputs = []
        index=0
        for sentence in val_df["sentence"]:
            print("/n")
            print("Here is the sentence:")
            print(sentence)
            print(index)
            index+=1
            response = client.chat(model=MODEL, messages=[
                {"role": "user", "content": "Here is the sentence:"+sentence1+prompt0},
                {"role": "asistant", "content": answer1},
                {"role": "user", "content": "Here is the sentence:"+sentence2+prompt0},
                {"role": "asistant", "content": answer2},
                {"role": "user", "content": "Here is the sentence:"+sentence3+prompt0},
                {"role": "asistant", "content": answer3},
                {"role": "user", "content": "Here is the sentence:"+sentence+prompt0}
                ],options={"temperature": 0.5})
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
        val_df.to_csv(os.path.join(MODEL_DIR, f"{tag}_llama_few_shot_run_3.csv"), index=False)
        print(classification_report(y_true, y_pred, digits=4))

        with open(os.path.join(MODEL_DIR, f'{tag}_llama_few_shotreport.txt'), 'a') as f:
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
        """
        sentence4 = """Chief Complaint   Patient presents with     New Patient     eval for mediport     Past Medical History   Diagnosis Date   
        Unspecified essential hypertension    Copd (chronic obstructive pulmonary disease) (hcc) 11/10/2015   Cad (coronary artery disease)    
        Personal history of tobacco use, presenting hazards to health    Iron deficiency anemia    Unspecified hearing loss, unspecified ear    
        Sob (shortness of breath)    Sebaceous cyst of left axilla    Pvd (peripheral vascular disease) (hcc)    Palpitations    
        Osa (obstructive sleep apnea)    Obesity    Mixed hyperlipidemia    Hematuria    Family history of ischemic heart disease    
        Essential (primary) hypertension    Edema    Ed (erectile dysfunction)     Aortic aneurysm of unspecified site, without rupture (hcc)    
        Deafness    Atrial fibrillation (hcc)    Abdominal aortic aneurysm (aaa) (hcc)    Hypoxia    
        Incisional hernia, without obstruction or gangrene 7/26/2019   Obesity (bmi 30.0-34.9)       
        Past Surgical History   Procedure Date   Foot/ankle surgery 2005    Ankle surgery   Cyst removed from neck 07/2011   
        Tonsillectomy, primary/secondary; < age 12    Knee surgery    Endovascular, aaa repair 03/17/2017   Tonsillectomy / adenoidectomy    
        Aortic aneurysm repair    Hernia repair-ventral    Cataract extraction, bilateral     10/2018 and 11/2018        
        Current Outpatient Prescriptions:  amLODIPine (NORVASC) 10 mg oral tablet Take 1 tablet by mouth daily   
        atorvastatin (LIPITOR) 20 mg oral tablet Take 1 tablet by mouth daily   FIBER CHOICE ORAL Take by mouth daily   
        lisinopril 40 mg oral tablet Take 1 tablet by mouth daily   ferrous sulfate 325 mg (65 mg iron) oral tablet TAKE 1 TABLET BY MOUTH THREE TIMES DAILY
        No Known Allergies   Social History     Social History Narrative     Not on file      Family History   
        Problem Relation Age of Onset     Dementia/Neurocognitive Disorder Biological Mother      Alzheimer's disease Biological Father
        Parkinson's Disease Biological Father      Other Sister      Heart Disease Other         Grandmother     Other Sister
        Congestive heart failure     Heart Disease Maternal Grandmother      Ca, Colon Paternal Uncle      
        Adverse Anesthesia Reaction No History of      Bleeding Disorders No History of      
        Blood Clotting Disorder No History of      Drug/Alcohol Abuse No History of      
        Ca, Rectal No History of      ROS:  Reviewed and documented in chart."""
        answer4 = """
                    Reasoning:
                    Step 1: Read the entire sentence and extract any relevant medical diagnoses or conditions listed.
                    Step 2: The sentence includes multiple chronic conditions and past medical history items, including "Atrial fibrillation (hcc)".
                    Step 3: Compare this against the known list of arrhythmia-related terms:
                    Includes: Afib, A fib, Atrial fibrillation, Atrial flutter, VF, VT, dysrhythmia, etc.
                    Step 4: The term "Atrial fibrillation" exactly matches a key term from the list.
                    Step 5: Atrial fibrillation is a specific type of cardiac arrhythmia, indicating abnormal electrical activity in the atria.
                    Step 6: Therefore, this sentence clearly contains information about an arrhythmia.
                    Answer:
                    Yes. The sentence includes “Atrial fibrillation (hcc)”, which matches the list of known arrhythmias. 
                    Atrial fibrillation is a well-recognized cardiac arrhythmia, so this sentence is classified as containing arrhythmia information."""
        val_df = pd.read_csv(os.path.join(SPLIT_DIR, tag, 'val.csv'))
        y_true = val_df["target"].tolist()
        y_pred = []
        llm_outputs = []
        index=0
        for sentence in val_df["sentence"]:
            print("/n")
            print("Here is the sentence:")
            print(sentence)
            print(index)
            index+=1
            response = client.chat(model=MODEL, messages=[
                {"role": "user", "content": "Here is the sentence:"+sentence4+prompt1},
                {"role": "asistant", "content": answer4},
                {"role": "user", "content": "Here is the sentence:"+sentence+prompt1}
                ],options={"temperature": 0.5})
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
        val_df.to_csv(os.path.join(MODEL_DIR, f"{tag}_llama_few_shot_run_4.csv"), index=False)
        print(classification_report(y_true, y_pred, digits=4))

        with open(os.path.join(MODEL_DIR, f'{tag}_llama_few_shotreport.txt'), 'a') as f:
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
        
        sentence5="""Trace edema bilateral lower extremities."""
        answer5="""Reasoning:
                Step 1: Read the sentence and Identify key terms that may indicate signs or evidence of heart failure.”
                Step 2: The phrase “bilateral lower extremities edema” directly matches one of the listed indicators: “bilateral leg edema”, which is a known clinical sign of fluid overload.
                Step 3: Although the sentence does not explicitly mention a diagnosis of heart failure, bilateral edema is a common and recognized symptom associated with congestive heart failure (CHF).
                Step 4: Since the sentence includes a relevant physical finding from the keyword list, it provides indirect evidence consistent with possible heart failure.
                Answer:
                Yes. The sentence mentions “bilateral lower extremities edema,” which matches “bilateral leg edema,” a recognized sign suggestive of heart failure or fluid overload."""
        sentence6="""Was on Neurontin in past, this was stopped as it was thought to potentially be contributing to her edema."""
        answer6="""Reasoning:
                Step 1: Read the sentence and Identify key terms that may indicate signs or evidence of heart failure.”
                Step 2: The term “edema” appears in the sentence. While it is not specifically labeled as "bilateral leg edema," it is one of the listed signs: “swelling, edema, fluid overload”.
                Step 3: The sentence also provides clinical reasoning: a medication (Neurontin) was discontinued due to its suspected role in causing or worsening edema, indicating concern for pathologic fluid retention.
                Step 4: Although heart failure is not explicitly diagnosed, edema is a hallmark symptom and a clinical clue that may suggest or overlap with heart failure.
                Answer:
                Yes. The sentence mentions "edema," which is a recognized sign associated with heart failure or fluid overload, and the context implies it was clinically significant enough to discontinue a medication."""
        sentence7="""General: Sitting on exam table with mild to moderate discomfort  Respiratory: Symmetrical chest wall expansion, nonlabored  Cardiac: Regular rate and rhythm  Integ: No wounds vascular: Nonpalpable distal pulses, palpable bilateral femoral pulses, feet warm and well-perfused, 2+ edema to the right and 1+ edema to the left      Assessment and Plan:     Testing:  Right Lower Limb  Ankle Pressure: 131  mm Hg.,  ABI: 1.04."""
        answer7="""Reasoning:
                Step 1: Review the clinical findings for any signs or evidence related to heart failure.
                Step 2: In the vascular section, the sentence notes “2+ edema to the right and 1+ edema to the left”.
                Step 3: Edema—especially in both lower extremities—is one of the listed indicators: “bilateral leg edema,” “swelling,” “edema,” or “fluid overload”, which can suggest underlying heart failure, particularly when symmetric and without another clear local cause.
                Step 4: While no explicit diagnosis of heart failure is made in the sentence, the presence of bilateral leg edema is considered a clinical sign consistent with fluid overload, which supports a potential heart failure diagnosis.
                Answer:
                Yes. The sentence includes “2+ edema to the right and 1+ edema to the left,” indicating bilateral leg edema, which is a recognized sign of possible heart failure or fluid overload."""
        sentence8="""Localized edema I have advised her that she can increase her morning Lasix pill from 40 up to 60 mg daily for one week, then she is to drop it back down to 40 mg."""
        answer8="""Reasoning:
                Step 1: Review the clinical findings for any signs or evidence related to heart failure
                Step 2: The sentence mentions “localized edema”, which falls under the broader category of “edema”, a listed clinical sign of potential heart failure or fluid overload.
                The sentence also references Lasix (furosemide), a loop diuretic commonly used to manage volume overload in heart failure patients.
                Step 3: The instruction to increase the dose of Lasix in response to edema suggests that the clinician is treating fluid retention, which is strongly associated with heart failure management.
                Step 4: While the edema is described as “localized” rather than “bilateral,” the presence of edema plus diuretic adjustment strongly implies concern for volume status, likely related to cardiac function.
                Answer:
                Yes. The sentence discusses “localized edema” and increasing Lasix dosage, which suggests treatment of fluid overload—commonly seen in heart failure—thus indicating possible underlying heart failure."""        
        sentence9="""Presents today with ulceration left lower leg upon examination of the ulceration we see the ulceration is smaller 
        per measurement we see no signs infection is not red not hot not swollen no purulent drainage is noted no odor is noted patient 
        has less edema around the wound patient has good granulation tissue within the wound and also some capillary budding 
        there also is some slight amount of loose yellow slough is determined is time to debride the area of nonviable subcutaneous dermal 
        and epidermal tissues slough then [PERSONALNAME] and Unna boot plus was applied along with drawtex between layers 
        patient has much less edema and appears to be healing very well patient having no discomfort or pain patient will return in 1 week for reevaluation    
        Review of Systems   Skin:  Positive for wound."""
        answer9="""Reasoning:
                Step 1: Review the clinical findings for any signs or evidence related to heart failure.
                Step 2: The sentence mentions “less edema around the wound” and “patient has much less edema”.
                Step 3: Edema is one of the listed signs and evidence related to heart failure, particularly when it is peripheral and not attributed to another cause.
                Step 4: The reduction in edema suggests effective management, which is often achieved through diuretic therapy like Lasix, commonly used in heart failure.
                Answer: 
                Yes. The sentence mentions edema in the lower leg, which is a recognized sign of fluid overload and can be associated with heart failure."""
        sentence10="""Comments: Zero to trace edema   Neurological:      Mental Status: He is alert."""
        answer10="""Reasoning:
                Step 1: Review the clinical findings for any signs or evidence related to heart failure.
                Step 2: The sentence mentions “zero to trace edema”.
                Step 3: Edema is one of the listed signs and evidence related to heart failure, particularly when it is peripheral and not attributed to another cause.
                Step 4: The presence of edema suggests fluid overload, which is a common finding in heart failure patients.
                Answer:
                Yes. The sentence mentions “zero to trace edema,” which is a recognized sign of fluid overload and can be associated with heart failure."""
        sentence11="""1 Bottle 10     melatonin 3 MG Tablet     VITAMIN B COMPLEX-C Capsule    Neck: supple, no adenopathy, no bruits, thyroid normal size, 
        non-tender, without nodularity  Lymph: no palpable lymphadenopathy  Heart: regular rate & rhythm, no murmurs and no gallops  
        Lungs: chest symmetric with normal AP diameter, no chest deformities noted, no chest wall tenderness, lungs clear to auscultation  
        Abdomen: abdomen soft, non-tender, normal bowel sounds and no masses or organomegaly  Extremities: less than 2 second capillary refill, 
        no joint deformities, effusion, or inflammation    Assessment:  Lumbar and piriformis pain wishes injection   Hyperlipidemia   afib  chf        
        Plan:  Continue present medication(s): CHF and afib continue med's and diltiazem Metoprolol and eliquis."""
        answer11="""Reasoning:
                Step 1: Review the clinical findings for any signs or evidence related to heart failure.
                Step 2: In the Assessment section, the sentence clearly lists “chf”, which stands for congestive heart failure — a direct match to the keyword “HF” (heart failure).
                Step 3: In the Plan section, it also mentions continuing medications for CHF, reinforcing that this is an active, documented diagnosis.
                Step 4: Since “CHF” is an abbreviation for congestive heart failure, and it's directly listed as part of the clinical plan and problem list, this sentence contains explicit evidence of heart failure.
                Answer:
                Yes. The sentence includes “chf” in the assessment and medication plan, which directly refers to congestive heart failure — a clearly defined instance of heart failure."""
        val_df = pd.read_csv(os.path.join(SPLIT_DIR, tag, 'val.csv'))
        y_true = val_df["target"].tolist()
        y_pred = []
        llm_outputs = []
        index=0
        for sentence in val_df["sentence"]:
            print("/n")
            print("Here is the sentence:")
            print(sentence)
            print(index)
            index+=1
            response = client.chat(model=MODEL, messages=[
                {"role": "user", "content": "Here is the sentence:"+sentence5+prompt2},
                {"role": "asistant", "content": answer5},
                {"role": "user", "content": "Here is the sentence:"+sentence6+prompt2},
                {"role": "asistant", "content": answer6},
                {"role": "user", "content": "Here is the sentence:"+sentence7+prompt2},
                {"role": "asistant", "content": answer7},
                {"role": "user", "content": "Here is the sentence:"+sentence8+prompt2},
                {"role": "asistant", "content": answer8},
                {"role": "user", "content": "Here is the sentence:"+sentence9+prompt2},
                {"role": "asistant", "content": answer9},
                {"role": "user", "content": "Here is the sentence:"+sentence10+prompt2},
                {"role": "asistant", "content": answer10},
                {"role": "user", "content": "Here is the sentence:"+sentence11+prompt2},
                {"role": "asistant", "content": answer11},
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
        val_df.to_csv(os.path.join(MODEL_DIR, f"{tag}_llama_few_shot.csv"), index=False)
        print(classification_report(y_true, y_pred, digits=4))

        with open(os.path.join(MODEL_DIR, f'{tag}_llama_few_shotreport.txt'), 'a') as f:
            f.write(classification_report(y_true, y_pred, digits=4))
            f.write("\n\n")
    elif tag == 'cartoxvalvularcomplications':
        prompt3 = f"""You are given a sentence, if that sentence contains any information related to instances of valvular complications, respond with yes and explain why. 
        If not, respond with no and explain why. These words being the signs and evidence:  
        TR, tricuspid insufficiency, incompetence, right AV valve regurgitation  
        AR, aortic regurgitation, aortic incompetence, aortic valve insufficiency  
        valve disorder, AV valve abnormality, valvular dysfunction.
        If these words are mentioned in the sentence, respond with yes and explain why.
        If these words are not mentioned in the sentence, respond with no and explain why.""" 
        sentence12="""PAST MEDICAL HISTORY:  Past Medical History   Diagnosis Date   Unspecified essential hypertension [ALPHANUMERICID]    
        Hypertension   Depressive disorder, not elsewhere classified 07/09/98    Depression, non-psychotic   Mitral valve disorders(424.0) 07/09/98    
        Mild MR on echo with thickened leaflets   Unspecified migraine 07/09/98   Unspecified gastritis and gastroduodenitis 12/07/99   
        Inguinal hernia without mention of obstruction or gangrene, unilateral or unspecified, (not specified as recurrent)     
        Inguinal hernia   Varicella without mention of complication    Rubella without mention of complication    Head injury, unspecified    
        Problems with sight @2009    left eye blurred vision   Osteoporosis     neck and back   Cataract    Hyperlipidemia    
        Headache disorder June 2014    from concussion- constant   Chronic neck and back pain        
        PAST SURGICAL HISTORY:  Past Surgical History   Procedure Date   Excis varicocel,abd aprch+fix hernia 1985   
        Hernia repair-inguinal 08/1986    Dr."""
        answer12="""Reasoning:
                    Step 1: Read the sentence carefully for any mention of valvular abnormalities or related diagnoses.
                    Step 2: The sentence includes: “Mitral valve disorders (424.0)” and “Mild MR on echo with thickened leaflets.”
                    Step 3: “MR” stands for mitral regurgitation, which is a valvular complication and falls under the broader category of valve disorder and valvular dysfunction.
                    Step 4: Although “MR” is not in the provided keyword list, “valve disorder” and “valvular dysfunction” are, and the diagnosis of mitral valve disorder with regurgitation directly implies valvular disease.
                    Answer:
                    Yes. The sentence includes “Mitral valve disorders” and “Mild MR,” which indicate a diagnosed valvular complication, fulfilling the criteria for identifying valvular dysfunction."""
        sentence13="""Aortic valvular disease."""
        answer13="""Reasoning:
                    Step 1: Read the sentence carefully for any mention of valvular abnormalities or related diagnoses.
                    Step 2: The sentence includes: “Aortic valvular disease.”
                    Step 3: Aortic valvular disease is a valvular complication and falls under the broader category of valve disorder and valvular dysfunction.
                    Step 4: Although “aortic valvular disease” is not in the provided keyword list, “valve disorder” and “valvular dysfunction” are, and the diagnosis of aortic valvular disease directly implies valvular disease.
                    Answer:
                    Yes. The sentence includes “Aortic valvular disease,” which indicates a diagnosed valvular complication, fulfilling the criteria for identifying valvular dysfunction."""
        sentence14="""Past Medical History:   Diagnosis Date     Anemia, unspecified      Asypmtomatic Cholelithiases 4/2/2015     
        Brachial neuritis or radiculitis NOS      Chronic airway obstruction, not elsewhere classified      Degeneration of cervical intervertebral disc      
        Depression      Disorder of lipoid metabolism      GERD (gastroesophageal reflux disease)      Malignant melanoma of skin (HCC) 11/13    
        BACK NEOPLASIA - SOMERSET DERMATOLOGIST - SURGERY ONLY NEEDED FOR TREATMENT     Mild aortic stenosis 2/26/2015     Mitral valve disorder     
        SMALL LEAK - DR [PERSONALNAME]     Nonmelanoma skin cancer      Nontoxic multinodular goiter      Obesity, unspecified      Osteoarthritis      
        Other and unspecified coagulation defects      Unspecified asthma(493.90)      Unspecified essential hypertension      Unspecified sleep apnea        
        Past Surgical History:   Procedure Laterality Date     BILATERAL AXILLA LYMPH NODES       CARDIAC CATH       carpal tunnel repair Left      
        COLONOSCOPY      COMPLETE     EXTRACTION OF WISDOM TOOTH       heel spur right heel       MELANOMA BACK x2  2013     melanoma left back and shoulder"""
        answer14="""Reasoning:
                    Step 1: Read the sentence carefully for any mention of valvular abnormalities or related diagnoses.
                    Step 2: The sentence contains both “Mitral valve disorder” and “Mild aortic stenosis.”
                    Step 3: “Mitral valve disorder” directly matches the keyword “valve disorder,” indicating an abnormality of the mitral valve. 
                    “Aortic stenosis” refers to narrowing of the aortic valve, which is a classic form of valvular dysfunction, even though the term “stenosis” itself is not on the list, its meaning clearly falls under valve-related pathology.
                    Step 4: Together, these represent confirmed valvular complications involving both the mitral and aortic valves."""
        sentence15="""cholecalciferol, vitamin D3, 25 mcg (1,000 unit) oral capsule Take 1 capsule by mouth daily   
        ciprofloxacin HCl (CIPRO) 500 mg oral tablet TAKE ONE TABLET BY MOUTH EVERY 12 HOURs for one day   
        ergocalciferol (VITAMIN D) 50,000 unit oral capsule Take 1 capsule by mouth once a week for 8 doses   
        fluconazole (DIFLUCAN) 150 mg oral tablet TAKE 1 TABLET BY MOUTH ONCE FOR 1 DOSE   
        furosemide (LASIX) 20 mg oral tablet Take 1 tablet by mouth daily   lancets (ONE TOUCH [PERSONALNAME]) 
        33 gauge by miscellaneous route 4 times daily before meals & nightly   lancing device with lancets (ONETOUCH [PERSONALNAME] PLUS LANC DEV) 
        by  miscellaneous route 4 times daily before meals & nightly   lidocaine (LIDODERM) 5 % patch Place 1 Patch onto the skin every 24 hours 
        Remove & Discard patch within 12 hours or as directed by MD   magnesium oxide (MAG-OX) 400 mg (241.3 mg magnesium) oral tablet 
        TAKE ONE TABLET BY MOUTH DAILY   mometasone (ELOCON) 0.1 % topical cream apply to rash on face twice a day  5 days on and 2 days off   
        NOVOFINE 32 32 gauge x 1/4"" misc needle Use 4 daily for injections   oxyCODONE (ROXICODONE) 15 mg oral immediate release tablet Take 5 mg 
        by mouth every 4 hours as needed for pain     oxycodone HCl,terephth/aspirin (OXYCODONE [PERSONALNAME] ORAL)    
        prochlorperazine maleate (COMPAZINE ORAL) Take by mouth     Allergies  Allergies   Allergen Reactions     
        Environmental Allergens Shortness of Breath     Dust Molds, Mites perfumes  Asthma attack, alloys, nickel, tin, copper     
        Interferon Alfa-2a Shortness of Breath     Latex Rash, Hives and Swelling     Ribavirin Shortness of Breath     
        Sulfa (Sulfonamide Antibiotics) Rash, Hives     Azithromycin Hydrogen Citrate Rash, Hives     Past Medical History   
        Past Medical History:   Diagnosis Date     Abnormal liver function tests 2008     Adenocarcinoma of pancreas (HCC) 06/28/2019    
        PATHOLOGIC STAGE (AJCC 8TH EDITION): ypT1c ypN1     Aortic stenosis, moderate 10/2019    The aortic valve area is 1.2 cm2 and 
        the mean gradient is 13 mmHg     Asthma      Basal cell carcinoma (BCC) of dorsum of nose 06/24/2005     Bicuspid aortic valve 10/2019     
        Cirrhosis of liver (HCC) 05/07/2008    on CT abdomen     Disorder of pancreas 10/2019    pancreatic duct leak, Chyle leak     
        Dry eyes, bilateral      Elevated AFP 03/2019     Gastric ulcer 01/02/2020     GERD (gastroesophageal reflux disease)      
        Hepatic encephalopathy (HCC) 10/2019     Hepatitis B core antibody positive 04/2008     Hepatitis C, chronic (HCC) 2008    
        HCV RNA  1a       History of blood transfusion 1969    plan." ribavirin but treatment was DC'd due to an allergic reaction."""
        answer15="""Reasoning:
        Step 1: Read the sentence carefully for any mention of valvular abnormalities or related diagnoses.
        Step 2: In the Past Medical History, the patient is diagnosed with both:
        “Aortic stenosis, moderate” (with details: aortic valve area 1.2 cm², mean gradient 13 mmHg).
        “Bicuspid aortic valve.”
        Step 3: Both aortic stenosis and bicuspid aortic valve represent structural abnormalities of the aortic valve, indicating valvular dysfunction.
        Step 4: While the term “aortic stenosis” is not verbatim from the keyword list, it clearly falls under “valve disorder” and “valvular complication”, as it impairs valve function.
        Answer:
        Yes. The sentence lists both “moderate aortic stenosis” and “bicuspid aortic valve,” which are clinically recognized valvular complications involving dysfunction of the aortic valve."""
        sentence16="""Continues tio have back and piriformis pain    Patient Active Problem List   Diagnosis Code     Aphakia [ALPHANUMERICID]     
        Blindness, one eye [ALPHANUMERICID]     Atrial fibrillation (HCC) [ALPHANUMERICID]     Heart failure, systolic (HCC) [ALPHANUMERICID]     
        S/P ablation of atrial fibrillation [ALPHANUMERICID]     Acute low back pain with sciatica [ALPHANUMERICID]     
        SOB (shortness of breath) [ALPHANUMERICID]     Mixed dyslipidemia [ALPHANUMERICID]     HTN (hypertension) I10     
        Nonrheumatic mitral valve regurgitation [ALPHANUMERICID]     Glaucoma suspect of both eyes [ALPHANUMERICID]     
        Chronic angle-closure glaucoma of both eyes, moderate stage [ALPHANUMERICID]     Age-related nuclear cataract of both eyes [ALPHANUMERICID]     
        Dry eyes, bilateral [ALPHANUMERICID]     Current Outpatient Medications   Medication Sig Dispense Refill     
        dilTIAZem (DILTIAZEM CD) 120 MG CP24 extended release capsule Take 1 Cap by mouth daily."""
        answer16="""Reasoning:
        Step 1: Read the sentence carefully for any mention of valvular abnormalities or related diagnoses.
        Step 2: The sentence includes: “Nonrheumatic mitral valve regurgitation”, which directly refers to a valvular abnormality of the mitral valve.
        Step 3: Mitral valve regurgitation is a valvular complication and falls under the broader category of valve disorder and valvular dysfunction.
        Step 4: Although “mitral valve regurgitation” is not in the provided keyword list, “valve disorder” and “valvular dysfunction” are, and the diagnosis of mitral valve regurgitation directly implies valvular disease.
        Answer:
        Yes. The sentence includes “Nonrheumatic mitral valve regurgitation,” which indicates a diagnosed valvular complication, fulfilling the criteria for identifying valvular dysfunction."""
        val_df = pd.read_csv(os.path.join(SPLIT_DIR, tag, 'val.csv'))
        y_true = val_df["target"].tolist()
        y_pred = []
        llm_outputs = []
        index=0
        for sentence in val_df["sentence"]:
            print("/n")
            print("Here is the sentence:")
            print(sentence)
            print(index)
            index+=1
            response = client.chat(model=MODEL, messages=[
                {"role": "user", "content": "Here is the sentence:"+sentence12+prompt3},
                {"role": "asistant", "content": answer12},
                {"role": "user", "content": "Here is the sentence:"+sentence13+prompt3},
                {"role": "asistant", "content": answer13},
                {"role": "user", "content": "Here is the sentence:"+sentence14+prompt3},
                {"role": "asistant", "content": answer14},
                {"role": "user", "content": "Here is the sentence:"+sentence15+prompt3},
                {"role": "asistant", "content": answer15},
                {"role": "user", "content": "Here is the sentence:"+sentence16+prompt3},
                {"role": "asistant", "content": answer16},
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
        val_df.to_csv(os.path.join(MODEL_DIR, f"{tag}_llama_few_shot.csv"), index=False)
        print(classification_report(y_true, y_pred, digits=4))

        with open(os.path.join(MODEL_DIR, f'{tag}_llama_few_shotreport.txt'), 'a') as f:
            f.write(classification_report(y_true, y_pred, digits=4))
            f.write("\n\n")
    elif tag == 'drugsofinterest':
        prompt4 = f"""You are given a sentence, if that sentence directly or indirectly mentions capecitabine, which contains 5-FU, or directly or indirectly mentions other drugs that also contains 5-FU, respond with yes and explain why. 
        If not, respond with no and explain why. These words being the signs and evidence:  
        Capecitabine, Xeloda, Xitabin  
        5-FU, 5-Fluorouracil, Fluoro Uracil, Adrucil, Carac, Flurablastin  
        CAPOX, CAPIRI, CAPEOX, CAPEMONO, FOLFOX, FOLFIRI, FOLFOXIRI, MFOLFOX, AIO, De Gramont regimen, XELOX, XELIRI, FOLFIRINOX.
        If these words are mentioned in the sentence, respond with yes and explain why.
        If these words are not mentioned in the sentence, respond with no and explain why.
       """
        sentence17="""CISplatin  Fluorouracil  Dexamethasone Sodium Phosphate  Fluzone Quadrivalent  Fosaprepitant Dimeglumine  Granisetron HCl  Magnesium [PERSONALNAME]    Active Treatment Medications Given this Visit  Displays each medication that has an approved dose recording on the  effective date."""
        answer17="""Reasoning:
                    Step 1: Read the sentence for any direct or indirect mentions of capecitabine, its brand names, or other drugs that contain 5-FU (5-fluorouracil).
                    Step 2: The sentence lists multiple chemotherapy-related medications, including “Fluorouracil”.
                    Step 3: “Fluorouracil” is a direct match to the keyword “5-FU”, also known as 5-Fluorouracil, which is the active compound that capecitabine metabolizes into.
                    Step 4: Since Fluorouracil is itself a 5-FU-containing drug, its presence satisfies the inclusion criteria.
                    Answer:
                    Yes. The sentence includes “Fluorouracil,” which is a direct form of 5-FU, the active chemotherapeutic agent also produced by capecitabine. Therefore, this sentence refers to a 5-FU-containing drug."""
        sentence18="""On 12/22/20, I performed bronchoscopy, EGD, laparoscopic  needs to meet with oncology - I would favor immunotherapy at this point"""
        answer18="""Reasoning:
                    Step 1: Read the sentence carefully to identify any mention of capecitabine (Xeloda) or related 5-FU–containing drugs or regimens.
                    Step 2: The sentence describes procedures performed on 12/22/20: bronchoscopy, EGD, laparoscopic surgery, and mentions a referral to oncology.
                    Step 3: The sentence also states a preference for immunotherapy, but it does not mention any chemotherapy drugs — neither capecitabine (Xeloda, Xitabin), nor any 5-FU drugs (Fluorouracil, Adrucil, Carac, etc.), nor any 5-FU–based regimens (FOLFOX, CAPOX, etc.).
                    Step 4: Because none of the required keywords are mentioned directly or indirectly, the sentence does not meet the criteria.
                    Answer:
                    No. The sentence discusses procedures and mentions a potential plan for immunotherapy but does not reference capecitabine, 5-FU, or any related regimens."""
        val_df = pd.read_csv(os.path.join(SPLIT_DIR, tag, 'val.csv'))
        y_true = val_df["target"].tolist()
        y_pred = []
        llm_outputs = []
        index=0
        for sentence in val_df["sentence"]:
            print("/n")
            print("Here is the sentence:")
            print(sentence)
            print(index)
            index+=1
            response = client.chat(model=MODEL, messages=[
                {"role": "user", "content": "Here is the sentence:"+sentence17+prompt4},
                {"role": "asistant", "content": answer17},
                {"role": "user", "content": "Here is the sentence:"+sentence18+prompt4},
                {"role": "asistant", "content": answer18},
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
        val_df.to_csv(os.path.join(MODEL_DIR, f"{tag}_llama_few_shot.csv"), index=False)
        print(classification_report(y_true, y_pred, digits=4))

        with open(os.path.join(MODEL_DIR, f'{tag}_llama_few_shotreport.txt'), 'a') as f:
            f.write(classification_report(y_true, y_pred, digits=4))
            f.write("\n\n")

if __name__ == "__main__":
    few_shot_llama(TAGS[4])