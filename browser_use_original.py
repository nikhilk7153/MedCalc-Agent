from langchain_openai import ChatOpenAI, AzureChatOpenAI
from browser_use import Agent
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

import asyncio
from browser_use.browser.browser import Browser, BrowserConfig, BrowserContextConfig

# Remove hardcoded credentials and fetch from env
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# Then create your LLM
llm = AzureChatOpenAI(
    model="gpt-4o",
    api_version='2024-02-15-preview',
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
)


async def main():
    browser = Browser(
        config=BrowserConfig(
            headless=False,
            disable_security=True,
            
        )
    )
    
    async with await browser.new_context() as context:
        try:
            patient_note = """
            A 13-year-old female patient was referred to the GI outpatient department of our hospital from a pediatric clinic to undergo the esophagogastroduodenoscopy for the evaluation of recurrent vomiting that lasted for 8 days. The vomiting usually occurred in the afternoon and was bilious in nature. Although vomiting was not related to food intake, she had avoided eating because of recurrent vomiting. She also complained of nausea, boring epigastric pain and palpitation, but denied diarrhea or melena. She had lost 8 kg of weight over 4 months, and her weight was 39.4 kg (10-25 percentile) and height was 156 cm (50-75 percentile) at presentation. Past history and family history were not remarkable. She was in Tanner stage 4. Menarche occurred 6 months ago, and menstrual cycles were irregular. Physical examination revealed a dehydrated tongue, cachexic and anxious appearance. Her blood pressure was 100/80 mmHg, pulse rate was 110/min, respiratory rate was 20/min and body temperature was 36.6℃. There was no definite enlargement or palpable mass on the anterior neck, eye abnormality or lymphadenopathy. The abdomen was soft with normal bowel sound and there was no direct or rebound tenderness on the abdomen. Electrocardiography showed sinus tachycardia. Plain abdominal X-ray showed nonspecific findings. Because she had weight loss and palpitation with anxiety, thyroid stimulating hormone (TSH) was included in the initial laboratory test. Esophagogastroduodenoscopy at the first visit of GI outpatient department revealed mild mucosal erythema on Z-line and acute erythematous gastritis with large amount of duodenogastric reflux (). Helicobacter pylori rapid urease test was negative. She was prescribed a proton pump inhibitor, prokinetic and mucosal coating agent (sucralfate) on the basis of her endoscopic diagnosis with GI department follow up after 2 weeks. Her vomiting improved during the first 2 days after medication, however, she visited the emergency department (ED) 5 days later because of recurred vomiting and boring epigastric pain. Her blood pressure was 140/80 mmHg, pulse rate was 88/min, respiratory rate was 20/min and body temperature was 36.8℃. Physical examination revealed mild epigastric direct tenderness. The physician of the ED found that she already had blood tests including TSH at GI outpatient department and it revealed TSH < 0.01 µIU/mL (normal range 0.35-5.55 µIU/mL). Complete blood count with differential, electrolyte and glucose were within normal limit. A liver function tests showed albumin 4.34 g/dL (normal range 3.8-5.3 g/dL), total bilirubin 2.47 mg/dL (normal range 0.3-1.2 mg/dL), alkaline phosphatase 139 IU/L (normal range 25-100 IU/L), AST 30 IU/L (normal range 0-35 IU/L) and ALT 38 IU/L (normal range 0-35 IU/L). A repeated liver function test at ED showed total bilirubin 1.7 mg/dL, alkaline phosphatase 133 IU/L, AST 30 IU/L, ALT 42 IU/L and magnesium and calcium were within normal limit. Subsequent blood tests for the thyroid showed free triiodothyroxine (T3) > 8.0 ng/mL (normal range 0.60-1.81 ng/mL), free thyroxine (T4) > 12.0 ng/dL (normal range 0.89-1.76 ng/dL), TSH receptor antibody 37.4% (normal range ≤ 15.0%), anti-thyroid microsomal antibody 6.22 U/mL (normal range ≤ 3.0 U/mL) and thyroglobulin antibody 6.91 U/mL (normal range ≤ 3.0 U/mL). Thyroid ultrasonography revealed diffusely enlarged glands, decreased parenchymal echogenecity and increased vascularity of both glands (). Treatment was initiated with propylthiouracil (PTU) 3.75mg/kg/day and propranolol 1 mg/kg/day, after 4 days of treatment, vomiting, epigastric pain and palpitation were improved remarkably. She was discharged with a maintenance dose of PTU (3.75 mg/kg/day) without prokinetic or proton pump inhibitor.
            She was hospitalized again 5 days later because vomiting had recurred. Liver function tests returned to normal limits, but thyroid function tests were still elevated (TSH < 0.01 µIU/mL, free T3 6.31 ng/mL and free T4 5.09 ng/dL). We switched treatment from PTU to methimazole 0.4 mg/kg/day and added inorganic iodine (Lugol's solution) for 5 days to prevent thyroid hormone release. After 3 days, there was no further vomiting, and free T3 and free T4 were decreased to 4.46 ng/mL and 2.24 ng/dL, respectively. She was discharged with a maintenance dose of methimazole (0.4 mg/kg/day) and propranolol (1 mg/kg/day).
            After 2 months with methimazole and propranolol maintenance therapy, her symptoms were stable and free T4 returned to normal limits (1.27 ng/dL) despite still having low level of TSH (< 0.01 µIU/mL). Over next 16 months with methimazole monotherapy, vomiting has not recurred and she became euthyroid with normalized TSH receptor antibody.
            """

            question = "Using the Ideal Body Weight Formula, what is the patient's ideal body weight in terms of kg? You should use the patient's medical values and health status when they were first admitted to the hospital prior to any treatment."

            url = "https://www.mdcalc.com/calc/68/ideal-body-weight-adjusted-body-weight"

            agent = Agent(
                task=f"""You are an helpful agent who will extract the information from a patient note and then enter the corresponding information into a medical calculator on MDCalc.com. 

                Here is the url of the medical calculator:
                {url}
                
                Here is the patient note: 
                {patient_note}

                Here is the question:

                {question}

                To obtain the result, YOU MUST enter the attribute information in the text box or click the correct radio button for all required fields on the page. After entering the information into the website, please wait and the computed answer will appear. Please return the answer as a JSON with the answer value with no other labels/text. I.e. {{"answer": 123}}
                """,
                llm=llm,
                browser_context=context
            )
            
            result = await agent.run()
            print(result)
            
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(main())