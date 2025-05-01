import asyncio
import json
import os
from openai import AzureOpenAI
from browser_calculator import run_browser_calculator

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_version='2024-02-15-preview',
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://azure-openai-miblab-ncu.openai.azure.com/"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY", "9a8467bfe81d443d97b1af452662c33c"),
)

# Test patient data
patient_note = """
            A 90-year-old African American female with a history of type 2 diabetes, essential hypertension, deep venous thrombosis, pulmonary embolism, and atrial flutter on chronic anticoagulation with apixaban presented for new-onset general tonic-clonic seizures, witnessed by her family. The patient had been bed-bound from arthritis in a nursing home, but her mental status had been intact. The patient had recently been hospitalized over 3 weeks ago for COVID-19 pneumonia and was discharged home with home hospice one week before readmission.
            According to her family, the patient's mental status had severely declined during her previous hospitalization with COVID-19. She developed staring spells after discharge that culminated into generalized tonic-clonic seizures on the day of the current admission. The episode lasted two seconds. Per her hospice nurse, blood pressures and blood glucose were well controlled at home. The home medications were thoroughly reviewed and found not to have any typical pharmacologic culprits. When she presented to the emergency department, she had two more witnessed seizures. Her blood pressure during this episode was 227/95 mm Hg, her heart rate was 98 beats/min, the temperature was 36.7°C, her respiratory rate was 16 breaths/min, and oxygen saturation was 93% while she breathed room air. Intravenous labetalol, lorazepam, and levetiracetam were administered. Her blood pressure decreased to 160/70 mm Hg and remained well controlled during her hospitalization. Head CT and basic laboratory work () were unremarkable. Her physical exam was notable for postictal confusion. She was alert and only oriented to person. An electroencephalogram (EEG) detected no evidence of seizure or epileptiform discharges, but generalized slowing with an intermittent focal slowing in the bilateral temporal regions was noticed (). Brain MRI demonstrated subcortical and cortical FLAIR signal abnormality involving the left greater than right parieto-occipital lobes and the left temporal lobe, in a pattern most compatible with posterior reversible encephalopathy syndrome (PRES) (Figures –). There was no acute intracranial hemorrhage or infarction.
            The patient had no further seizures after being treated with levetiracetam. Her mental state gradually returned to normal. She became more eager to participate in physical therapy and be more independent. She set a goal to walk without her walker and to cook for her family and friends. Her family was pleasantly surprised by this improvement in her mentation. She had no further seizure activity in the hospital and was discharged back home with home care services. She is doing well six months after discharge, is seizure-free, and follows her scheduled appointments. Follow-up MRI four months later after presentation showed complete or near resolution of the lesions (Figures –).
            No wr`itten consent has been obtained from the patient as there is no patient identifiable data included in the case report.
            """


# Calculator information
calculator_name = "CHA₂DS₂-VASc Score for Atrial Fibrillation Stroke Risk"
calculator_url =  "https://www.mdcalc.com/calc/801/cha2ds2-vasc-score-atrial-fibrillation-stroke-risk"

async def test_browser_calculator():
    print(f"Testing browser calculator with {calculator_name}...")
    try:
        result = await run_browser_calculator(
            calculator_name=calculator_name,
            calculator_url=calculator_url,
            patient_data=patient_note,
            llm_client=client
        )
        
        print("Result:", json.dumps(result, indent=2))
        
        if result.get("success", False):
            print("\nSuccess! Browser calculator worked correctly.")
            if "screenshot_path" in result:
                print(f"Screenshot saved at: {result['screenshot_path']}")
        else:
            print("\nError:", result.get("error", "Unknown error"))
            if "missing_values" in result:
                print("Missing values:", ", ".join(result.get("missing_values", [])))
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_browser_calculator()) 