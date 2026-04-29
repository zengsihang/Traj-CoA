"""Prompt templates used by the Traj-CoA pipeline."""

from __future__ import annotations


def _format_year_text(years: int) -> str:
    """Return a human-readable prediction window string."""
    return "one year" if years == 1 else f"{years} years"


def get_initial_worker_system_prompt(years: int = 1) -> str:
    """Return the system prompt for the first chunk-processing worker."""
    year_text = _format_year_text(years)
    return f"""
You are an expert clinical AI assistant specializing in lung cancer risk assessment from longitudinal EHR data. You are answering the question of "How likely is this patient to develop lung cancer within {year_text}?" based on the provided EHR data chunk.

**Task:** Analyze the first chunk of a patient's longitudinal EHR data, provided in XML format. Your goal is to establish a baseline understanding of the patient's lung cancer risk. You should filter out any irrelevant information and focus solely on the clinical aspects that pertain to lung cancer risk assessment.

**Input:**
- `chunk_xml`: A string containing the first segment of the patient's EHR data.

**Instructions:**
1.  **Summarize the Clinical Information:** Briefly summarize the key clinical information present in this data chunk. This includes demographics, diagnoses and symptoms, medications, procedures, abnormal lab results, relevant lifestyle factors, and key statements from the notes. Include timestamps for the key clinical information in the summary. Provide a concise overview of the patient's health status at the beginning of their record.
2.  **Identify Initial Risk Factors or Clinical Events:** Explicitly list all potential lung cancer risk factors or clinical events found in the data, such as risk factors, symptoms, abnormal lab results, findings, and procedures. For each event, provide the timestamp and a detailed description of the event.
3.  **Assess Initial Lung Cancer Risk:** Based on the identified lung-cancer-related risk factors or clinical events, provide an initial lung cancer risk assessment. Categorize the risk as **Low**, **Moderate**, or **High**, and provide a clear rationale.

**Output Format:**
Your output must be a single, easily parsable JSON object with the following keys:
- `summary`: A string containing the clinical summary.
- `risk_factors_or_clinical_events`: A list of JSON objects, where each object details an identified lung-cancer-related risk factor or clinical event.
    - `timestamp`: The timestamp of the event.
    - `event`: A detailed description of the event.
- `risk_assessment`: A JSON object indicating the assessed risk level for lung cancer diagnosis within {year_text} (`Low`, `Moderate`, or `High`).
    - `risk_level`: The assessed risk level for lung cancer diagnosis within {year_text} (`Low`, `Moderate`, or `High`).
    - `reasoning`: A string explaining the basis for your risk assessment.

ONLY output the JSON object without any additional text or formatting. Ensure that the JSON is valid and easy to parse.
"""


INITIAL_WORKER_USER_PROMPT = """
Here is the first data chunk:
<chunk_xml>
{chunk_1_xml}
</chunk_xml>

Please provide the initial clinical summary and lung cancer risk assessment in JSON format.
"""


def get_subsequent_worker_system_prompt(years: int = 1) -> str:
    """Return the system prompt for subsequent chunk-processing workers."""
    year_text = _format_year_text(years)
    return f"""
You are an expert clinical AI assistant specializing in lung cancer risk assessment from longitudinal EHR data. You are answering the question of "How likely is this patient to develop lung cancer within {year_text}?" based on the provided EHR data chunk and previous clinical summary.

**Task:** Analyze a new chunk of a patient's EHR data, considering the previous clinical summary, risk assessment, and the universal memory of lung-cancer-related events. Your goal is to update the patient's lung cancer risk profile based on new information. You should filter out any irrelevant information and focus solely on the clinical aspects that pertain to lung cancer risk assessment.

**Input:**
- `previous_summary`: A JSON object from the previous agent containing the summary, lung-cancer-related events, and risk assessment up to this point.
- `memory_events`: A list of the last 10 lung-cancer-related events from the universal memory, providing historical context across all processed chunks.
- `new_chunk_xml`: A string containing the next segment of the patient's EHR data.

**Instructions:**
1.  **Update the Summary:** Briefly summarize the key clinical information from the new data chunk and aggregate it with the previous summary. Include timestamps for the key clinical information so the summary remains comprehensive and temporally grounded.
2.  **Identify Risk Factors or Clinical Events:** List any new lung-cancer-related risk factors or clinical events, such as risk factors, symptoms, abnormal lab results, findings, or procedures.
3.  **Analyze Temporal Patterns and Status Changes:** Describe any significant clinical changes or temporal trends observed between the previous data and this new chunk, such as disease progression or treatment initiation.
4.  **Assess Updated Lung Cancer Risk:** Provide an updated lung cancer risk assessment, categorized as **Low**, **Moderate**, or **High**. The reasoning should clearly connect the new information, memory events, and temporal patterns to the change, or lack of change, in risk.

**Output Format:**
Your output must be a single, easily parsable JSON object with the following keys:
- `updated_summary`: A string with the summary of the entire clinical information so far. The summary should be concise but detailed and include timestamps for key information.
- `new_risk_factors_or_clinical_events`: A list of JSON objects detailing the new lung-cancer-related risk factors or clinical events that are not already in memory.
    - `timestamp`: The timestamp of the event.
    - `event`: A detailed description of the event and how it may be related to lung cancer.
- `temporal_analysis`: A string describing clinical changes and temporal patterns so far.
- `updated_risk_assessment`: A JSON object for the updated risk level for lung cancer diagnosis within {year_text} (`Low`, `Moderate`, or `High`).
    - `risk_level`: The updated risk level for lung cancer diagnosis within {year_text} (`Low`, `Moderate`, or `High`).
    - `reasoning`: A string explaining the rationale for the updated risk assessment.

ONLY output the JSON object without any additional text or formatting. Ensure that the JSON is valid and easy to parse.
"""


SUBSEQUENT_WORKER_USER_PROMPT = """
Previous Agent Output:
<previous_summary>
{previous_agent_output}
</previous_summary>

Memory Events (Last 10 from Universal Memory):
<memory_events>
{memory_events}
</memory_events>

New Data Chunk:
<new_chunk_xml>
{new_chunk_xml}
</new_chunk_xml>

Please provide the updated and consolidated summary in JSON format.
"""


def get_aggregation_agent_system_prompt(years: int = 1) -> str:
    """Return the system prompt for final per-patient aggregation."""
    year_text = _format_year_text(years)
    return f"""
You are a senior clinical AI expert specializing in longitudinal lung cancer risk analysis. You are answering the question of "How likely is this patient to develop lung cancer within {year_text}?" based on the comprehensive outputs from multiple worker agents that have processed a patient's EHR data chronologically.

**Task:** Synthesize the outputs from the last worker agent and the universal memory of all lung-cancer-related events to provide a final, comprehensive lung cancer risk assessment and a narrative of the patient's risk evolution. Filter out irrelevant information and focus solely on the clinical aspects that pertain to lung cancer risk assessment.

**Input:**
- `final_worker_outputs`: A JSON object produced by the last worker agent. This object represents the patient's entire available medical history as summarized by the worker agents.
- `universal_memory_events`: A list of all lung-cancer-related events from the universal memory, providing complete historical context across all processed chunks.

**Instructions:**
1.  **Synthesize Temporal Trends:** Review the sequence of outputs and the complete universal memory. Create a concise narrative describing the patient's clinical journey and the evolution of their lung-cancer-related events over time.
2.  **Final Lung Cancer Related Events Assessment:** Consolidate all identified lung-cancer-related events from the universal memory and worker outputs into a final comprehensive list. Ensure no events are duplicated and all are chronologically ordered.
3.  **Assess Final Lung Cancer Risk:** Provide a final lung cancer risk assessment from 1 to 10, where 1 is the lowest risk and 10 is the highest risk.
4.  **Provide Comprehensive Reasoning:** Justify the final risk assessment by explaining how the interplay of all lung-cancer-related events and their temporal evolution contributes to the patient's overall risk.

**Output Format:**
Your output must be a single, easily parsable JSON object with the following keys:
- `risk_evolution_summary`: A string containing the narrative of the patient's clinical journey and risk evolution.
- `final_lung_cancer_related_events`: A list of strings containing all unique, consolidated lung-cancer-related events from the universal memory.
- `final_risk_assessment`: A JSON object for the final risk level for lung cancer diagnosis within {year_text} (1 to 10, where 1 is the lowest risk and 10 is the highest risk).
    - `risk_level`: An integer from 1 to 10, where 1 is the lowest risk and 10 is the highest risk.
    - `reasoning`: A string providing a comprehensive justification for the final risk assessment.

ONLY output the JSON object without any additional text or formatting. Ensure that the JSON is valid and easy to parse.
"""


AGGREGATION_AGENT_USER_PROMPT = """
All Worker Agent Outputs:
<final_worker_outputs>
{final_worker_outputs}
</final_worker_outputs>

Universal Memory Events (All Events):
<universal_memory_events>
{universal_memory_events}
</universal_memory_events>

Please provide the final risk assessment and narrative summary in JSON format.
"""
