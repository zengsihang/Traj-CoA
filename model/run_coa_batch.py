"""Batch Chain-of-Agents inference for longitudinal EHR risk prediction."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import srsly
from json_repair import repair_json
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from prompts import (
    AGGREGATION_AGENT_USER_PROMPT,
    INITIAL_WORKER_USER_PROMPT,
    SUBSEQUENT_WORKER_USER_PROMPT,
    get_aggregation_agent_system_prompt,
    get_initial_worker_system_prompt,
    get_subsequent_worker_system_prompt,
)

CHARS_PER_TOKEN_ESTIMATE = 4


class LungCancerMemory:
    """Stores temporally ordered lung-cancer-related events for one patient."""

    def __init__(self, max_memory_events: int = 10) -> None:
        self.all_events: List[Dict[str, Any]] = []
        self.max_memory_events = max_memory_events

    def add_events(self, new_events: Sequence[Dict[str, Any]]) -> None:
        """Append valid events and keep them sorted by timestamp."""
        for event in new_events:
            if (
                isinstance(event, dict)
                and event.get("timestamp") is not None
                and event.get("event") is not None
            ):
                self.all_events.append(event)

        self.all_events.sort(key=lambda item: item.get("timestamp", ""))

    def get_recent_events(self, count: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return the most recent events for prompt context."""
        count = self.max_memory_events if count is None else count
        return self.all_events[-count:] if len(self.all_events) > count else list(self.all_events)

    def get_all_events(self) -> List[Dict[str, Any]]:
        """Return all stored events."""
        return list(self.all_events)

    def get_memory_summary(self) -> str:
        """Return a compact summary used in saved outputs."""
        return (
            f"Memory contains {len(self.all_events)} total events, "
            f"showing last {self.max_memory_events} for subsequent agents"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the Traj-CoA Chain-of-Agents pipeline over longitudinal EHR XML "
            "records and save per-patient predictions plus evaluation metrics."
        )
    )
    parser.add_argument("--input-file", required=True, help="Path to the XML-formatted patient dataset JSON.")
    parser.add_argument(
        "--output-file",
        default="outputs/chain_of_agent_output.json",
        help="Path for the per-patient output JSON.",
    )
    parser.add_argument("--year", type=int, default=1, choices=[1, 2, 3], help="Prediction window in years.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct", help="Base model identifier.")
    parser.add_argument("--lora-path", default=None, help="Optional path to a LoRA adapter.")
    parser.add_argument("--max-new-tokens", type=int, default=8000, help="Maximum generated tokens per call.")
    parser.add_argument("--chunk-max-tokens", type=int, default=8000, help="Approximate maximum tokens per EHR chunk.")
    parser.add_argument("--max-chunks", type=int, default=15, help="Maximum number of chunks processed per patient.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling parameter.")
    parser.add_argument("--top-k", type=int, default=64, help="Top-k sampling parameter.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size.")
    parser.add_argument("--data-parallel-size", type=int, default=1, help="vLLM data parallel size.")
    parser.add_argument("--max-model-len", type=int, default=None, help="Optional vLLM max model length.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="Fraction of GPU memory reserved by vLLM.",
    )
    parser.add_argument(
        "--async-scheduling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable vLLM async scheduling.",
    )
    parser.add_argument(
        "--memory-window",
        type=int,
        default=10,
        help="Number of recent memory events passed to subsequent worker prompts.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run on a random subset of at most 10 patients for faster debugging.",
    )
    return parser.parse_args()


def validate_prediction_year(input_path: Path, expected_year: int) -> None:
    """Validate that the CLI year matches any year encoded in the input path."""
    matches = re.findall(r"(\d+)_year", str(input_path))
    if not matches:
        return

    inferred_year = int(matches[-1])
    if inferred_year != expected_year:
        raise ValueError(
            f"--year={expected_year} does not match the dataset path year ({inferred_year}) in {input_path}."
        )


def extract_events_from_output(output_obj: Any) -> List[Dict[str, Any]]:
    """Extract event dictionaries from worker JSON output."""
    try:
        output_data = json.loads(output_obj) if isinstance(output_obj, str) else output_obj
    except (json.JSONDecodeError, TypeError) as exc:
        print(f"Warning: could not parse agent output for event extraction: {exc}")
        return []

    if not isinstance(output_data, dict):
        return []

    events = output_data.get("risk_factors_or_clinical_events")
    if events is None:
        events = output_data.get("new_risk_factors_or_clinical_events", [])

    formatted_events = []
    for event in events:
        if isinstance(event, dict) and "timestamp" in event and "event" in event:
            formatted_events.append(event)
    return formatted_events


def format_memory_events_for_prompt(events: Sequence[Dict[str, Any]]) -> str:
    """Format memory events as bullet points for prompt injection."""
    if not events:
        return "No previous events in memory."

    return "\n".join(
        f"- {event.get('timestamp', 'Unknown')}: {event.get('event', 'No description')}"
        for event in events
    )


def estimate_tokens(text: str) -> int:
    """Rough token estimate for medical XML text."""
    return len(text) // CHARS_PER_TOKEN_ESTIMATE


def find_completed_dttm_positions(xml_text: str) -> List[int]:
    """Locate positions of opening ``CompletedDTTM`` tags."""
    pattern = r"<CompletedDTTM[^>]*>"
    return [match.start() for match in re.finditer(pattern, xml_text, re.IGNORECASE)]


def split_large_segment_with_temporal_info(segment: str, max_tokens: int) -> List[str]:
    """Split oversized segments while preserving their timestamp tag."""
    opening_match = re.search(r"<CompletedDTTM[^>]*>", segment, re.IGNORECASE)
    closing_match = re.search(r"</CompletedDTTM>", segment, re.IGNORECASE)

    if not opening_match or not closing_match:
        char_limit = max_tokens * CHARS_PER_TOKEN_ESTIMATE
        return [segment[i : i + char_limit] for i in range(0, len(segment), char_limit)]

    opening_tag = opening_match.group(0)
    closing_tag = closing_match.group(0)
    timestamp = segment[opening_match.end() : closing_match.start()]
    medical_data = segment[closing_match.end() :]

    char_limit = max_tokens * CHARS_PER_TOKEN_ESTIMATE
    chunks = []
    for index in range(0, len(medical_data), char_limit):
        medical_data_part = medical_data[index : index + char_limit]
        chunks.append(f"{opening_tag}{timestamp}{closing_tag}\n{medical_data_part}")
    return chunks


def split_ehr_xml_by_tokens(xml_data: str, max_tokens: int = 8000) -> List[str]:
    """Split XML into approximately token-bounded chunks on temporal boundaries."""
    split_points = find_completed_dttm_positions(xml_data)

    if not split_points:
        if estimate_tokens(xml_data) <= max_tokens:
            return [xml_data]
        char_limit = max_tokens * CHARS_PER_TOKEN_ESTIMATE
        return [xml_data[i : i + char_limit] for i in range(0, len(xml_data), char_limit)]

    positions = [0, *split_points, len(xml_data)]
    segments = [xml_data[positions[i] : positions[i + 1]] for i in range(len(positions) - 1)]

    chunks: List[str] = []
    current_chunk = ""
    for segment in segments:
        potential_chunk = current_chunk + segment
        if estimate_tokens(potential_chunk) <= max_tokens:
            current_chunk = potential_chunk
            continue

        if current_chunk:
            chunks.append(current_chunk)

        if estimate_tokens(segment) > max_tokens:
            chunks.extend(split_large_segment_with_temporal_info(segment, max_tokens))
            current_chunk = ""
        else:
            current_chunk = segment

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def strip_reasoning_tags(text: str) -> str:
    """Remove common assistant wrapper text before JSON extraction."""
    cleaned = text
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1]
    if "assistantfinal" in cleaned:
        cleaned = cleaned.split("assistantfinal")[-1]
    return cleaned


def find_first_json_object(text: Any) -> Any:
    """Extract and repair the first JSON object from a model response."""
    if not isinstance(text, str):
        return text

    cleaned = strip_reasoning_tags(text)
    start_index = cleaned.find("{")
    if start_index == -1:
        return cleaned.strip()

    brace_depth = 0
    in_string = False
    escape_next = False

    for index in range(start_index, len(cleaned)):
        char = cleaned[index]

        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
            if brace_depth == 0:
                candidate = cleaned[start_index : index + 1]
                try:
                    return repair_json(candidate)
                except Exception:
                    return candidate

    try:
        return repair_json(cleaned[start_index:])
    except Exception:
        return cleaned[start_index:].strip()


def parse_final_risk_score(json_obj: Any) -> Optional[float]:
    """Extract the final numerical risk score from the aggregation output."""
    try:
        payload = json_obj if isinstance(json_obj, dict) else srsly.json_loads(json_obj)
    except Exception:
        return None

    risk_level = payload.get("final_risk_assessment", {}).get("risk_level")
    if risk_level is None:
        return None

    try:
        return float(risk_level)
    except (TypeError, ValueError):
        return None


def calculate_metrics(predictions: Sequence[float], ground_truth: Sequence[int]) -> Dict[str, Any]:
    """Calculate ranking and thresholded binary metrics."""
    if not predictions:
        return {
            "precision": None,
            "recall": None,
            "f1_score": None,
            "accuracy": None,
            "specificity": None,
            "sensitivity": None,
            "auroc": None,
            "auprc": None,
            "true_positives": 0,
            "true_negatives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

    y_true = [int(label) for label in ground_truth]
    y_scores = [float(score) for score in predictions]
    y_pred_bin = [1 if score >= 6 else 0 for score in y_scores]

    try:
        auroc = roc_auc_score(y_true, y_scores)
    except Exception:
        auroc = None

    try:
        auprc = average_precision_score(y_true, y_scores)
    except Exception:
        auprc = None

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred_bin,
        average="binary",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": accuracy,
        "specificity": specificity,
        "sensitivity": recall,
        "auroc": auroc,
        "auprc": auprc,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def truncate_chunks(chunks: List[str], max_chunks: int) -> List[str]:
    """Keep both early and late trajectory context when truncating long records."""
    if len(chunks) <= max_chunks:
        return chunks

    front_count = max_chunks // 2
    back_count = max_chunks - front_count
    return chunks[:front_count] + chunks[-back_count:]


def build_prompt(tokenizer: AutoTokenizer, system_prompt: str, user_prompt: str) -> str:
    """Convert chat messages into a model-ready prompt."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_batch_outputs(
    llm: LLM,
    prompts: Sequence[str],
    sampling_params: SamplingParams,
    lora_path: Optional[str],
) -> List[str]:
    """Generate one output per prompt, optionally with a LoRA adapter."""
    if lora_path:
        outputs = llm.generate(
            list(prompts),
            sampling_params,
            lora_request=LoRARequest("adapter", 1, lora_path),
        )
    else:
        outputs = llm.generate(list(prompts), sampling_params)
    return [response.outputs[0].text for response in outputs]


def select_patients_for_debug(
    patients: List[Dict[str, Any]],
    patient_ids: List[str],
) -> tuple[List[Dict[str, Any]], List[str]]:
    """Sample a small deterministic subset for local debugging."""
    if len(patients) <= 10:
        return patients, patient_ids

    import random

    random.seed(0)
    sample_indices = sorted(random.sample(range(len(patients)), 10))
    return [patients[i] for i in sample_indices], [patient_ids[i] for i in sample_indices]


def ensure_output_parent(path: Path) -> None:
    """Create the parent directory for an output path if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Entry point for batched inference."""
    args = parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    validate_prediction_year(input_path, args.year)

    data = srsly.read_json(input_path)
    if not data:
        raise ValueError(f"No patient records found in {input_path}.")

    patients = list(data.values())
    patient_ids = list(data.keys())
    if args.debug:
        patients, patient_ids = select_patients_for_debug(patients, patient_ids)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm_kwargs = {
        "model": args.model_name,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "data_parallel_size": args.data_parallel_size,
        "async_scheduling": args.async_scheduling,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 128
        print(f"Using LoRA adapter from: {args.lora_path}")

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    print("Preprocessing: splitting chunks for all patients")
    all_chunks = []
    for patient in tqdm(patients):
        patient_xml = patient.get("xml", "")
        chunks = split_ehr_xml_by_tokens(patient_xml, max_tokens=args.chunk_max_tokens)
        all_chunks.append(truncate_chunks(chunks, args.max_chunks))
    print("Preprocessing: done")

    patient_memories = [LungCancerMemory(max_memory_events=args.memory_window) for _ in patients]
    all_worker_outputs: List[List[Any]] = [[] for _ in patients]
    num_patients = len(patients)
    max_chunks = max((len(chunks) for chunks in all_chunks), default=0)
    print(f"Number of patients: {num_patients}, max chunks: {max_chunks}")

    print("Initial worker agent (chunk 0)")
    prompts = []
    patient_indices = []
    for index, chunks in enumerate(all_chunks):
        if not chunks:
            continue
        prompts.append(
            build_prompt(
                tokenizer,
                get_initial_worker_system_prompt(args.year),
                INITIAL_WORKER_USER_PROMPT.format(chunk_1_xml=chunks[0]),
            )
        )
        patient_indices.append(index)

    initial_outputs = generate_batch_outputs(llm, prompts, sampling_params, args.lora_path) if prompts else []
    for patient_index, raw_text in zip(patient_indices, initial_outputs):
        json_output = find_first_json_object(raw_text)
        all_worker_outputs[patient_index].append(json_output)
        patient_memories[patient_index].add_events(extract_events_from_output(json_output))

    for chunk_index in range(1, max_chunks):
        print(f"Subsequent worker agent (chunk {chunk_index})")
        prompts = []
        patient_indices = []
        for patient_index, chunks in enumerate(all_chunks):
            if len(chunks) <= chunk_index or not all_worker_outputs[patient_index]:
                continue
            prompts.append(
                build_prompt(
                    tokenizer,
                    get_subsequent_worker_system_prompt(args.year),
                    SUBSEQUENT_WORKER_USER_PROMPT.format(
                        previous_agent_output=all_worker_outputs[patient_index][-1],
                        memory_events=format_memory_events_for_prompt(
                            patient_memories[patient_index].get_recent_events()
                        ),
                        new_chunk_xml=chunks[chunk_index],
                    ),
                )
            )
            patient_indices.append(patient_index)

        batch_outputs = generate_batch_outputs(llm, prompts, sampling_params, args.lora_path) if prompts else []
        for patient_index, raw_text in zip(patient_indices, batch_outputs):
            json_output = find_first_json_object(raw_text)
            all_worker_outputs[patient_index].append(json_output)
            patient_memories[patient_index].add_events(extract_events_from_output(json_output))

    print("Aggregation agent (final output)")
    prompts = []
    final_patient_indices = []
    for patient_index in range(num_patients):
        if not all_worker_outputs[patient_index]:
            continue
        prompts.append(
            build_prompt(
                tokenizer,
                get_aggregation_agent_system_prompt(args.year),
                AGGREGATION_AGENT_USER_PROMPT.format(
                    final_worker_outputs=all_worker_outputs[patient_index][-1],
                    universal_memory_events=format_memory_events_for_prompt(
                        patient_memories[patient_index].get_all_events()
                    ),
                ),
            )
        )
        final_patient_indices.append(patient_index)

    final_outputs: List[Any] = [None for _ in range(num_patients)]
    batch_outputs = generate_batch_outputs(llm, prompts, sampling_params, args.lora_path) if prompts else []
    for patient_index, raw_text in zip(final_patient_indices, batch_outputs):
        final_outputs[patient_index] = find_first_json_object(raw_text)

    results = []
    scored_predictions: List[float] = []
    scored_labels: List[int] = []
    scored_patient_ids: List[str] = []

    for patient_index, patient in enumerate(patients):
        label = int(patient["is_case"])
        risk_score = parse_final_risk_score(final_outputs[patient_index])
        results.append(
            {
                "patient_id": patient_ids[patient_index],
                "all_worker_outputs": all_worker_outputs[patient_index],
                "final_output": final_outputs[patient_index],
                "predicted_risk_score": risk_score,
                "label": label,
                "memory_events": patient_memories[patient_index].get_all_events(),
                "memory_summary": patient_memories[patient_index].get_memory_summary(),
            }
        )

        if risk_score is not None:
            scored_predictions.append(risk_score)
            scored_labels.append(label)
            scored_patient_ids.append(patient_ids[patient_index])

    ensure_output_parent(output_path)
    srsly.write_json(output_path, results)
    print(f"Final output saved to {output_path}")

    metrics = calculate_metrics(scored_predictions, scored_labels)
    print("Calculating metrics")
    print(f"Metrics: {metrics}")
    print("\n=== EVALUATION RESULTS ===")
    print(f"AUROC: {metrics['auroc']}")
    print(f"AUPRC: {metrics['auprc']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1 Score: {metrics['f1_score']}")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Specificity: {metrics['specificity']}")
    print(f"Sensitivity: {metrics['sensitivity']}")
    print("\nConfusion Matrix:")
    print(f"TP: {metrics['true_positives']}, TN: {metrics['true_negatives']}")
    print(f"FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")

    metrics_output = {
        "predictions": scored_predictions,
        "metrics": metrics,
        "ground_truth": scored_labels,
        "evaluated_patient_ids": scored_patient_ids,
    }
    metrics_path = output_path.with_name(f"{output_path.stem}_metrics.json")
    srsly.write_json(metrics_path, metrics_output)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
