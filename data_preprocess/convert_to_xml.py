"""Convert longitudinal EHR JSON records into XML strings for Traj-CoA."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict

import srsly
from dict2xml import dict2xml
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert longitudinal EHR patient records from JSON dictionaries to XML strings."
    )
    parser.add_argument("--input-file", default=None, help="Path to the input JSON file.")
    parser.add_argument("--output-file", default=None, help="Path to the output JSON file.")
    parser.add_argument("--data-dir", default=".", help="Base directory used when input/output paths are omitted.")
    parser.add_argument("--data-type", default="test", choices=["train", "test"], help="Dataset split name.")
    parser.add_argument("--seed", type=int, default=0, help="Dataset seed encoded in the filename.")
    parser.add_argument("--year", type=int, default=1, choices=[1, 2, 3], help="Prediction window in years.")
    parser.add_argument("--full", action="store_true", help="Convert the full cohort instead of a sampled subset.")
    parser.add_argument("--sample-size", type=int, default=300, help="Number of patients to sample when not using --full.")
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used for cohort subsampling.",
    )
    return parser.parse_args()


def resolve_input_path(args: argparse.Namespace) -> Path:
    """Resolve the input file path from CLI arguments."""
    if args.input_file:
        return Path(args.input_file)
    return Path(args.data_dir) / f"{args.year}_year" / f"longitudinal_ehr_{args.data_type}_seed{args.seed}_sample_dict.json"


def resolve_output_path(args: argparse.Namespace) -> Path:
    """Resolve the output file path from CLI arguments."""
    if args.output_file:
        return Path(args.output_file)
    suffix = "full" if args.full else f"small{args.sample_size}"
    return (
        Path(args.data_dir)
        / f"{args.year}_year"
        / f"longitudinal_ehr_{args.data_type}_seed{args.seed}_sample_xml_{suffix}.json"
    )


def sample_patients(
    data: Dict[str, Dict[str, Any]],
    sample_size: int,
    sample_seed: int,
) -> Dict[str, Dict[str, Any]]:
    """Return a deterministic subsample of the patient dictionary."""
    if sample_size > len(data):
        raise ValueError(f"Requested sample_size={sample_size}, but only {len(data)} patients are available.")

    rng = random.Random(sample_seed)
    sample_keys = rng.sample(list(data.keys()), sample_size)
    return {key: data[key] for key in sample_keys}


def build_patient_xml_record(patient: Dict[str, Any]) -> Dict[str, Any]:
    """Build the XML payload for a single patient record."""
    patient_info = patient["patient_info"]
    longitudinal_ehr = [
        {key: value for key, value in visit.items() if key != "time_to_event"}
        for visit in patient["longitudinal_data"]
    ]
    xml_dict = {
        "Demographics": {
            "BirthYear": patient_info.get("BirthYear"),
            "Ethnicity": patient_info.get("Ethnicity"),
            "Race": patient_info.get("EthnicHeritage"),
            "Sex": patient_info.get("Sex"),
        },
        "Longitudinal EHR": longitudinal_ehr,
    }
    return {
        "xml": dict2xml(xml_dict, wrap="Patient"),
        "is_case": patient["is_case"],
    }


def main() -> None:
    """Entry point for JSON-to-XML conversion."""
    args = parse_args()
    input_path = resolve_input_path(args)
    output_path = resolve_output_path(args)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data = srsly.read_json(input_path)
    if not args.full:
        data = sample_patients(data, sample_size=args.sample_size, sample_seed=args.sample_seed)

    cases = sum(1 for patient in data.values() if patient["is_case"])
    controls = len(data) - cases
    print(f"Number of cases: {cases}")
    print(f"Number of controls: {controls}")

    processed_data = {}
    for patient_id in tqdm(data, desc="Converting patients"):
        processed_data[patient_id] = build_patient_xml_record(data[patient_id])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    srsly.write_json(output_path, processed_data)
    print(f"Saved XML-formatted dataset to {output_path}")


if __name__ == "__main__":
    main()
