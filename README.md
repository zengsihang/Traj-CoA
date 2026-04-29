# Traj-CoA

Traj-CoA is a lightweight research code release for longitudinal lung cancer risk assessment from EHR trajectories using a Chain-of-Agents prompting pipeline. The repository contains:

- a preprocessing script that converts patient-level longitudinal EHR JSON into XML strings
- a batched inference script that chunks each trajectory, runs sequential worker prompts, and aggregates a final risk score
- prompt templates used for the worker and aggregation agents

This repo is intended as the code artifact for a NeurIPS workshop submission. It is intentionally small and focused on the core experimental pipeline rather than a full training framework.

## Repository Layout

```text
Traj-CoA/
├── data_preprocess/
│   └── convert_to_xml.py
├── model/
│   ├── prompts.py
│   └── run_coa_batch.py
├── LICENSE
└── requirements.txt
```

## Environment

The code was written for Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`vllm` typically requires a CUDA-enabled environment for practical inference.

## Data Expectations

Patient data is not included in this repository. The scripts expect a JSON object keyed by patient id. Each patient entry should contain at least:

```json
{
  "patient_id": {
    "patient_info": {
      "BirthYear": 1950,
      "Ethnicity": "Unknown",
      "EthnicHeritage": "Unknown",
      "Sex": "F"
    },
    "longitudinal_data": [
      {
        "CompletedDTTM": "2018-01-01",
        "...": "..."
      }
    ],
    "is_case": 0
  }
}
```

## Preprocess to XML

```bash
python data_preprocess/convert_to_xml.py \
  --input-file /path/to/longitudinal_ehr_test_seed0_sample_dict.json \
  --output-file /path/to/longitudinal_ehr_test_seed0_sample_xml_small300.json \
  --year 1 \
  --sample-size 300
```

Use `--full` to convert the entire cohort without subsampling.

## Run Traj-CoA Inference

```bash
python model/run_coa_batch.py \
  --input-file /path/to/longitudinal_ehr_test_seed0_sample_xml_small300.json \
  --output-file outputs/chain_of_agent_output.json \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --year 1
```

Optional arguments expose common vLLM settings such as `--tensor-parallel-size`, `--gpu-memory-utilization`, `--lora-path`, and `--max-model-len`.

The inference script writes:

- a per-patient prediction file
- a companion metrics file with AUROC, AUPRC, and thresholded binary metrics

## Notes

- Longitudinal EHR data often contains sensitive information. Review your de-identification and data-sharing constraints before reproducing experiments.
- The repo currently covers inference and preprocessing only. Training scripts, dataset construction, and paper figures are outside this minimal release.

## License

This project is released under the license in [LICENSE](LICENSE).
