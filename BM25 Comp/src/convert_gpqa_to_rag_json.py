import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


def pick_first_nonempty(row: Dict[str, str], keys: List[str]) -> str:
    for key in keys:
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def to_doc_text(row: Dict[str, str], include_question: bool) -> str:
    question = (row.get("Question") or "").strip()
    explanation = pick_first_nonempty(
        row,
        ["Explanation", "Extra Revised Explanation", "Pre-Revision Explanation"],
    )
    correct = (row.get("Correct Answer") or "").strip()

    parts: List[str] = []
    if include_question and question:
        parts.append(f"Question: {question}")
    if explanation:
        parts.append(f"Explanation: {explanation}")
    if correct:
        parts.append(f"Correct answer: {correct}")

    if not parts:
        # Final fallback when explanation is missing.
        wrong_1 = (row.get("Incorrect Answer 1") or "").strip()
        wrong_2 = (row.get("Incorrect Answer 2") or "").strip()
        wrong_3 = (row.get("Incorrect Answer 3") or "").strip()
        parts.append(
            " ".join(
                x
                for x in [
                    f"Question: {question}" if question else "",
                    f"Choice A: {correct}" if correct else "",
                    f"Choice B: {wrong_1}" if wrong_1 else "",
                    f"Choice C: {wrong_2}" if wrong_2 else "",
                    f"Choice D: {wrong_3}" if wrong_3 else "",
                ]
                if x
            )
        )

    return "\n".join(parts)


def convert_gpqa_csv(input_csv: Path, output_json: Path, limit: Optional[int], include_question: bool) -> None:
    documents = []
    qa = []

    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            question = (row.get("Question") or "").strip()
            correct = (row.get("Correct Answer") or "").strip()
            if not question or not correct:
                continue

            record_id = (row.get("Record ID") or "").strip()
            doc_id = record_id if record_id else f"gpqa_doc_{idx:06d}"
            qid = f"gpqa_q_{idx:06d}"

            doc_text = to_doc_text(row=row, include_question=include_question)
            title = pick_first_nonempty(row, ["Subdomain", "High-level domain"]) or "gpqa"

            documents.append(
                {
                    "id": doc_id,
                    "title": title,
                    "text": doc_text,
                }
            )

            qa.append(
                {
                    "id": qid,
                    "question": question,
                    "answers": [correct],
                    "gold_doc_ids": [doc_id],
                    "choices": [
                        correct,
                        (row.get("Incorrect Answer 1") or "").strip(),
                        (row.get("Incorrect Answer 2") or "").strip(),
                        (row.get("Incorrect Answer 3") or "").strip(),
                    ],
                }
            )

            if limit is not None and len(qa) >= limit:
                break

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump({"documents": documents, "qa": qa}, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(qa)} QA rows from {input_csv} -> {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GPQA CSV to rag_compare JSON format")
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--include_question_in_doc",
        action="store_true",
        help="Include question text in each document (easier retrieval, less realistic).",
    )
    args = parser.parse_args()

    convert_gpqa_csv(
        input_csv=args.input_csv,
        output_json=args.output_json,
        limit=args.limit,
        include_question=args.include_question_in_doc,
    )


if __name__ == "__main__":
    main()
