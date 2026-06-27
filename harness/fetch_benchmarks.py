#!/usr/bin/env python3
"""Fetch benchmark prompts for four task categories via the HuggingFace
datasets-server REST API (stdlib only -- no `datasets`/`pyarrow`).

Output: benchmarks.json  ->  {category: [ {id, prompt}, ... ]}

Categories -> datasets:
  math      : openai/gsm8k           (grade-school math word problems)
  code      : openai/openai_humaneval(python function-completion)
  knowledge : cais/mmlu              (multiple-choice exam questions)
  language  : Helsinki-NLP/opus-100  (English->French translation)
"""
import json
import sys
import time
import urllib.parse
import urllib.request

ROWS = "https://datasets-server.huggingface.co/rows"
N_PER_CATEGORY = 25  # prompts per category


def fetch_rows(dataset, config, split, length, offset=0):
    qs = urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": split,
         "offset": offset, "length": length})
    url = f"{ROWS}?{qs}"
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "moe-dissect"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.load(r)
            return [item["row"] for item in data["rows"]]
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt+1} ({dataset}): {e}", file=sys.stderr)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {dataset}")


def build_math():
    rows = fetch_rows("openai/gsm8k", "main", "test", N_PER_CATEGORY)
    out = []
    for i, r in enumerate(rows):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append({"id": f"gsm8k-{i}", "prompt": r["question"].strip(),
                    "answer": gold})  # final numeric answer
    return out


def build_code():
    rows = fetch_rows("openai/openai_humaneval", "openai_humaneval", "test", N_PER_CATEGORY)
    return [{"id": r.get("task_id", f"he-{i}"),
             "prompt": "Complete the following Python function:\n\n" + r["prompt"].rstrip()}
            for i, r in enumerate(rows)]


def build_knowledge():
    rows = fetch_rows("cais/mmlu", "all", "test", N_PER_CATEGORY * 3)
    letters = "ABCD"
    out = []
    for i, r in enumerate(rows[:N_PER_CATEGORY]):
        choices = r["choices"]
        opts = "\n".join(f"{letters[j]}. {c}" for j, c in enumerate(choices))
        prompt = (f"Answer the multiple-choice question. Reply with the letter "
                  f"and a brief justification.\n\nQuestion: {r['question']}\n{opts}\nAnswer:")
        out.append({"id": f"mmlu-{i}", "prompt": prompt,
                    "answer": letters[int(r["answer"])]})  # gold letter
    return out


def build_neutral():
    """Content-free control prompts: generic continuation/chit-chat with no
    clear task, used as a routing baseline to contrast against the task categories."""
    seeds = [
        "Tell me a little about yourself.",
        "Continue this text: The morning was quiet and",
        "Write a few sentences about a walk in the park.",
        "What are some things people enjoy on weekends?",
        "Describe an ordinary kitchen table.",
        "Say something interesting.",
        "Continue: Once upon a time there was",
        "Talk about the colour blue.",
        "Describe what a city sounds like at night.",
        "Tell me about clouds.",
    ]
    return [{"id": f"neutral-{i}", "prompt": s} for i, s in enumerate(seeds)]


def build_language():
    rows = fetch_rows("Helsinki-NLP/opus-100", "en-fr", "test", N_PER_CATEGORY)
    out = []
    for i, r in enumerate(rows):
        en = r["translation"]["en"]
        out.append({"id": f"opus-{i}",
                    "prompt": "Translate the following English text to French:\n\n" + en})
    return out


def main():
    builders = {
        "math": build_math,
        "code": build_code,
        "knowledge": build_knowledge,
        "language": build_language,
        "neutral": build_neutral,   # content-free routing baseline (control)
    }
    result = {}
    for cat, fn in builders.items():
        print(f"fetching {cat} ...", file=sys.stderr)
        result[cat] = fn()
        print(f"  got {len(result[cat])} prompts", file=sys.stderr)
    with open("benchmarks.json", "w") as f:
        json.dump(result, f, indent=2)
    total = sum(len(v) for v in result.values())
    print(f"wrote benchmarks.json ({total} prompts across {len(result)} categories)")


if __name__ == "__main__":
    main()
