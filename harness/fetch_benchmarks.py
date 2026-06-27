#!/usr/bin/env python3
"""Fetch fine-grained benchmark prompts grouped into task *families*.

Category names are `family_member` (e.g. sci_physics); the atlas colors by
family hue with a per-member shade. Sources come from the HuggingFace
datasets-server REST API (stdlib only). A category that fails to fetch is
skipped with a warning rather than aborting the whole set.

Output: benchmarks.json -> {category: [ {id, prompt, answer?}, ... ]}
"""
import json
import sys
import time
import urllib.parse
import urllib.request

ROWS = "https://datasets-server.huggingface.co/rows"
N = 12  # prompts per category (kept modest: ~15 categories -> ~180 prompts)
LETTERS = "ABCD"


def fetch_rows(dataset, config, split, length, offset=0):
    qs = urllib.parse.urlencode({"dataset": dataset, "config": config,
                                 "split": split, "offset": offset, "length": length})
    for attempt in range(4):
        try:
            req = urllib.request.Request(f"{ROWS}?{qs}", headers={"User-Agent": "moe-dissect"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return [it["row"] for it in json.load(r)["rows"]]
        except Exception as e:  # noqa: BLE001
            print(f"    retry {attempt+1} ({dataset}/{config}): {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed: {dataset}/{config}")


def mc_prompt(r):
    opts = "\n".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(r["choices"]))
    return (f"Answer the multiple-choice question. Reply with the letter and a "
            f"brief justification.\n\nQuestion: {r['question']}\n{opts}\nAnswer:")


def mmlu(subject):
    def build():
        rows = fetch_rows("cais/mmlu", subject, "test", N)
        return [{"id": f"{subject}-{i}", "prompt": mc_prompt(r),
                 "answer": LETTERS[int(r["answer"])]} for i, r in enumerate(rows)]
    return build


def gsm8k():
    rows = fetch_rows("openai/gsm8k", "main", "test", N)
    return [{"id": f"gsm8k-{i}", "prompt": r["question"].strip(),
             "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
            for i, r in enumerate(rows)]


def humaneval():
    rows = fetch_rows("openai/openai_humaneval", "openai_humaneval", "test", N)
    return [{"id": r.get("task_id", f"he-{i}"),
             "prompt": "Complete the following Python function:\n\n" + r["prompt"].rstrip()}
            for i, r in enumerate(rows)]


def mbpp():
    rows = fetch_rows("google-research-datasets/mbpp", "full", "test", N)
    return [{"id": f"mbpp-{i}",
             "prompt": "Write a Python function for this task:\n\n" + r["text"].strip()}
            for i, r in enumerate(rows)]


def translate():
    rows = fetch_rows("Helsinki-NLP/opus-100", "en-fr", "test", N)
    return [{"id": f"opus-{i}",
             "prompt": "Translate the following English text to French:\n\n" + r["translation"]["en"]}
            for i, r in enumerate(rows)]


def summarize():
    rows = fetch_rows("EdinburghNLP/xsum", "default", "test", N)
    return [{"id": f"xsum-{i}",
             "prompt": "Summarize the following article in one sentence:\n\n" + r["document"][:1500]}
            for i, r in enumerate(rows)]


def neutral():
    seeds = ["Tell me a little about yourself.", "Continue this text: The morning was quiet and",
             "Write a few sentences about a walk in the park.", "Describe an ordinary kitchen table.",
             "Say something interesting.", "Continue: Once upon a time there was",
             "Talk about the colour blue.", "Describe what a city sounds like at night.",
             "Tell me about clouds.", "What do people enjoy on weekends?",
             "Describe a cup of coffee.", "Write a sentence about the sea."]
    return [{"id": f"neutral-{i}", "prompt": s} for i, s in enumerate(seeds[:N])]


# category name -> builder.  Name prefix (before "_") is the family used for color.
CATEGORIES = {
    "math_word": gsm8k, "math_algebra": mmlu("abstract_algebra"),
    "code_python": humaneval, "code_mbpp": mbpp,
    "sci_physics": mmlu("college_physics"), "sci_biology": mmlu("high_school_biology"),
    "sci_chem": mmlu("high_school_chemistry"),
    "hum_history": mmlu("high_school_world_history"), "hum_philosophy": mmlu("philosophy"),
    "hum_law": mmlu("professional_law"),
    "soc_econ": mmlu("high_school_macroeconomics"), "soc_psych": mmlu("high_school_psychology"),
    "lang_translate": translate, "lang_summarize": summarize,
    "neutral": neutral,
}


def main():
    out = {}
    for name, fn in CATEGORIES.items():
        try:
            out[name] = fn()
            print(f"  {name}: {len(out[name])}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP {name}: {e}", file=sys.stderr)
    with open("benchmarks.json", "w") as f:
        json.dump(out, f, indent=2)
    fams = sorted({c.split("_")[0] for c in out})
    print(f"wrote benchmarks.json: {len(out)} categories, {len(fams)} families "
          f"({', '.join(fams)}), {sum(len(v) for v in out.values())} prompts")


if __name__ == "__main__":
    main()
