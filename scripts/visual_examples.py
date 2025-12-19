import argparse
import ast
import os
from typing import Dict, List, Tuple


def parse_predictions(path: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Parse predictions.txt of the form:
        video2960
        Pred: a man is playing guitar
        Refs: ['a man plays guitar', '...']

        video2961
        ...
    Returns:
        preds[vid] = "pred sentence"
        refs[vid]  = ["ref1", "ref2", ...]
    """
    preds = {}
    refs = {}

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        vid = line  # e.g. "video2960"
        if i + 1 >= n:
            break

        # Pred line
        pred_line = lines[i + 1].strip()
        if not pred_line.startswith("Pred:"):
            i += 1
            continue
        pred = pred_line[len("Pred:"):].strip()

        # Refs line
        if i + 2 < n:
            refs_line = lines[i + 2].strip()
        else:
            refs_line = "Refs: []"

        if refs_line.startswith("Refs:"):
            refs_str = refs_line[len("Refs:"):].strip()
            try:
                refs_list = ast.literal_eval(refs_str)
                if isinstance(refs_list, list):
                    refs_list = [str(r).strip() for r in refs_list if str(r).strip()]
                else:
                    refs_list = [str(refs_list).strip()]
            except Exception:
                refs_list = [refs_str]
        else:
            refs_list = []

        preds[vid] = pred
        refs[vid] = refs_list

        # skip optional blank line after each block
        i += 3
        if i < n and not lines[i].strip():
            i += 1

    return preds, refs


def escape_md(text: str) -> str:
    """
    Escape markdown pipes to avoid breaking table.
    """
    return text.replace("|", "\\|")


def main():
    ap = argparse.ArgumentParser(
        description="Visualize qualitative examples from multiple runs."
    )
    ap.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="List of 'tag=path/to/predictions.txt', e.g. "
             "xe=outs/xe/predictions.txt clip=outs/clip/predictions.txt",
    )
    ap.add_argument(
        "--out",
        default="viz_examples.md",
        help="Output markdown file.",
    )
    ap.add_argument(
        "--max_examples",
        type=int,
        default=30,
        help="Maximum number of examples to dump.",
    )
    args = ap.parse_args()

    # Parse all runs
    run_pred = {}
    run_refs = {}
    run_tags = []

    for spec in args.runs:
        if "=" not in spec:
            raise ValueError(f"Invalid --runs spec: {spec}")
        tag, path = spec.split("=", 1)
        tag = tag.strip()
        path = path.strip()
        preds, refs = parse_predictions(path)
        run_pred[tag] = preds
        run_refs[tag] = refs
        run_tags.append(tag)

    # Common set of video ids across all runs
    common_vids = None
    for tag in run_tags:
        vids = set(run_pred[tag].keys())
        common_vids = vids if common_vids is None else (common_vids & vids)

    if not common_vids:
        raise RuntimeError("No common video ids across runs.")

    common_vids = sorted(list(common_vids))

    # Pick "interesting" examples: where not all models give the same text
    selected = []
    for vid in common_vids:
        preds_here = [run_pred[tag][vid] for tag in run_tags]
        if len(set(preds_here)) > 1:
            selected.append(vid)
        if len(selected) >= args.max_examples:
            break

    if not selected:
        # fallback: just take first few
        selected = common_vids[: args.max_examples]

    print(f"Selected {len(selected)} examples for visualization.")

    # Dump markdown table
    with open(args.out, "w", encoding="utf-8") as f:
        # Header
        header = ["Video", "References"] + run_tags
        f.write("| " + " | ".join(header) + " |\n")
        f.write("| " + " | ".join(["---"] * len(header)) + " |\n")

        for vid in selected:
            # merge all refs from first run (they都一样，因为来自 raw-captions)
            refs_all = run_refs[run_tags[0]].get(vid, [])
            refs_str = "<br/>".join(escape_md(r) for r in refs_all)

            row = [escape_md(vid), refs_str]
            for tag in run_tags:
                pred = run_pred[tag].get(vid, "")
                row.append(escape_md(pred))
            f.write("| " + " | ".join(row) + " |\n")

    print(f"Markdown written to {args.out}")


if __name__ == "__main__":
    main()
