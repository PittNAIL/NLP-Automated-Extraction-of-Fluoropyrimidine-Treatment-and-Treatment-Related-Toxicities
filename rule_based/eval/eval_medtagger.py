import argparse
import re
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate MedTagger XML outputs against filename-derived ground truth."
    )
    p.add_argument(
        "--dir",
        default=r"C:\github\train_cape\HFS_input\arrhythmia_output",
        help="Directory containing MedTagger XML files (*.txt.xml).",
    )
    p.add_argument(
        "--tag",
        default="cartoxarrhythmia",
        help='Target normTarget tag to evaluate (e.g., "handfootpreventative").',
    )
    p.add_argument(
        "--accept-status",
        nargs="*",
        default=["Present"],
        help='Accepted CM status values to count as positive (default: ["Present"]). '
             "Use empty to accept any status.",
    )
    return p.parse_args()

LABEL_REGEX = re.compile(r"_(?P<label>[01])\.txt\.xml$", re.IGNORECASE)

def get_ground_truth_from_name(path: Path) -> int | None:
    """
    Extract ground truth label from filename suffix: *_0.txt.xml or *_1.txt.xml
    Returns 0/1 int or None if pattern not matched.
    """
    m = LABEL_REGEX.search(path.name)
    if not m:
        return None
    return int(m.group("label"))

def predicted_positive(xml_path: Path, target_tag: str, accept_status: list[str] | None) -> int:
    """
    Return 1 if any <CM> element has normTarget==target_tag and (optional) status in accept_status.
    Otherwise 0.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        # Malformed XML -> treat as no prediction
        return 0

    # Typical structure: <MedTagger><TAGS><CM .../></TAGS></MedTagger>
    tags_node = root.find("TAGS")
    if tags_node is None:
        return 0

    for cm in tags_node.findall("CM"):
        norm = (cm.attrib.get("normTarget") or "").strip()
        if norm != target_tag:
            continue
        if accept_status is not None and len(accept_status) > 0:
            status = (cm.attrib.get("status") or "").strip()
            if status not in accept_status:
                continue
        # Found a qualifying prediction
        return 1

    return 0

def safe_div(n, d):
    return n / d if d else 0.0

def main():
    args = parse_args()
    root_dir = Path(args.dir)
    target_tag = args.tag
    accept_status = args.accept_status if args.accept_status is not None else []

    files = sorted(root_dir.glob("*.txt.xml"))
    if not files:
        print(f"No XML files found in: {root_dir}", file=sys.stderr)
        sys.exit(1)

    TP = FP = FN = TN = 0
    bad_names = []
    errors = []

    # Optional: keep examples for quick inspection
    fp_examples = []
    fn_examples = []

    for f in files:
        gt = get_ground_truth_from_name(f)
        if gt is None:
            bad_names.append(f.name)
            continue

        pred = predicted_positive(f, target_tag=target_tag, accept_status=accept_status)

        if gt == 1 and pred == 1:
            TP += 1
        elif gt == 0 and pred == 1:
            FP += 1
            if len(fp_examples) < 10:
                fp_examples.append(f.name)
        elif gt == 1 and pred == 0:
            FN += 1
            if len(fn_examples) < 10:
                fn_examples.append(f.name)
        else:
            TN += 1

    precision = safe_div(TP, TP + FP)
    recall    = safe_div(TP, TP + FN)
    f1        = safe_div(2 * precision * recall, precision + recall)

    print("=== MedTagger Evaluation ===")
    print(f"Directory      : {root_dir}")
    print(f"Target tag     : {target_tag}")
    if accept_status:
        print(f"Accept status  : {accept_status}")
    else:
        print("Accept status  : (any)")

    total = TP + FP + FN + TN
    print(f"Files evaluated: {total}")
    if bad_names:
        print(f"Skipped (bad names): {len(bad_names)}")
        for n in bad_names[:10]:
            print(f"  - {n}")

    print("\nConfusion Matrix (Predicted vs Ground Truth):")
    print(f"  TP: {TP}  FP: {FP}")
    print(f"  FN: {FN}  TN: {TN}")

    print("\nMetrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1-score : {f1:.4f}")

    if fp_examples:
        print("\nSample False Positives:")
        for n in fp_examples:
            print(f"  - {n}")
    if fn_examples:
        print("\nSample False Negatives:")
        for n in fn_examples:
            print(f"  - {n}")

if __name__ == "__main__":
    main()
