import subprocess
from typing import List, Tuple
from pathlib import Path
import sys

from lxml import etree

EVAL_DIR = Path("/tmp/eval")


def prepare_filesystem():
    if EVAL_DIR.exists():
        for file in EVAL_DIR.glob("./*"):
            file.unlink()
    else:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)


def get_text_content(file: str, skip_empty_gt: bool) -> Tuple[List[str], List[str]]:
    gt, pred = [], []

    root = etree.parse(file).getroot()
    lines = root.findall(".//{*}TextLine")
    for line in lines:
        gt_equiv = [gt_equiv for gt_equiv in line.findall("./{*}TextEquiv") if gt_equiv.get("index") == "0"]
        pred_equiv = [pred_equiv for pred_equiv in line.findall("./{*}TextEquiv") if pred_equiv.get("index") == "1"]

        if len(gt_equiv) == 1:
            gt.append("".join(gt_equiv[0].find("./{*}Unicode").itertext()))
        else:
            if skip_empty_gt: continue
            gt.append("")

        if len(pred_equiv) == 1:
            pred.append("".join(pred_equiv[0].find("./{*}Unicode").itertext()))
        else:
            pred.append("")

    return gt, pred


def save_eval_files(files: List[str], skip_empty_gt: bool):
    for file in files:
        gt, pred = get_text_content(file, skip_empty_gt)

        with Path(EVAL_DIR, f"{Path(file).name.split('.')[0]}.gt.txt").open("w") as gtfile:
            gtfile.write("\n".join(gt))
        with Path(EVAL_DIR, f"{Path(file).name.split('.')[0]}.pred.txt").open("w") as predfile:
            predfile.write("\n".join(pred))


def run_eval(n_confusions: int, skip_empty_gt: bool, num_threads: int):
    command = ["calamari-eval"]
    command.extend(["--gt.texts", f"{EVAL_DIR}/*.gt.txt"])
    command.extend(["--gt.pred_extension", f".pred.txt"])
    command.extend(["--n_confusions", f"{n_confusions}"])
    command.extend(["--evaluator.setup.num_processes", f"{num_threads}"])

    if skip_empty_gt:
        command.extend(["--evaluator.skip_empty_gt", "True"])

    subprocess.run(command, stderr=sys.stderr, stdout=sys.stdout)


def cleanup():
    for file in EVAL_DIR.glob("./*"):
        file.unlink()
