from ocr4all_helper_scripts.helpers import calamari_eval_helper

import tempfile
import contextlib
from pathlib import Path

import click


@click.command("calamari-eval-wrapper", help="Evaluates OCR quality via calamari-eval.")
@click.argument("FILES", nargs=-1)
@click.option("--num_threads", type=int, default=1)
@click.option("--n_confusions", type=int, default=10)
@click.option("--skip_empty_gt", is_flag=True, type=bool, default=False)
def calamari_eval_cli(files, num_threads, n_confusions, skip_empty_gt):
    outfile, outfile_name = tempfile.mkstemp()
    # Necessary because calamari-eval progress bars destroy OCR4all console output
    with contextlib.redirect_stdout(outfile):
        calamari_eval_helper.prepare_filesystem()
        calamari_eval_helper.save_eval_files(files)
        calamari_eval_helper.run_eval(n_confusions, skip_empty_gt, num_threads)
        calamari_eval_helper.cleanup()
    with Path(outfile_name).open("r") as fp:
        for line in fp.readlines():
            print(line)
    Path(outfile_name).unlink()


if __name__ == "__main__":
    calamari_eval_cli()
