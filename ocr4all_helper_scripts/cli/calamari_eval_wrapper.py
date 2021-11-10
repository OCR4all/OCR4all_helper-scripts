from ocr4all_helper_scripts.helpers import calamari_eval_helper

import click


@click.command("calamari-eval-wrapper", help="Evaluates OCR quality via calamari-eval.")
@click.argument("FILES", nargs=-1)
@click.option("--num_threads", type=int, default=1)
@click.option("--n-confusions", type=int, default=10)
@click.option("--skip_empty_gt", is_flag=True, type=bool, default=False)
def calamari_eval_cli(files, num_threads, n_confusions, skip_empty_gt):
    calamari_eval_helper.prepare_filesystem()
    calamari_eval_helper.save_eval_files(files)
    calamari_eval_helper.run_eval(n_confusions, skip_empty_gt, num_threads)
    calamari_eval_helper.cleanup()


if __name__ == "__main__":
    calamari_eval_cli()
