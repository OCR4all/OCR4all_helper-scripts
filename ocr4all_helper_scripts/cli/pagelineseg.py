from ocr4all_helper_scripts.helpers import pagelineseg_helper

from pathlib import Path
import json
from multiprocessing.pool import ThreadPool

import click


@click.command("pagelineseg",
               help="Line segmentation with regions read from a PAGE xml file")
@click.option("--dataset", type=str, required=True,
              help="Path to the input dataset in json format with a list of image path, PAGE XML path and optional "
                   "output path. (Will overwrite pagexml if no output path is given)")
@click.option("--remove-images", is_flag=True,
              help="Remove ImageRegions from the image before processing TextRegions for TextLines. Can be used if "
                   "ImageRegions overlap with TextRegions.")
@click.option("--minscale", type=float, default=12.0,
              help="Minimum scale permitted.")
@click.option("--maxlines", type=int, default=300,
              help="Maximum number of lines permitted.")
@click.option("--threshold", type=float, default=0.2,
              help="Baseline threshold.")
@click.option("--usegauss", is_flag=True, help="Use gaussian instead of uniform.")
@click.option("-s", "--scale", type=float, default=None,
              help="Scale of the input image used for the line segmentation. Will be estimated if not defined, 0 or "
                   "smaller.")
@click.option("--hscale", type=float, default=1.0,
              help="Non-standard scaling of horizontal parameters.")
@click.option("--vscale", type=float, default=1.0,
              help="Non-standard scaling of vertical parameters.")
@click.option("--filter-strength", type=float, default=1.0,
              help="Strength individual characters are filtered out when creating a textline.")
@click.option("-m", "--maxskew", type=float, default=2.0,
              help="Maximal estimated skew of an image.")
@click.option("--skewsteps", type=int, default=8,
              help="Steps between 0 and +maxskew/-maxskew to estimate the possible skew of a region. Higher values "
                   "will be more precise but will also take longer.")
@click.option("-p", "--parallel", type=int, default=1,
              help="Number of threads parallelly working on images.")
@click.option("-x", "--smear-x", type=float, default=2.0,
              help="Smearing strength in X direction for the algorithm calculating the textline polygon wrapping all "
                   "contents.")
@click.option("-y", "--smear-y", type=float, default=1.0,
              help="Smearing strength in Y direction for the algorithm calculating the textline polygon wrapping all "
                   "contents.")
@click.option("--growth-x", type=float, default=1.1,
              help="Growth in X direction for every iteration of the textline polygon finding. Will speed up the "
                   "algorithm at the cost of precision.")
@click.option("--growth-y", type=float, default=1.1,
              help="Growth in Y direction for every iteration of the textline polygon finding. Will speed up the "
                   "algorithm at the cost of precision.")
@click.option("--fail-save", type=int, default=1000,
              help="Fail save to counter infinite loops when combining contours to a precise textline. Will connect "
                   "remaining contours with lines.")
@click.option("--max-blackseps", type=int, default=0,
              help="Maximum amount of black column separators.")
@click.option("--widen-blackseps", type=int, default=10,
              help="Widen black separators (to account for warping).")
@click.option("--max-whiteseps", type=int, default=-1,
              help="Maximum amount of whitespace column separators.")
@click.option("--minheight-whiteseps", type=int, default=10,
              help="Minimum column height (units=scale).")
@click.option("--bounding-rectangle", is_flag=True, default=False, help="Uses bounding rectangles instead of polygons.")
def pagelineseg_cli(dataset: str, remove_images: bool, minscale: float, maxlines: int, threshold: float,
                    usegauss: bool, scale: float, hscale: float, vscale: float, filter_strength: float, maxskew: float,
                    skewsteps: int, parallel: int, smear_x: float, smear_y: float, growth_x: float, growth_y: float,
                    fail_save: int, max_blackseps: int, widen_blackseps: int, max_whiteseps: int,
                    minheight_whiteseps: int, bounding_rectangle: bool):
    with Path(dataset).open('r') as data_file:
        dataset = json.load(data_file)

    # Parallel processes for the pagexmllineseg cli
    def parallel_proc(data):
        if len(data) == 3:
            image, pagexml, path_out = data
        elif len(data) == 2:
            image, pagexml = data
            path_out = pagexml
        else:
            raise ValueError(f"Invalid data line with length {len(data)} instead of 2 or 3")

        xml_output = pagelineseg_helper.pagelineseg(pagexml, image,
                                                    scale=scale,
                                                    vscale=vscale,
                                                    hscale=hscale,
                                                    max_blackseps=max_blackseps,
                                                    widen_blackseps=widen_blackseps,
                                                    max_whiteseps=max_whiteseps,
                                                    minheight_whiteseps=minheight_whiteseps,
                                                    minscale=minscale,
                                                    maxlines=maxlines,
                                                    smear_strength=(smear_x, smear_y),
                                                    growth=(growth_x, growth_y),
                                                    filter_strength=filter_strength,
                                                    fail_save_iterations=fail_save,
                                                    maxskew=maxskew,
                                                    skewsteps=skewsteps,
                                                    usegauss=usegauss,
                                                    remove_images=remove_images,
                                                    bounding_box=bounding_rectangle)

        with Path(path_out).open("w+") as output_file:
            pagelineseg_helper.s_print(f"Save annotations into '{path_out}'")
            output_file.write(xml_output)

    pagelineseg_helper.s_print(f"Process {len(dataset)} images, with {parallel} in parallel")

    # Pool of all parallel processed pagexmllineseg
    with ThreadPool(processes=min(parallel, len(dataset))) as pool:
        pool.map(parallel_proc, dataset)


if __name__ == "__main__":
    pagelineseg_cli()
