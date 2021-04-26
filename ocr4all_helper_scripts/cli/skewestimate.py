from ocr4all_helper_scripts.helpers import skewestimate_helper

import json
from pathlib import Path
from multiprocessing.pool import ThreadPool

import click


@click.command("skewestimate", help="Calculate skew angles for regions read from a PAGE XML file.")
@click.option("--dataset", required=True, type=str,
              help="Path to the input dataset in json format with a list of image path, PAGE XML path and optional "
                   "output path. (Will overwrite PAGE XML if no output path is given.")
@click.option("-s", "--from-scratch", is_flag=True,
              help="Overwrite existing orientation angels, by calculating them from scratch.")
@click.option("-m", "--maxskew", type=float, default=2.0,
              help="Maximal skew of an image.")
@click.option("--skewsteps", type=int, default=8,
              help="Steps bewteen 0 and +maxskew/-maxskew to estimate a skew of a region. Higher values will be more "
                   "precise but will also take longer.")
@click.option("-p", "--parallel", type=int, default=1, help="Number of threads parallelly working on images.")
def skewestimate_cli(dataset, from_scratch, maxskew, skewsteps, parallel):
    with Path(dataset).open("r") as data_file:
        dataset = json.load(data_file)

    # Parallel processes for the pagexmllineseg
    def parallel_proc(data):
        if len(data) == 3:
            image, pagexml, pagexml_out = data
        elif len(data) == 2:
            image, pagexml = data
            pagexml_out = pagexml
        else:
            raise ValueError("Invalid data line with length {} "
                             "instead of 2 or 3".format(len(data)))

        xml_output, _ = skewestimate_helper.pagexmlskewestimate(pagexml, image, from_scratch,
                                            maxskew, skewsteps)
        with open(pagexml_out, 'w+') as output_file:
            skewestimate_helper.s_print("Save annotations into '{}'".format(pagexml_out))
            output_file.write(xml_output)

    skewestimate_helper.s_print("Process {} images, with {} in parallel"
            .format(len(dataset), parallel))

    # Pool of all parallel processed pagexmllineseg
    with ThreadPool(processes=min(parallel, len(dataset))) as pool:
        pool.map(parallel_proc, dataset)


if __name__ == "__main__":
    skewestimate_cli()
