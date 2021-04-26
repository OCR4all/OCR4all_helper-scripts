# OCR4all_helper-scripts
Different python scripts used in the OCR4all workflow.

## Installation
### Locally
#### Clone repository
`git clone https://github.com/OCR4all/OCR4all_helper-scripts`
#### Run install in cloned repository
`pip install .`

### PyPi
`pip install ocr4all_helper_scripts`

## CLI usage
### ocr4all-helper-scripts
```
Usage: ocr4all-helper-scripts [OPTIONS] COMMAND [ARGS]...

  CLI entrypoint for OCR4all-helper-scripts

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  legacy-convert  Convert legacy OCR4all projects to latest.
  pagelineseg     Line segmentation with regions read from a PAGE xml file
  skewestimate    Calculate skew angles for regions read from a PAGE XML...
```

### Subcommands
#### legacy-convert
``` 
Usage: ocr4all-helper-scripts legacy-convert [OPTIONS]

  Convert legacy OCR4all projects to latest.

Options:
  -p, --path TEXT  Path to the OCR4all project.  [required]
  --help           Show this message and exit.

```

#### pagelineseg 
```
Usage: ocr4all-helper-scripts pagelineseg [OPTIONS]

  Line segmentation with regions read from a PAGE xml file

Options:
  --dataset TEXT               Path to the input dataset in json format with a
                               list of image path, PAGE XML path and optional
                               output path. (Will overwrite pagexml if no
                               output path is given)  [required]

  --remove-images              Remove ImageRegions from the image before
                               processing TextRegions for TextLines. Can be
                               used if ImageRegions overlap with TextRegions.

  --minscale FLOAT             Minimum scale permitted.
  --maxlines FLOAT             Maximum number of lines permitted.
  --threshold FLOAT            Baseline threshold.
  --usegauss                   Use gaussian instead of uniform.
  -s, --scale FLOAT            Scale of the input image used for the line
                               segmentation. Will be estimated if not defined,
                               0 or smaller.

  --hscale FLOAT               Non-standard scaling of horizontal parameters.
  --vscale FLOAT               Non-standard scaling of vertical parameters.
  --filter-strength FLOAT      Strength individual characters are filtered out
                               when creating a textline.

  -m, --maxskew FLOAT          Maximal estimated skew of an image.
  --skewsteps INTEGER          Steps between 0 and +maxskew/-maxskew to
                               estimate the possible skew of a region. Higher
                               values will be more precise but will also take
                               longer.

  -p, --parallel INTEGER       Number of threads parallelly working on images.
  -x, --smear-x FLOAT          Smearing strength in X direction for the
                               algorithm calculating the textline polygon
                               wrapping all contents.

  -y, --smear-y FLOAT          Smearing strength in Y direction for the
                               algorithm calculating the textline polygon
                               wrapping all contents.

  --growth-x FLOAT             Growth in X direction for every iteration of
                               the textline polygon finding. Will speed up the
                               algorithm at the cost of precision.

  --growth-y FLOAT             Growth in Y direction for every iteration of
                               the textline polygon finding. Will speed up the
                               algorithm at the cost of precision.

  --fail-save INTEGER          Fail save to counter infinite loops when
                               combining contours to a precise textline. Will
                               connect remaining contours with lines.

  --max-blackseps INTEGER      Maximum amount of black column separators.
  --widen-blackseps INTEGER    Widen black separators (to account for
                               warping).

  --max-whiteseps INTEGER      Maximum amount of whitespace column separators.
  --minheight-whiteseps FLOAT  Minimum column height (units=scale).
  --help                       Show this message and exit.

```

#### skewestimate
``` 
Usage: ocr4all-helper-scripts skewestimate [OPTIONS]

  Calculate skew angles for regions read from a PAGE XML file.

Options:
  --dataset TEXT          Path to the input dataset in json format with a list
                          of image path, PAGE XML path and optional output
                          path. (Will overwrite PAGE XML if no output path is
                          given.  [required]

  -s, --from-scratch      Overwrite existing orientation angels, by
                          calculating them from scratch.

  -m, --maxskew FLOAT     Maximal skew of an image.
  --skewsteps INTEGER     Steps bewteen 0 and +maxskew/-maxskew to estimate a
                          skew of a region. Higher values will be more precise
                          but will also take longer.

  -p, --parallel INTEGER  Number of threads parallelly working on images.
  --help                  Show this message and exit.

```