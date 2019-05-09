# Distillation of crop models to learn plant physiology theories using machine learning

TODO: ADD LINK TO PAPER HERE

## Requirements
- See `conda.yml`
- conda ([anaconda](https://www.anaconda.com/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html))
- (option) pysimriw `pip install -U git+https://github.com/ky0on/pysimriw`

## Usage
0. Run `git clone --recursive https://github.com/ky0on/simriw`
1. Download datasets from [zenodo.org](https://zenodo.org/record/2582678) (unzip and move to `./meashdata` and `./simdata`)
2. Run `conda env create --file conda.yml`
3. Run `source activate simriw`
4. Run `export SLACK_API_EK_BOT={YOUR_SLACK_API_KEY}`
5. Run `python ml.py --noise=0.001 --epochs=100 --optimizer=adam` (results will be saved in `./output/MMDD-HHMMSS`)
6. Run `sh vis_ml_saliency.sh {./output/MMDD-HHMMSS/best.h5}` to generate saliency maps
7. Open `./output/MMDD-HHMMSS/saliency_*.tiff`
