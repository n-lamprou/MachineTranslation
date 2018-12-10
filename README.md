# Machine Translation


## Project Details

This project aims to build end-to-end machine translation pipeline capable of translating English text input to French output. 

Different neural network architectures are evaluated and compared and details of their implementation can be found in the report.


 
## Setup
 
1. Create (and activate) a new environment with Python 3.6.
** Linux or Mac: **

```bash
conda create --name mtdl python=3.6
source activate mtdl
```

2. Clone the repository and navigate to root of the repo. Then install several dependencies.

```bash
git clone https://github.com/n-lamprou/MachineTranslation.git
cd MachineTranslation
pip install .
```

3. For using jupyter notebooks, create an IPython kernel for the cvdl environment.

```bash
python -m ipykernel install --user --name mtdl --display-name "mtdl"
```

## Instructions

### Training

To train a translator, the `learn.py` script needs to be executed. The additional `-net` arguement is usedd to choose the network architecture to be used. An example is shown below and the flag `-h` can be used to display all options.

```bash
python learn.py -net EmbeddingRNN
```

### Translating

To translate some text run the `translate.py` script,  followed by the network architecture of choice using the `-net` flag.
In the terminal window, type in the English phrase you need to trannslate and hit return.

```bash
python translate.py -net EmbeddingRNN
```

