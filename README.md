# Fraud detection in synthetic financial dataset

Classification of fraudulent transactions using a synthetic dataset generated using the simulator called PaySim.

## Dataset

[Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/ntnu-testimon/paysim1)

## Usage

You should have [virtualenv](https://virtualenv.pypa.io/en/stable/) installed. Clone the repository and create a Python environment with 

```bash
  virtualenv env
```

Install all dependencies

```bash
  env/bin/pip install -r requirements.txt
```

It might be necessary to give execution permission to the Python files with `chmod +x`. It is assumed that you've unzipped the dataset and you've renamed it to `data.csv`. To run a file (e.g. `visualize.py`) do

```bash
  ./visualize.py
```
