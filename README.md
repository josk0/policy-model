# Policy Model

## Summary

just an idea…

## Installation
recreate environment, e.g. using

``` bash
conda env create --file environment.yml
```

or manually create environment
```bash
conda install install python numpy pandas tqdm matplotlib solara pytest scipy ipython networkx
python -m pip install mesa
```
(Note: the mesa package is installed with pip since it's not up-to-date in conda-forge)

## How to run
To run the model interactively, in this directory, run the following command

```bash
solara run app.py
```

## Files

* ``model.py``: Contains the agent class, and the overall model class.
* ``agents.py``: Contains the agent class.
* ``app.py``: Contains the code for the interactive Solara visualization.

## Acknowledgement

based on "Virus on a Network" example of mesa.