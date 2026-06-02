# DeepLogger

Machine-Learning used for borehole televiewer data interpretation


Optional (suggested): Create a conda environment:
```
conda create -n deeplogger python=3.11
conda activate deeplogger
```

1. Clone the repository:
```
git clone https://gitlab.com/shakasa/deeplogger.git

cd deeplogger

```
2. Install the package locally by using the file `setup.py` by doing:
```
python -m pip install -e .
```

3. Download data from polybox using pooch

In the terminal type:
```
bash python deeplogger/download_data.py
```

Running this code a second time will not trigger a download since the file already exists.
