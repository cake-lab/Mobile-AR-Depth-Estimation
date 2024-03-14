from efm_datasets.utils.config import read_config
from efm_datasets.utils.setup import setup_dataset
from scripts.display.DisplayDataset import DisplayDataset

import sys

config = sys.argv[1] 
name = sys.argv[2]

cfg = read_config(config)
dataset = setup_dataset(cfg.dict[name])[0]

display = DisplayDataset(dataset)
display.loop()
