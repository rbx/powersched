#### powersched

`python ./checkenv.py` - Check the environment.

`python ./testenv.py` - Short training run.

`python ./train.py` - Infinite training run with tensorboard logs and intermediate models save.

`python ./train_iter.py` - Sequentially launch `./train.py` with different weights.

`./train.py` accepts `--render` argument with "human" or "none" ("none" is the default). "none" trains silently, while "human" runs intentionally slower, adds some debug output and graph output after each episode.

