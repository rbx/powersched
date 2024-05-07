#### powersched

`python ./power_checkenv.py` - check the environment.

`python ./power_testenv.py` - a short training run.

`python ./power_train.py` - infinite training run with tensorboard logs and intermediate models save.

`./power_train.py` accepts `--render` argument with "human" or "none" ("none" is the default). "none" trains silently, while "human" runs intentionally slower, adds some debug output and graph output after each episode.

