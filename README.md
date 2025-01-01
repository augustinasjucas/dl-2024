# dl-2024

`reproduce.py` stores a sample experiment that aims to reproduce Florian's results.

In order to implement a metric, you just need to implement Metric class. Also, you can (and probably should) add wandb logging for that metric. Not that it would be best if for every metric you did logging only in the `after_task` method (otherwise, stuff might mess up, if you are not using `sweepy_logging`!).
For metric implementations, check `FlorianProbing` for an example - there only `after_all_tasks` was useful, but in other metrics other callback methods might be useful. 

Note that `FlorianProbing` has this (important!) flag `sweepy_logging`. If it is set to False, you get very nice graphs in W&B. However it does not allow for sweeps to be run - if you used `sweepy_logging=False`, you would have to have a separate script that starts a bunch of `reproduce.py`s with different parameters, and then the different sweeps would have to be different projects. However, if you set `sweepy_logging=True` and also use SweepyLogging class instead of BeautifulLogging class, every experiment WILL be a SINGLE run, so you can do sweeps (logging is configured nicely there). However, W&B graphs will suck in that case.

