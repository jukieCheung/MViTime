# Mvitime

This is the code implementation of paper  *Contrastive Learning for Sleep Staging based on Inter Subject Correlation*. More details about this paper could be known in https://arxiv.org/abs/2305.03178

## Preperation

### Environment Setup 

We recommend to setup the environment through `conda`.

```shell
$ conda env create -f environment.yml
```

### Data Preparation

The full dataset Sleep-edf can be downloaded [here](https://physionet.org/content/sleep-edfx/1.0.0/).

But the dataset we are using is a processed dataset with the same size as AttnSleep, so you can also obtain the data we are using [here](https://researchdata.ntu.edu.sg/dataverse/attnSleep), sourced from the author team of AttnSleep.

Notice! 
Please note that we have merged the individual record data files for each subject into one file for input into our Dataloader. In fact, you can also try writing a dataset.py file yourself to load the record data for each individual subject.
