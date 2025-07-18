# Mvitime

This is the code implementation of paper  *Contrastive Learning for Sleep Staging based on Inter Subject Correlation*. More details about this paper could be known in [here](https://arxiv.org/abs/2305.03178) or [here](https://doi.org/10.1007/978-3-031-44213-1_29).

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


## 📄 Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{zhang2023contrastive,
  title={Contrastive Learning for Sleep Staging based on Inter Subject Correlation},
  author={Zhang, Tongxu and Wang, Bei},
  booktitle={International Conference on Artificial Neural Networks},
  pages={343--355},
  year={2023},
  organization={Springer}
}
