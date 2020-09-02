# Amortized Variational Filtering

Code to accompany the paper [A General Method for Amortizing Variational Filtering](https://arxiv.org/abs/1811.05090) by Marino et al., NeurIPS, 2018.

## Installation & Environment Set-Up

First, clone the repository by opening a terminal and running:
```
$ git clone https://github.com/joelouismarino/amortized-variational-filtering
```
Then enter the project directory:
```
$ cd amortized-variational-filtering
```
To avoid dependency conflicts, create a conda environment using Anaconda, which you can download [here](https://www.anaconda.com/download/#linux). Once you have installed Anaconda, use the environment file, `avf_env.yml`, to create the environment and install the dependencies:
```
$ conda env create -f avf_env.yml
```
To activate the environment, run:
```
$ conda activate avf
```
The terminal should now appear as follows:
```
(avf) $
```
Within the environment, install PyTorch by visiting the list of versions [here](https://pytorch.org/previous-versions/), and installing version `0.3.0.post4` for your version of CUDA (`8.0` or `9.0`). For example, with CUDA `9.0`, you would run:
```
(avf) $ pip install http://download.pytorch.org/whl/cu90/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl
```
Note: if you want to use the [BAIR Robot Pushing](https://arxiv.org/pdf/1710.05268.pdf) dataset, you will also need to install [tensorflow](https://www.tensorflow.org/). At this point, you should be able to run code within the AVF repository. After running the code, when you are ready to exit the environment, run:
```
(avf) $ conda deactivate
```
or simply close out of the terminal window.

## Training/Running Models

### Starting visdom

The code uses [visdom](https://github.com/facebookresearch/visdom) to plot various metrics during training. To start visdom, open a terminal window, activate the environment (see above), and run:
```
(avf) $ python -m visdom.server
```
You can now open a web browser and navigate to `http://localhost:8097/` to view the visdom output. Each experiment is saved as separate visdom environment using the time stamp of when the training run was started.

### Configuration

Each experiment is defined by a set of configuration dictionaries, which are found in the `config` directory. These consist of `train_config`, `run_config`, `data_config`, and `model_config`. The `train_config` dictionary contains training specifications, such as the batch size, learning rate, etc. The `run_config` dictionary contains specifications for the current run, such as which GPU to use, where to log the results, etc. The `data_config` dictionary contains specifications for the data, like the dataset and sequence length. Finally, the `model_config` contains the specifications for the generative and inference models.

You can define an experiment by altering each of the config files in the `config` directory. Alternatively, you can use one of the pre-defined configurations from the paper, which are located in the `experiments` directory. We will now walk through how to run the code with each of these approaches.

### Starting an experiment

To run the code with your own set of configuration parameters (located in the `config` directory), run:
```
(avf) $ python train.py
```
To instead run one of the pre-defined experiment configurations, use the command line arguments `--dataset`, `--model`, and `--inference`. For example, to run SVG on the KTH Actions dataset with AVF, run:
```
(avf) $ python train.py --dataset 'kth_actions' --model 'svg' --inference 'avf'
```
To run the baseline filtering method (proposed in [Denton et al., 2018](https://arxiv.org/abs/1802.07687)), run:
```
(avf) $ python train.py --dataset 'kth_actions' --model 'svg' --inference 'baseline'
```
This will load configuration parameters for `model_config` and `train_config` from the corresponding experiment in the `experiments` directory. Note that you will still need to define `config/run_config.py`, as well as `data_path` in `config/data_config.py`. If the dataset has not been downloaded yet, the code will automatically download the data to the `data_path`. Note that not all models can be run with all datasets. The TIMIT dataset must be downloaded manually [here](https://github.com/philipperemy/timit).

### Evaluation

Evaluation for some datasets is performed on each test sequence separately, i.e. not cropping out sub-sequences. Since test sequences can be of various lengths, you have to perform evaluation with a `batch_size` (see `config/train_config.py`) of 1. To evaluate a model, specify the `resume_path` in `config/run_config.py`, then run:
```
(avf) $ python evaluate.py
```
This will run evaluation and save the results using `pickle`.

## Extending the code

The repository is set-up to be fairly straightforward to extend to new datasets and new models. We'll now briefly describe how to do each of these.

### New datasets

The datasets are defined in `util/data/datasets/`. To add a new dataset, create a class for the dataset in this directory and import the dataset in `util/data/datasets/__init__.py`. For example, you could define a dataset called `MyDataset` in a file `util/data/datasets/my_dataset.py`:
```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, args):
        # define initialization here

    def __getitem__(self, index):
        # define getitem method here

    def __len__(self):
        # define dataset length method here
```
Then, in `util/data/datasets/__init__.py`, include the following line:
```python
from my_dataset import MyDataset
```
The datasets are loaded in `util/data/load_dataset.py`. Add another `elif` statement in this file for your dataset and create `train`, `val`, and `test` datasets. For example,
```python
    elif dataset_name == 'my_dataset':
        # do whatever downloading, etc. you need to do here
        from datasets import MyDataset
        train = MyDataset(train_args)
        val = MyDataset(val_args)
        test = MyDataset(test_args)
```
Note that you may want to include data transforms in the dataset (e.g. random sequence crops, image crops/flips, normalization, etc.). See the other datasets in `load_dataset.py` for examples of how to include these. Once you have added your dataset to `load_dataset.py`, you can use your dataset by specifying the name for `dataset_name` in `config/data_config.py`. You can also specify the `data_type` to add additional arguments to `data_config`. When in doubt, use `other`. You will also need to specify the length of sequences for training. Here, we set this to `10`. For example,
```python
data_config = {
    'data_path': '/path/to/datasets/',
    'dataset_name': 'my_dataset',
    'data_type': 'other', # video, audio, other, etc.
    'sequence_length': 10
}
```
Note that the current set of models have different versions for various datasets, e.g. based on the number of data dimensions, etc. You can specify which version of the model to use in `config/model_config.py`, and/or define a new version of the model in the corresponding model file found in `lib/models/`.

### New models

To add a new model to the repository, you will need to define it in `lib/models/` as a separate file. All models inherit from the base class `LatentVariableModel` defined in `lib/models/latent_variable_model.py`, which defines the set of methods that are expected in the code. These required methods are
* `infer`: perform inference for a given data input step
* `generate`: generate a prediction/reconstruction from the model
* `step`: step the latent dynamics forward one step
* `re_init`: re-initialize the model's state
* `inference_parameters`: returns the parameters of the inference model
* `generative_parameters`: returns the parameters of the generative model
* `inference_mode`: puts the model into inference mode, only backpropagating into the inference model
* `generative_mode`: puts the model into generative mode, backpropagating through the generative model.

You can also internally define whatever other methods you need for your model. For instance, the models are currently implemented with an internal `_construct` method. To create a model called `MyModel`, you would create a new file in `lib/models` called `my_model.py` with something like the following:
```python
from latent_variable_model import LatentVariableModel

class MyModel(LatentVariableModel):
    def __init__(self, model_config):
        super(MyModel).__init__(model_config)
        # initialize the model here

    def infer(self, observation):
        # define inference method here

    ... # define the rest of the required methods (see above)
```

Then, in the file `lib/models/load_model.py`, add another `elif` statement for your model:

```python
    elif model_name == 'my_model':
        from my_model import MyModel
        return MyModel(model_config)
```

Finally, to use your model, specify your model in the `architecture` field in `config/model_config.py`. For instance:

```python
model_config = {
    'architecture': 'my_model',
    ...
}
```

## Contact

If you have any questions, feel free to send me an email at `jmarino` [at] `caltech` [dot] `edu`, or post an issue on Github.
