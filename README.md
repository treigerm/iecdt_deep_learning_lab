# IECDT Deep Learning Lab

## Lab Instructions

This repository enables two main pieces of functionality:
- Fine-tune/train a ResNet model to predict cloud fraction from GOES-16 satellite imagery
- Train an autoencoder to learn to compress the GOES-16 satellite images into a vector representation
See the sections below for more details on how to run the code.

The goal of the lab is for you to pick a specific research question that you want to investigate over the coming week.
The code is just meant to provide a starting point for your experiments.
It is purposefully minimal to encourage your own exploration.

### Data

Additionally, each tile also has associated labels which are all computed from the GOES-16 cloud mask:
- Cloud Fraction: Percentage of tile that is covered by clouds
- Mean Cloud Lengthscale: Average size cloud in the tile
- Cloud iOrg: Cloud organization index. See Appendix A of [Organization of tropical convection in low vertical wind shears: Role of updraft entrainment](https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2016MS000802) for an explanation of the index.

### Potential Research Questions

If you struggle to come up with research questions, here are some inspirations:
- Get best possible performance on predicting labels (lowest mean-squared error):
  - Freeze/fine-tune pre-trained model or train from scratch?
  - Hyperparameter tuning (batch size, learning rate, …)
  - Effect of using different model sizes and/or architectures?
    - You can have a look at the list of available pre-trained models [here](https://pytorch.org/vision/main/models.html#classification)
  - Effect of data augmentations (e.g. horizontally flipping input images)?
- Can we discover meaningful cloud regimes through unsupervised learning?
  - Train autoencoder and look at clusters in latent space. If you have not heard about autoencoders before Section 14.1 of the [deep learning book](https://www.deeplearningbook.org/contents/autoencoders.html) provides a good introduction.
  - Can we predict the cloud labels from the learned embeddings? Does this perform better or worse than using a pre-trained model?
  - Does the reconstruction quality increase if we use alternative autoencoders? Some alternative architectures you could try:
    - Variational Autoencoder (VAE)
    - [Vector-quantized VAE (VQ-VAE)](https://arxiv.org/abs/1711.00937)

If you are new to neural networks then Chapters 13, 14, and 16 of the [fastai book](https://fastai.github.io/fastbook2e/convolutions.html) provide an easy to read introduction.
Even if you trained neural networks before the book provides helpful advice for common problems encountered in practice.


## Setup

### Fork the repository

Fork the repository so that you can push any changes you make to your own forked version of the project. Then clone the repository into your workspace on JASMIN.

### Install dependencies

[Login to JASMIN](https://help.jasmin.ac.uk/docs/getting-started/how-to-login/):
```bash
ssh -A <YOUR JASMIN USERNAME>@login-01.jasmin.ac.uk
```
From there log into the interactive GPU node:
```bash
ssh -A gpuhost001.jc.rl.ac.uk
```

Then install [uv](https://docs.astral.sh/uv/) for dependency management:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To install the virtual environment for this project, first navigate to the project repository:
```bash
cd path/to/iecdt_deep_learning_lab
```
Then run the following commands:
```bash
uv venv --seed
uv pip install -e .
```

### JASMIN GPU Access

To test and debug your code, you want to launch an interactive SLURM session to get access to a GPU (the `gpuhost001.jc.rl.ac.uk` only has 4 GPUs, if all CDT students use them at the same time they will be quickly oversubscribed):
```bash
srun --gres=gpu:1 --partition=orchid --account=orchid --cpus-per-task=3 --pty /bin/bash
```
More help on accessing the GPUs on JASMIN can be found [here](https://help.jasmin.ac.uk/docs/batch-computing/orchid-gpu-cluster/).


## Running Pre-Trained ResNet

Export Weights and Biases [API key](https://docs.wandb.ai/support/find_api_key/):
```bash
export WANDB_API_KEY=<YOUR API KEY HERE>
```
The code uploads the results to a project named `iedct_deep_learning_lab` so ensure you have a project with that name set up or change the project name in `config_resnet/wandb/default.yaml`.

You can use the following command to check whether your environment is setup correctly:
```bash
uv run train_resnet.py wandb.mode="offline" smoke_test=True log_freq=10 device="cuda"
```
The script uses [Hydra](https://hydra.cc/) to manage configuration files.
For the ResNet training the configuration files can be found in the `config_resnet` directory.
All the default parameters specified in the config files can be changed from the command-line, as the command above demonstrates.
Have a look at the `main` function in `train_resnet.py` to see how the configuration parameters can be accessed from the training script.

Running the `train_resnet.py` script in an interactive SLURM session should only be done to test your code.
For longer training runs you want to submit a SLURM batch job.
Luckily, you can use the `train_resnet.py` script to submit SLURM batch jobs!
When calling the training script with the `-m` flag, the scripts will automatically be submitted to run as a SLURM job (the name can be used to filter groups of submissions in Weights and Biases):
```bash
uv run train_resnet.py -m name="test_slurm_submission" device="cuda"
```
This will submit the SLURM jobs and then wait until the SLURM jobs have terminated.
Usually, you can just kill the execution of the script after the SLURM jobs have been launched using `CTRL-C`.
Run 
```bash
squeue -u <YOUR JASMIN USERNAME>
```
to check that the SLURM jobs are succesfully running.
You can also adjust the SLURM job parameters, i.e. to change the number of requested CPU cores:
```bash
uv run train_resnet.py -m name="test_slurm_submission" device="cuda" hydra.launcher.cpus_per_task=6
```
See `config_resnet/hydra/launcher/slurm.yaml` for the default SLURM parameters.

The `-m` flag is particularly useful for submitting a sequence of jobs, i.e. running:
```bash
uv run train_resnet.py -m name="test_slurm_submission" device="cuda" learning_rate=0.01,0.1
```
will submit two SLURM jobs.
One that runs the training script `learning_rate=0.01` and another with `learning_rate=0.1`.
For more details about the Hydra submitit plugin consult [this page](https://hydra.cc/docs/plugins/submitit_launcher/).

### Experiment logs

For each execution of the training script, the experiment logs will be stored in `logs/{name}/runs/{CURRENT_DATA_AND_TIME}/` (or `logs/{name}/multiruns/{CURRENT_DATA_AND_TIME}/` if you use the `-m` flag).
Note that the training script `train_resnet.py` automatically changes its working directory to the logging directory of the run, i.e. if you call `torch.save(model.state_dict(), "model.pth")` from within the training script it saves the `model.pth` file in `logs/{name}/runs/{CURRENT_DATA_AND_TIME}/`. 

Hydra will also save the explicit configuration of the parameters that were used for this experiment in the logging directory, as well as a log of the output written by the `logging.info` commands.

### Notebook to analyse final layer embeddings

The notebook at `notebooks/embedding_analysis.ipynb` briefly demonstrates how you can analyse the embedding space of your trained ResNet models. 
It can be interesting to compare the embedding space of models that are trained differently (e.g. fine-tuned vs. trained from scratch).

## Running Autoencoder

The `train_autoencoder.py` script trains an autoencoder architecture specified in the `iecdt_lab/autoencoder.py` file with hyperparameters specified in `config_ae`.
You can run the script in the same manner as the `train_resnet.py` script, e.g.
```bash
uv run train_resnet.py wandb.mode="offline" smoke_test=True log_freq=10 device="cuda"
```
for debugging.

## Code layout

```
config_ae/ 
config_resnet/
  hydra/
    launcher/
      slurm.yaml            # SLURM job submission parameters
    default.yaml            # Hydra configuration parameters
  wandb/
    default.yaml            # Weights and Biases configuration parameters
  train.yaml                # Hyperparameters for ResNet training
iecdt_lab/
  __init__.py
  autoencoder.py            # Autoencoder implementation
  data_loader.py            # PyTorch Dataset
logs/
notebooks/
  embedding_analysis.ipynb  # Demonstrates how to get embeddings of trained models.
.gitignore
.python-version             # Specifies Python version
pyproject.toml              # Specifies dependency constraints
uv.lock                     # Exact version information about dependencies
train_autoencoder.py
train_resnet.py
```