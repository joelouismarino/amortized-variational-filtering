# Amortized Variational Filtering
PyTorch implementation of (amortized) variational filtering

## Data IO To-Do:
- [x] Data loading pipeline.
- [x] Option to resize a to standard size.
- [x] Implement data transformations (crop, etc.).
- [x] Batch data loading (multiple videos).
- [x] Implement other data loading options (directory of files).
- [x] Load standard benchmark sequence data sets.
- [x] Rewrite TIMIT dataset.
- [x] Get Blizzard dataset working.
- [ ] Get IAM OnDB dataset working.
- [ ] Get YouTube dataset working again. Make closer to PyTorch dataset.

## Model To-Do:
- [x] Re-implement low-level modules.
- [x] Implement convolutional model.
- [x] Incorporate multiple sample draws.
- [x] Implement neural network dynamics model.
- [x] Fix issue with using GPU 1.
- [x] Fix issue with dimensions for using errors, gradients.
- [x] Implement latent variable model base class.
- [x] Use CDF for Gaussian conditional log likelihood evaluation.
- [x] Implement SVG
- [x] Implement VRNN
- [x] Implement SRNN
- [ ] Implement DKS

## Overall To-Do:
- [x] Figure out why the loss is becoming negative...it was an issue with the sampling dimension.
- [x] Implement saving and loading of models, optimizers.
- [x] Find a set of simple videos to benchmark the model on.
- [x] Plot the losses.
- [ ] Output video, audio, handwriting reconstructions.
- [ ] Option to load config from experiment directory
- [ ] Plot losses on same plot when resuming a checkpoint.
- [ ] Fix all random seeds.
