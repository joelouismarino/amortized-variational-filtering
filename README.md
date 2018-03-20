# Amortized Variational Filtering
PyTorch implementation of (amortized) variational filtering

## Data IO To-Do:
- [x] Data loading pipeline.
- [x] Option to resize a to standard size.
- [x] Implement data transformations (crop, etc.).
- [x] Batch data loading (multiple videos).
- [x] Implement other data loading options (directory of files).
- [x] Load standard benchmark sequence data sets.
- [ ] Get YouTube dataset working again. Make closer to PyTorch dataset.

## Model To-Do:
- [x] Re-implement low-level modules.
- [x] Implement convolutional model.
- [x] Incorporate multiple sample draws.
- [x] Implement neural network dynamics model.
- [x] Fix issue with using GPU 1.
- [x] Fix issue with dimensions for using errors, gradients.
- [x] Implement latent variable model base class.
- [ ] Fix issue with dimensions for using concat networks.
- [ ] Implement baseline dynamical models (VRNN, SRNN, DKS, etc.)
- [ ] Implement inference techniques for other models.

## Overall To-Do:
- [x] Figure out why the loss is becoming negative...it was an issue with the sampling dimension.
- [x] Implement saving and loading of models, optimizers.
- [x] Find a set of simple videos to benchmark the model on.
- [ ] Plot the losses.

## For (Much) Later
- [ ] Implement change of variables on observed variables.
- [ ] Implement change of variables on approx. posterior and prior.
- [ ] Make separate state / cause latent variables.
- [ ] Generalized coordinates.
