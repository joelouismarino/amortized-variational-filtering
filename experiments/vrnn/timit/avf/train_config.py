train_config = {
    'batch_size': 64,
    'inference_iterations': 2,
    'sequence_samples': 1,
    'step_samples': 1,
    'optimizer': 'adam',
    'optimize_inf_online': True,
    'inference_learning_rate': 0.0001,
    'generation_learning_rate': 0.0001,
    'clip_grad_norm': None,
    'kl_annealing_epochs': 0,
}
