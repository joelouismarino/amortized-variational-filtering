train_config = {
    'batch_size': 20,
    'inference_iterations': 1,
    'sequence_samples': 1,
    'step_samples': 1,
    'optimizer': 'adam',
    'optimize_inf_online': False,
    'inference_learning_rate': 0.001,
    'generation_learning_rate': 0.001,
    'clip_grad_norm': None,
    'kl_annealing_epochs': 0,
}
