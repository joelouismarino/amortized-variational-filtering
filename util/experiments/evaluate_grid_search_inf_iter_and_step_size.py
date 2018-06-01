import os
import subprocess

import json
import time
from itertools import cycle

from config.data_config import clean_data_config
from config.model_config import clean_model_config

run_config = {
    'save_iter': 100,
    'resume_path': None,
    'log_root_path': '/home/ubuntu/logs/avf_logs/evaluate_grid_search_inf_iter_and_step_size',
}

train_config = {
    'batch_size': 1,
    'sequence_samples': 1,
    'optimizer': 'adam',
    'optimize_inf_online': True,
    'inference_learning_rate': 0.0001,
    'generation_learning_rate': 0.0001,
    'clip_grad_norm': None,
    'kl_annealing_epochs': 0,
}

data_config = {
    'data_path': '/home/ubuntu/datasets/',
    'dataset_name': 'timit',
    'data_type': 'audio',  # video, audio, tracking, other
    'sequence_length': 40,
}
clean_data_config(data_config)

model_config = {
    'architecture': 'vrnn',
    'inference_procedure': 'gradient',
    'modified': True,
    'global_output_log_var': False,
    'normalize_latent_samples': False,
}
clean_model_config(model_config)

cuda_device = cycle(range(0, 16))
experiments = os.listdir('/home/ubuntu/logs/avf_logs/grid_search_inf_iter_and_step_size')
for step_samples in [1, 2, 4, 8]:
    for inference_iterations in [1, 2, 4, 8]:
        train_config['step_samples'] = step_samples
        train_config['inference_iterations'] = inference_iterations
        run_config['cuda_device'] = cuda_device.next()
        run_config['log_dir'] = '{}_step_{}_iter_'.format(step_samples, inference_iterations)
        experiment = [i for i in experiments if i.startswith(run_config['log_dir'])]
        assert len(experiment) == 1
        run_config['resume_path'] = os.path.join('/home/ubuntu/logs/avf_logs/grid_search_inf_iter_and_step_size', experiment[0])

        r = json.dumps(run_config)
        t = json.dumps(train_config)
        d = json.dumps(data_config)
        m = json.dumps(model_config)
        cmd = "python {} '{}' '{}' '{}' '{}'".format(os.path.join(os.path.abspath('.'), 'evaluate.py'), r, t, d, m)
        print('Running: {}'.format(cmd))
        subprocess.Popen(cmd, shell=True, env=os.environ.copy()) #, stdout=open(os.devnull, 'w')
        time.sleep(1)
