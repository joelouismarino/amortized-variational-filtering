model_config = {
    'architecture': 'vrnn',
    'inference_procedure': 'gradient',
    'modified': True,
    'global_output_log_var': False,
}

assert model_config['inference_procedure'] in ['direct', 'gradient', 'error', 'sgd']

# update type for iterative inference ('highway', 'learned_sgd')
# whether or not to concatenate the observation to inference model input
# normalization type for the inputs ('layer', 'batch', None)
if model_config['inference_procedure'] in ['gradient', 'error']:
    model_config['update_type'] = 'learned_sgd'
    model_config['concat_observation'] = False
    model_config['input_normalization'] = 'layer'
    model_config['norm_parameters'] = True

if model_config['inference_procedure'] == 'sgd':
    model_config['learning_rate'] = 0.01

## SVG
if model_config['architecture'].lower() == 'svg':
    model_config['model_type'] = 'kth_actions'

## VRNN
if model_config['architecture'].lower() == 'vrnn':
    model_config['model_type'] = 'timit'

## SRNN
if model_config['architecture'].lower() == 'srnn':
    model_config['model_type'] = 'midi'
