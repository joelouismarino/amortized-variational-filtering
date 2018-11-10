model_config = {
    'architecture': 'vrnn',
    'inference_procedure': 'direct',
    'modified': False,
    'global_output_log_var': False,
    'normalize_latent_samples': False
}

def clean_model_config(model_config):
    assert model_config['inference_procedure'] in ['direct', 'gradient']

    # update type for iterative inference ('highway', 'learned_sgd')
    # whether or not to concatenate the observation to inference model input
    # normalization type for the inputs ('layer', 'batch', None)
    if model_config['inference_procedure'] in ['gradient']:
        model_config['update_type'] = 'learned_sgd'
        model_config['concat_observation'] = False
        model_config['input_normalization'] = 'layer'
        model_config['norm_parameters'] = True

    ## SVG
    if model_config['architecture'].lower() == 'svg':
        model_config['model_type'] = 'kth_actions'

    ## VRNN
    if model_config['architecture'].lower() == 'vrnn':
        model_config['model_type'] = 'timit'

    ## SRNN
    if model_config['architecture'].lower() == 'srnn':
        model_config['model_type'] = 'midi'

clean_model_config(model_config)
