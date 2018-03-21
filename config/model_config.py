model_config = {
    'architecture': 'svg',
    'inference_procedure': 'direct'


    # 'encoding_form': ['posterior'],
    # 'concat_levels': False,
    # 'transform_input': False, # currently has no effect
    # 'constant_prior_variances': False,
    # 'output_distribution': 'gaussian', # currently only supported distribution
    #
    # 'n_latent': [64],
    #
    # 'n_det_enc': [0,],
    # 'n_det_dec': [0, 0],
    #
    # 'n_layers_enc': [3],
    # 'n_layers_dec': [3, 2],
    #
    # 'n_filters_enc': [64],
    # 'n_filters_dec': [64, 64],
    #
    # 'filter_size': [5, 5],
    #
    # 'non_linearity_enc': 'elu',
    # 'non_linearity_dec': 'elu',
    #
    # 'connection_type_enc': 'highway',
    # 'connection_type_dec': 'highway',
    #
    # 'batch_norm_enc': False,
    # 'batch_norm_dec': False,
    #
    # 'weight_norm_enc': False,
    # 'weight_norm_dec': False,
    #
    # 'dropout_enc': 0.0,
    # 'dropout_dec': 0.0
}

## SVG
if model_config['architecture'].lower() == 'svg':
    model_config['model_type'] = 'bair_robot_pushing'

## VRNN
if model_config['architecture'].lower() == 'vrnn':
    model_config['model_type'] = 'blizzard'
