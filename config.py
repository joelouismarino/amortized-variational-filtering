run_config = {
    'batch_size': 1,
    'inference_iterations': 1,
    'cuda_device': 1,
    'resume_path': None,
    'experiment_path': '' # currently has no effect
}

train_config = {
    'optimizer': 'adam',
    'encoder_learning_rate': 0.00001,
    'decoder_learning_rate': 0.00001
}

data_config = {
    'data_path': '/media/joe/SSD/datasets/',
    'dataset_name': 'BAIR_robot_pushing',
    'seqeunce_length': 20,

    'img_size': [64, 64],
    'img_hz_flip': True,
    'img_rotation': 0,
    'img_crop': [64, 64],

    'youtube_url_file': 'data/youtube_8m_urls/Science.txt',
}

model_config = {
    'encoding_form': ['posterior'],
    'concat_levels': False,
    'transform_input': False, # currently has no effect
    'constant_prior_variances': False,
    'output_distribution': 'gaussian', # currently only supported distribution
    'normalizing_flows': False, # currently has no effect

    'n_latent': [64],

    'n_det_enc': [0,],
    'n_det_dec': [0, 0],

    'n_layers_enc': [3],
    'n_layers_dec': [3, 2],

    'n_filters_enc': [64],
    'n_filters_dec': [64, 64],

    'filter_size': [5, 5],

    'non_linearity_enc': 'elu',
    'non_linearity_dec': 'elu',

    'connection_type_enc': 'highway',
    'connection_type_dec': 'highway',

    'batch_norm_enc': False,
    'batch_norm_dec': False,

    'weight_norm_enc': False,
    'weight_norm_dec': False,

    'dropout_enc': 0.0,
    'dropout_dec': 0.0
}
