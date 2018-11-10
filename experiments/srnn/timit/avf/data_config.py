data_config = {
    'data_path': '/path/to/datasets/',
    'dataset_name': 'timit',
    'data_type': 'audio', # video, audio, other, etc.
    'sequence_length': 40,
}

def clean_data_config(data_config):
    if data_config['data_type'] == 'video':
        data_config['crop_size'] = [64, 64]
        data_config['img_size'] = [64, 64]
        data_config['img_hz_flip'] = False
        data_config['img_rotation'] = 0

    if data_config['data_type'] == 'audio':
        data_config['window'] = 200

clean_data_config(data_config)
