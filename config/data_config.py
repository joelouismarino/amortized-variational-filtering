data_config = {
    'data_path': '/media/joe/SSD/datasets/',
    'dataset_name': 'timit',
    'data_type': 'audio', # video, audio, tracking, other
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

    if data_config['data_type'] == 'tracking':
        pass

    if data_config['dataset_name'] == 'youtube':
        data_config['youtube_url_file'] = 'data/youtube_8m_urls/Science.txt'

clean_data_config(data_config)
