data_config = {
    'data_path': '/media/joe/SSD/datasets/',
    'dataset_name': 'BAIR_robot_pushing',
    'seqeunce_length': 20,

    'img_size': [64, 64],
    'img_hz_flip': True,
    'img_rotation': 0,
    'img_crop': [64, 64],
}

if data_config['dataset_name'] == 'youtube':
    data_config['youtube_url_file'] = 'data/youtube_8m_urls/Science.txt'
