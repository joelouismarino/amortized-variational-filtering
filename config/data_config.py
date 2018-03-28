data_config = {
    'data_path': '/media/joe/SSD/datasets/',
    'dataset_name': 'BAIR_robot_pushing',
    'data_type': 'video',
    'seqeunce_length': 20,
}

if data_config['data_type'] == 'video':
    data_config['img_size'] = [128, 128]
    data_config['img_hz_flip'] = True
    data_config['img_rotation'] = 0
    data_config['img_crop'] = [128, 128]

if data_config['dataset_name'] == 'youtube':
    data_config['youtube_url_file'] = 'data/youtube_8m_urls/Science.txt'
