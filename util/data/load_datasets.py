import os
from misc_data_util.url_save import save
import misc_data_util.tarfile_progress as tarfile


def load_datasets(dataset_name, data_path):
    """
    Downloads and loads a variety of standard benchmark sequence datasets.
    Arguments:
        dataset_name: the name of the dataset to loading
        data_path: path to directory where data should be saved
    Returns:

    """

    assert os.path.exists(data_path), 'Data path not found. Please specify a valid path.'

    if dataset_name in ['blizzard', 'BLIZZARD']:
        if not os.path.exists(os.path.join(data_path, 'blizzard')):
            os.makedirs(os.path.join(data_path, 'blizzard'))
        if not os.path.exists(os.path.join(data_path, 'blizzard', 'blizzard_wavs_and_scores_2016_release_version_1')):
            print('Downloading blizzard dataset...')
            # save('http://data.cstr.ed.ac.uk/blizzard/wavs_and_scores/blizzard_wavs_and_scores_2016_release_version_1.tar.bz2',
            #                     os.path.join(data_path, 'blizzard', 'blizzard_wavs_and_scores_2016.tar.bz2'))
            # print('Done.')
            # print('Extracting tar file...')
            # tar = tarfile.open(os.path.join(data_path, 'blizzard', 'blizzard_wavs_and_scores_2016.tar.bz2'))
            # tar.extractall(os.path.join(data_path, 'blizzard'), progress = tarfile.progressprint)
            # tar.close()
            # os.remove(os.path.join(data_path, 'blizzard', 'blizzard_wavs_and_scores_2016.tar.bz2'))
            print('Done.')


    if dataset_name == 'timit':
        if not os.path.exists(os.path.join(data_path, 'timit')):
            os.makedirs(os.path.join(data_path, 'timit'))
        if not os.path.exists(os.path.join(data_path, 'timit', ))

    if dataset_name == 'onomatopoeia':
        if not os.path.exists(os.path.join(data_path, 'onomatopoeia')):
            os.makedirs(os.path.join(data_path, 'onomatopoeia'))

    if dataset_name == 'accent':
        if not os.path.exists(os.path.join(data_path, 'accent')):
            os.makedirs(os.path.join(data_path, 'accent'))

    if dataset_name == 'iam_ondb':
        if not os.path.exists(os.path.join(data_path, 'iam_ondb')):
            os.makedirs(os.path.join(data_path, 'iam_ondb'))
