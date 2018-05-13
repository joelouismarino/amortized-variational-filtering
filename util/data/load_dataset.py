import os
import torch
import cPickle
import tarfile
import misc_data_util.transforms as trans
from zipfile import ZipFile
from misc_data_util.url_save import save


def load_dataset(data_config):
    """
    Downloads and loads a variety of standard benchmark sequence datasets.
    Arguments:
        data_config (dict): dictionary containing data configuration arguments
    Returns:
        tuple of (train, val, test), each of which is a PyTorch dataset.
    """
    data_path = data_config['data_path'] # path to data directory
    if data_path is not None:
        assert os.path.exists(data_path), 'Data path not found.'

    dataset_name = data_config['dataset_name'] # the name of the dataset to load
    dataset_name = dataset_name.lower() # cast dataset_name to lower case
    train = val = test = None

    ############################################################################
    ## Speech datasets
    ############################################################################
    if dataset_name == 'blizzard':
        if not os.path.exists(os.path.join(data_path, 'blizzard')):
            raise ValueError('Blizzard dataset does not exist. Please manually \
                             download the dataset by obtaining a license from \
                             http://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/, \
                             then copy the file Lessac_Blizzard2013_CatherineByers_train.tar.bz2 \
                             into a directory named blizzard in your data directory.')

        # untar the data file
        if not os.path.exists(os.path.join(data_path, 'blizzard', 'data')):
            assert os.path.exists(os.path.join(data_path, 'blizzard',
                                  'Lessac_Blizzard2013_CatherineByers_train.tar.bz2'))
            print('Untarring Blizzard dataset...')
            tar = tarfile.open(os.path.join(data_path, 'blizzard',
                                  'Lessac_Blizzard2013_CatherineByers_train.tar.bz2'), "r:bz2")
            os.makedirs(os.path.join(data_path, 'blizzard', 'data'))
            tar.extractall(os.path.join(data_path, 'blizzard', 'data'))
            tar.close()
            os.remove(os.path.join(data_path, 'blizzard',
                        'Lessac_Blizzard2013_CatherineByers_train.tar.bz2'))
            print('Done.')

        # convert the data into train/val/test
        if not os.path.exists(os.path.join(data_path, 'blizzard', 'train')):
            print('Converting Blizzard dataset...')
            from misc_data_util.convert_blizzard import convert
            convert(os.path.join(data_path, 'blizzard'))
            print('Done.')

        from datasets import Blizzard
        mean, std = cPickle.load(open(os.path.join(data_path, 'blizzard', 'statistics.p'), 'r'))
        total_len = data_config['window'] * data_config['sequence_length']
        data_trans = trans.Compose([trans.RandomSequenceCrop(total_len),
                                    trans.Normalize(mean, std),
                                    trans.BinSequence(data_config['window']),
                                    trans.ToTensor()])

        train = Blizzard(os.path.join(data_path, 'blizzard', 'train'), data_trans)
        val = Blizzard(os.path.join(data_path, 'blizzard', 'val'), data_trans)
        test = Blizzard(os.path.join(data_path, 'blizzard', 'test'), data_trans)

    elif dataset_name == 'timit':
        if not os.path.exists(os.path.join(data_path, 'timit')):
            raise ValueError('TIMIT dataset does not exist. Please manually \
                             download the dataset from \
                             https://github.com/philipperemy/timit, \
                             then place the .zip file in a directory called \
                             timit in your data directory.')

        if os.path.exists(os.path.join(data_path, 'timit', 'TIMIT.zip')):
            print('Unzipping TIMIT dataset...')
            zip_ref = ZipFile(os.path.join(data_path, 'timit', 'TIMIT.zip'), 'r')
            zip_ref.extractall(os.path.join(data_path, 'timit'))
            zip_ref.close()
            os.remove(os.path.join(data_path, 'timit', 'TIMIT.zip'))
            print('Done.')

        if not os.path.exists(os.path.join(data_path, 'timit', 'train')):
            print('Converting TIMIT dataset...')
            from misc_data_util.convert_timit import convert
            convert(os.path.join(data_path, 'timit'))
            print('Done.')

        from datasets import TIMIT
        mean, std = cPickle.load(open(os.path.join(data_path, 'timit', 'statistics.p'), 'r'))
        total_len = data_config['window'] * data_config['sequence_length']
        data_trans = trans.Compose([trans.RandomSequenceCrop(total_len),
                                    trans.Normalize(mean, std),
                                    trans.BinSequence(data_config['window']),
                                    trans.ToTensor()])

        train = TIMIT(os.path.join(data_path, 'timit', 'train', 'train.p'), data_trans)
        val = TIMIT(os.path.join(data_path, 'timit', 'val', 'val.p'), data_trans)
        test = TIMIT(os.path.join(data_path, 'timit', 'test', 'test.p'), data_trans)

    ############################################################################
    ## Handwriting datasets
    ############################################################################
    elif dataset_name == 'iam_ondb':
        if not os.path.exists(os.path.join(data_path, 'iam_ondb')):
            raise ValueError('IAM_OnDB dataset does not exist. Please manually \
                             download the dataset by obtaining a license from \
                             http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database, \
                             then create a directory called iam_ondb in your data directory.')
        # TODO: load IAM_OnDB

    ############################################################################
    ## MIDI datasets
    ############################################################################
    elif dataset_name == 'piano_midi':
        if not os.path.exists(os.path.join(data_path, 'piano_midi')):
            os.makedirs(os.path.join(data_path, 'piano_midi'))
        if not os.path.exists(os.path.join(data_path, 'piano_midi', 'Piano-midi.de.pickle')):
            print('Downloading piano-midi dataset...')
            save('http://www-etud.iro.umontreal.ca/~boulanni/Piano-midi.de.pickle',
                 os.path.join(data_path, 'piano_midi', 'Piano-midi.de.pickle'))
            print('Done.')
        piano_roll = cPickle.load(os.path.join(data_path, 'piano_midi', 'Piano-midi.de.pickle'))
        # TODO: load piano_midi

    elif dataset_name == 'nottingham':
        if not os.path.exists(os.path.join(data_path, 'nottingham')):
            os.makedirs(os.path.join(data_path, 'nottingham'))
        if not os.path.exists(os.path.join(data_path, 'nottingham', 'Nottingham.pickle')):
            print('Downloading nottingham dataset...')
            save('http://www-etud.iro.umontreal.ca/~boulanni/Nottingham.pickle',
                 os.path.join(data_path, 'nottingham', 'Nottingham.pickle'))
            print('Done.')
        piano_roll = cPickle.load(os.path.join(data_path, 'nottingham', 'Nottingham.pickle'))
        # TODO: load nottingham

    elif dataset_name == 'muse':
        if not os.path.exists(os.path.join(data_path, 'muse')):
            os.makedirs(os.path.join(data_path, 'muse'))
        if not os.path.exists(os.path.join(data_path, 'muse', 'MuseData.pickle')):
            print('Downloading muse dataset...')
            save('http://www-etud.iro.umontreal.ca/~boulanni/MuseData.pickle',
                 os.path.join(data_path, 'muse', 'MuseData.pickle'))
            print('Done.')
        piano_roll = cPickle.load(os.path.join(data_path, 'muse', 'MuseData.pickle'))
        # TODO: load muse

    elif dataset_name == 'jsb_chorales':
        if not os.path.exists(os.path.join(data_path, 'jsb_chorales')):
            os.makedirs(os.path.join(data_path, 'jsb_chorales'))
        if not os.path.exists(os.path.join(data_path, 'jsb_chorales', 'JSBChorales.pickle')):
            print('Downloading jsb_chorales dataset...')
            save('http://www-etud.iro.umontreal.ca/~boulanni/JSBChorales.pickle',
                 os.path.join(data_path, 'jsb_chorales', 'JSBChorales.pickle'))
            print('Done.')
        piano_roll = cPickle.load(os.path.join(data_path, 'jsb_chorales', 'JSBChorales.pickle'))
        # TODO: load jsb_chorales

    ############################################################################
    ## Tracking datasets
    ############################################################################
    elif dataset_name == 'bball':
        assert os.path.exists(os.path.join(data_path, 'bball')), 'Basketball data not downloaded.'

        if not os.path.exists(os.path.join(data_path, 'bball', 'train', 'Xtr_role.p')):
            print('Converting basketball dataset...')
            from misc_data_util.convert_bball import convert
            convert(os.path.join(data_path, 'bball'))

        from datasets import Basketball
        LENGTH = 94
        WIDTH = 50
        SCALE = torch.zeros(50, 2)
        SCALE[:, 0] = LENGTH; SCALE[:, 1] = WIDTH
        transforms = [trans.ToTensor(), trans.Normalize(25., SCALE), trans.RandomSequenceCrop(data_config['sequence_length'])]
        transforms = trans.Compose(transforms)

        train = Basketball(os.path.join(data_path, 'bball', 'train', 'Xtr_role'), transforms)
        test = Basketball(os.path.join(data_path, 'bball', 'test', 'Xte_role'), transforms)

    ############################################################################
    ## Video datasets
    ############################################################################
    elif dataset_name == 'stochastic_moving_mnist':
        if not os.path.exists(os.path.join(data_path, 'stochastic_moving_MNIST')):
            os.makedirs(os.path.join(data_path, 'stochastic_moving_MNIST'))
            # download stochastic_moving_MNIST dataset
        # TODO: load stochastic_moving_MNIST

    elif dataset_name == 'kth_actions':
        if not os.path.exists(os.path.join(data_path, 'kth_actions')):
            os.makedirs(os.path.join(data_path, 'kth_actions'))
        if not os.path.exists(os.path.join(data_path, 'kth_actions', 'train')):
            print('Downloading KTH Actions dataset...')
            actions = ['walking', 'jogging', 'running', 'boxing', 'handwaving', 'handclapping']
            for action in actions:
                print('Downloading ' + action + '...')
                save('http://www.nada.kth.se/cvap/actions/' + action + '.zip',
                    os.path.join(data_path, 'kth_actions', action + '.zip'))
                print('\n')
            print('Done.')

            print('Unzipping KTH Actions dataset...')
            for action in actions:
                print('Unzipping ' + action + '...')
                zip_ref = ZipFile(os.path.join(data_path, 'kth_actions', action + '.zip'), 'r')
                os.makedirs(os.path.join(data_path, 'kth_actions', action))
                zip_ref.extractall(os.path.join(data_path, 'kth_actions', action))
                zip_ref.close()
                os.remove(os.path.join(data_path, 'kth_actions', action + '.zip'))
            print('Done.')

            print('Processing KTH Actions dataset...')
            from misc_data_util.convert_kth_actions import convert
            convert(os.path.join(data_path, 'kth_actions'))
            import shutil
            for action in actions:
                shutil.rmtree(os.path.join(data_path, 'kth_actions', action))
            print('Done.')

        from datasets import KTHActions
        train_transforms = []
        if data_config['img_hz_flip']:
            train_transforms.append(trans.RandomHorizontalFlip())
        transforms = [trans.Resize(data_config['img_size']),
                      trans.RandomSequenceCrop(data_config['sequence_length']),
                      trans.ImageToTensor(),
                      trans.ConcatSequence()]
        train_trans = trans.Compose(train_transforms + transforms)
        val_test_trans = trans.Compose(transforms)
        train = KTHActions(os.path.join(data_path, 'kth_actions', 'train'), train_trans)
        val   = KTHActions(os.path.join(data_path, 'kth_actions', 'val'), val_test_trans)
        test  = KTHActions(os.path.join(data_path, 'kth_actions', 'test'), val_test_trans)

    elif dataset_name == 'bair_robot_pushing':
        if not os.path.exists(os.path.join(data_path, 'bair_robot_pushing')):
            os.makedirs(os.path.join(data_path, 'bair_robot_pushing'))

        if not os.path.exists(os.path.join(data_path, 'bair_robot_pushing', 'train')):
            print('Downloading BAIR Robot Pushing dataset...')
            save('http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar',
                os.path.join(data_path, 'bair_robot_pushing', 'bair_robot_pushing_dataset_v0.tar'))
            print('Done.')

            print('Untarring BAIR Robot Pushing dataset...')
            tar = tarfile.open(os.path.join(data_path, 'bair_robot_pushing', 'bair_robot_pushing_dataset_v0.tar'))
            tar.extractall(os.path.join(data_path, 'bair_robot_pushing'))
            tar.close()
            os.remove(os.path.join(data_path, 'bair_robot_pushing', 'bair_robot_pushing_dataset_v0.tar'))
            print('Done.')

            print('Converting TF records...')
            from misc_data_util.convert_bair import convert
            convert(os.path.join(data_path, 'bair_robot_pushing'))
            import shutil
            shutil.rmtree(os.path.join(data_path, 'bair_robot_pushing', 'softmotion30_44k'))
            print('Done.')

        # TODO: make a val set for BAIR, needs to be done in the convert function

        from datasets import BAIRRobotPushing
        train_transforms = []
        if data_config['img_hz_flip']:
            train_transforms.append(trans.RandomHorizontalFlip())
        transforms = [trans.Resize(data_config['img_size']),
                      trans.RandomSequenceCrop(data_config['sequence_length']),
                      trans.ImageToTensor(),
                      trans.ConcatSequence()]
        train_trans = trans.Compose(train_transforms + transforms)
        test_trans = trans.Compose(transforms)
        train = BAIRRobotPushing(os.path.join(data_path, 'bair_robot_pushing', 'train'), train_trans)
        test  = BAIRRobotPushing(os.path.join(data_path, 'bair_robot_pushing', 'test'), test_trans)

    elif dataset_name == 'youtube':
        pass

    else:
        raise Exception('Dataset name not found.')

    return train, val, test
