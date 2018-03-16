import os
import cPickle
import tarfile
import misc_data_util.transforms as trans
from zipfile import ZipFile
from misc_data_util.url_save import save


def load_dataset(dataset_name, data_path):
    """
    Downloads and loads a variety of standard benchmark sequence datasets.
    Arguments:
        dataset_name: the name of the dataset to loading
        data_path: path to directory where data should be saved
    Returns:
        tuple of (train, val, test), each of which is a PyTorch dataset.
    """

    assert os.path.exists(data_path), 'Data path not found. Please specify a valid path.'

    train = val = test = None
    dataset_name = dataset_name.lower() # standardize the dataset_name to lower case

    ############################################################################
    ## Speech datasets
    ############################################################################
    if dataset_name == 'blizzard':
        if not os.path.exists(os.path.join(data_path, 'blizzard')):
            raise ValueError('BLIZZARD dataset does not exist. Please manually \
                             download the dataset by obtaining a license from \
                             http://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/, \
                             then create a directory called blizzard in your data directory.')
        # TODO: load BLIZZARD

    if dataset_name == 'timit':
        if not os.path.exists(os.path.join(data_path, 'timit')):
            raise ValueError('TIMIT dataset does not exist. Please manually \
                             download the dataset by obtaining a license from \
                             https://github.com/philipperemy/timit, \
                             then create a directory called timit in your data directory.')
        # TODO: load TIMIT

    ############################################################################
    ## Handwriting datasets
    ############################################################################
    if dataset_name == 'iam_ondb':
        if not os.path.exists(os.path.join(data_path, 'iam_ondb')):
            raise ValueError('IAM_OnDB dataset does not exist. Please manually \
                             download the dataset by obtaining a license from \
                             http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database, \
                             then create a directory called iam_ondb in your data directory.')
        # TODO: load IAM_OnDB

    ############################################################################
    ## MIDI datasets
    ############################################################################
    if dataset_name == 'piano_midi':
        if not os.path.exists(os.path.join(data_path, 'piano_midi')):
            os.makedirs(os.path.join(data_path, 'piano_midi'))
        if not os.path.exists(os.path.join(data_path, 'piano_midi', 'Piano-midi.de.pickle')):
            print('Downloading piano-midi dataset...')
            save('http://www-etud.iro.umontreal.ca/~boulanni/Piano-midi.de.pickle',
                 os.path.join(data_path, 'piano_midi', 'Piano-midi.de.pickle'))
            print('Done.')
        piano_roll = cPickle.load(os.path.join(data_path, 'piano_midi', 'Piano-midi.de.pickle'))
        # TODO: load piano_midi

    if dataset_name == 'nottingham':
        if not os.path.exists(os.path.join(data_path, 'nottingham')):
            os.makedirs(os.path.join(data_path, 'nottingham'))
        if not os.path.exists(os.path.join(data_path, 'nottingham', 'Nottingham.pickle')):
            print('Downloading nottingham dataset...')
            save('http://www-etud.iro.umontreal.ca/~boulanni/Nottingham.pickle',
                 os.path.join(data_path, 'nottingham', 'Nottingham.pickle'))
            print('Done.')
        piano_roll = cPickle.load(os.path.join(data_path, 'nottingham', 'Nottingham.pickle'))
        # TODO: load nottingham

    if dataset_name == 'muse':
        if not os.path.exists(os.path.join(data_path, 'muse')):
            os.makedirs(os.path.join(data_path, 'muse'))
        if not os.path.exists(os.path.join(data_path, 'muse', 'MuseData.pickle')):
            print('Downloading muse dataset...')
            save('http://www-etud.iro.umontreal.ca/~boulanni/MuseData.pickle',
                 os.path.join(data_path, 'muse', 'MuseData.pickle'))
            print('Done.')
        piano_roll = cPickle.load(os.path.join(data_path, 'muse', 'MuseData.pickle'))
        # TODO: load muse

    if dataset_name == 'jsb_chorales':
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
    ## Video datasets
    ############################################################################
    if dataset_name == 'stochastic_moving_mnist':
        if not os.path.exists(os.path.join(data_path, 'stochastic_moving_MNIST')):
            os.makedirs(os.path.join(data_path, 'stochastic_moving_MNIST'))
            # download stochastic_moving_MNIST dataset
        # TODO: load stochastic_moving_MNIST

    if dataset_name == 'kth_actions':
        if not os.path.exists(os.path.join(data_path, 'KTH_actions')):
            os.makedirs(os.path.join(data_path, 'KTH_actions'))
        if not os.path.exists(os.path.join(data_path, 'KTH_actions', 'train')):
            print('Downloading KTH Actions dataset...')
            actions = ['walking', 'jogging', 'running', 'boxing', 'handwaving', 'handclapping']
            for action in actions:
                print('Downloading ' + action + '...')
                save('http://www.nada.kth.se/cvap/actions/' + action + '.zip',
                    os.path.join(data_path, 'KTH_actions', action + '.zip'))
                print('\n')
            print('Done.')

            print('Unzipping KTH Actions dataset...')
            for action in actions:
                print('Unzipping ' + action + '...')
                zip_ref = ZipFile(os.path.join(data_path, 'KTH_actions', action + '.zip'), 'r')
                os.makedirs(os.path.join(data_path, 'KTH_actions', action))
                zip_ref.extractall(os.path.join(data_path, 'KTH_actions', action))
                zip_ref.close()
                os.remove(os.path.join(data_path, 'KTH_actions', action + '.zip'))
            print('Done.')

            print('Processing KTH Actions dataset...')
            from misc_data_util.convert_kth_actions import convert
            convert(os.path.join(data_path, 'KTH_actions'))
            import shutil
            for action in actions:
                shutil.rmtree(os.path.join(data_path, 'KTH_actions', action))
            print('Done.')

        from datasets.kth_actions import KTHActions
        train_trans = trans.Compose([trans.RandomHorizontalFlip(),
                                     trans.Resize(64),
                                     trans.RandomSequenceCrop(20),
                                     trans.ToTensor(),
                                     trans.ConcatSequence()])
        val_test_trans = trans.Compose([trans.Resize(64),
                                    trans.RandomSequenceCrop(20),
                                    trans.ToTensor(),
                                    trans.ConcatSequence()])
        train = KTHActions(os.path.join(data_path, 'KTH_actions', 'train'), train_trans)
        val   = KTHActions(os.path.join(data_path, 'KTH_actions', 'val'), val_test_trans)
        test  = KTHActions(os.path.join(data_path, 'KTH_actions', 'test'), val_test_trans)

    if dataset_name == 'bair_robot_pushing':
        if not os.path.exists(os.path.join(data_path, 'BAIR_robot_pushing')):
            os.makedirs(os.path.join(data_path, 'BAIR_robot_pushing'))

        if not os.path.exists(os.path.join(data_path, 'BAIR_robot_pushing', 'train')):
            print('Downloading BAIR Robot Pushing dataset...')
            save('http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar',
                os.path.join(data_path, 'BAIR_robot_pushing', 'bair_robot_pushing_dataset_v0.tar'))
            print('Done.')

            print('Untarring BAIR Robot Pushing dataset...')
            tar = tarfile.open(os.path.join(data_path, 'BAIR_robot_pushing', 'bair_robot_pushing_dataset_v0.tar'))
            tar.extractall(os.path.join(data_path, 'BAIR_robot_pushing'))
            tar.close()
            os.remove(os.path.join(data_path, 'BAIR_robot_pushing', 'bair_robot_pushing_dataset_v0.tar'))
            print('Done.')

            print('Converting TF records...')
            from misc_data_util.convert_bair import convert
            convert(os.path.join(data_path, 'BAIR_robot_pushing'))
            import shutil
            shutil.rmtree(os.path.join(data_path, 'BAIR_robot_pushing', 'softmotion30_44k'))
            print('Done.')

        from datasets.bair_robot_pushing import BAIRRobotPushing
        train_trans = trans.Compose([trans.RandomHorizontalFlip(),
                                     trans.Resize(64),
                                     trans.RandomSequenceCrop(20),
                                     trans.ToTensor(),
                                     trans.ConcatSequence()])
        test_trans = trans.Compose([trans.Resize(64),
                                    trans.RandomSequenceCrop(20),
                                    trans.ToTensor(),
                                    trans.ConcatSequence()])
        train = BAIRRobotPushing(os.path.join(data_path, 'BAIR_robot_pushing', 'train'), train_trans)
        test  = BAIRRobotPushing(os.path.join(data_path, 'BAIR_robot_pushing', 'test'), test_trans)

    return train, val, test
