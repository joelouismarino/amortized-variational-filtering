import os

split_person_ids ={'train': [11, 12, 13, 14, 15, 16, 17, 18],
                   'val': [19, 20, 21, 23, 24, 25, 1, 04],
                    'test': [22, 2, 3, 5, 6, 7, 8, 9, 10]}

actions = ['walking', 'jogging', 'running', 'boxing', 'handwaving', 'handclapping']


def convert(data_path):

    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(data_path, split))

        person_ids = split_person_ids[split]

        for person_id in person_ids:
            for action in actions:
                for setting in ['d1', 'd2', 'd3', 'd4']:
                    # load the video
                    pass
