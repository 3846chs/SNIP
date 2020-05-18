import os
import itertools
import numpy as np

import mnist
import cifar

import urllib

try:
    from urllib.error import URLError
    from urllib.request import urlretrieve
except ImportError:
    from urllib2 import URLError
    from urllib import urlretrieve

MNIST_RESOURCES = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz',
]

class Dataset(object):
    def __init__(self, datasource, path_data, **kwargs):
        self.datasource = datasource
        self.path_data = path_data
        self.rand = np.random.RandomState(9)
        
        if not os.path.exists(self.path_data):
            original_umask = os.umask(0)
            desired_permission = 0o0777
            os.makedirs(self.path_data, desired_permission)
            os.umask(original_umask)

        if self.datasource == 'mnist':
            # TODO: need port forwarding docker container's tcp port to host's tcp port
            # for resource in MNIST_RESOURCES:
            #     path = os.path.join(self.path_data, resource)
            #     print(path)
            #     url = 'http://yann.lecun.com/exdb/mnist/{}'.format(resource)
            #     download(path, url)

            self.num_classes = 10
            self.dataset = mnist.read_data(os.path.join(self.path_data, 'MNIST'))
        elif self.datasource == 'cifar-10':
            self.num_classes = 10
            self.dataset = cifar.read_data(os.path.join(self.path_data, 'cifar-10-batches-py'))
        else:
            raise NotImplementedError
        self.split_dataset('train', 'val', int(self.dataset['train']['input'].shape[0] * 0.1),
            self.rand)
        self.num_example = {k: self.dataset[k]['input'].shape[0] for k in self.dataset.keys()}
        self.example_generator = {
            'train': self.iterate_example('train'),
            'val': self.iterate_example('val'),
            'test': self.iterate_example('test', shuffle=False),
        }

    def iterate_example(self, mode, shuffle=True):
        epochs = itertools.count()
        for i in epochs:
            example_ids = list(range(self.num_example[mode]))
            if shuffle:
                self.rand.shuffle(example_ids)
            for example_id in example_ids:
                yield {
                    'input': self.dataset[mode]['input'][example_id],
                    'label': self.dataset[mode]['label'][example_id],
                    'id': example_id,
                }

    def get_next_batch(self, mode, batch_size):
        inputs, labels, ids = [], [], []
        for i in range(batch_size):
            example = next(self.example_generator[mode])
            inputs.append(example['input'])
            labels.append(example['label'])
            ids.append(example['id'])
        return {
            'input': np.asarray(inputs),
            'label': np.asarray(labels),
            'id': np.asarray(ids),
        }

    def generate_example_epoch(self, mode):
        example_ids = range(self.num_example[mode])
        for example_id in example_ids:
            yield {
                'input': self.dataset[mode]['input'][example_id],
                'label': self.dataset[mode]['label'][example_id],
                'id': example_id,
            }

    def split_dataset(self, source, target, number, rand):
        keys = ['input', 'label']
        indices = list(range(self.dataset[source]['input'].shape[0]))
        rand.shuffle(indices)
        ind_target = indices[:number]
        ind_remain = indices[number:]
        self.dataset[target] = {k: self.dataset[source][k][ind_target] for k in keys}
        self.dataset[source] = {k: self.dataset[source][k][ind_remain] for k in keys}

# copied from https://gist.github.com/goldsborough/6dd52a5e01ed73a642c1e772084bcd03 
# and revised by cmpark
def download(destination_path, url):
    if os.path.exists(destination_path):
        print('{} already exists, skipping ...'.format(destination_path))
    else:
        print('Downloading {} ...'.format(url))
        try:
            urlretrieve(url, destination_path)
        except URLError:
            raise RuntimeError('Error downloading resource!')
        finally:
            # Just a newline.
            print()

def report_download_progress(chunk_number, chunk_size, file_size):
    if file_size != -1:
        percent = min(1, (chunk_number * chunk_size) / file_size)
        bar = '#' * int(64 * percent)
        sys.stdout.write('\r0% |{:<64}| {}%'.format(bar, int(percent * 100)))