import os
import tarfile
import zipfile
import shutil
from pathlib import Path
from urllib.parse import urlsplit
import requests
from tqdm.auto import tqdm

DATASETS = {
    'fashion_mnist': {
        'train_data': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'train_labels': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'test_data': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'test_labels': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
        'label_names': 
            {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal',
            6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}
    },
    'mnist': {
        'train_data': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'train_labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'test_data': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'test_labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    }
}

def list_datasets():
    return list(DATASETS.keys())

class Dataset(object):
    def __init__(self, name, base_path=Path('~/data'), load=False):
        if name in list_datasets():
            self.name = name
        else:
            raise ValueError(f'Unknown dataset name {name}. Must be in {str(list_datasets())}')
        self.base_path = base_path.expanduser()
        if not self.base_path.is_dir(): self.base_path.mkdir()
        if load:
            self.load()
        self.filenames = self._check_loaded()
    
    def load(self, train=True, test=True, force=False, pbar=False):
        filenames = []
        ds_path = self.base_path.joinpath(self.name)
        if not ds_path.is_dir(): ds_path.mkdir()
        urls = []
        if train:
            urls.append(DATASETS[self.name]['train_data'])
            urls.append(DATASETS[self.name]['train_labels'])
        if test:
            urls.append(DATASETS[self.name]['test_data'])
            urls.append(DATASETS[self.name]['test_labels'])
        for url in urls:
            fn = os.path.basename(urlsplit(url).path)
            ds_path_fn = ds_path.joinpath(fn)
            if force or not ds_path_fn.exists():
                _download(url, ds_path_fn, pbar=pbar)
                filenames.append(ds_path_fn)
        self.filenames = self._check_loaded()
        return self.filenames
    
    def label_names(self):
        return DATASETS[self.name].get('label_names')
    
    def _check_loaded(self):
        filenames = {'train_data': None, 'train_labels': None, 'test_data': None, 'test_labels': None}
        if not self.base_path.is_dir(): 
            self.filenames = filenames
            return filenames
        ds_path = self.base_path.joinpath(self.name)
        if not ds_path.is_dir(): 
            self.filenames = filenames
            return filenames

        for key in filenames.keys():
            url = DATASETS[self.name][key]
            fn = os.path.basename(urlsplit(url).path)
            ds_path_fn = ds_path.joinpath(fn)
            if ds_path_fn.exists():
                filenames[key] = ds_path_fn
        self.filenames = filenames
        return filenames

def _download(source, target, pbar=False, chunk_size=1024):
    r = requests.get(source, stream=True)
    with open(target, 'wb') as fd:
        if pbar:
            bar = tqdm(unit='B', total=int(r.headers['Content-Length']))
            for chunk in tqdm(r.iter_content(chunk_size=chunk_size)):
                if chunk:
                    bar.update(len(chunk))
                    fd.write(chunk)
        else:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)

def _extract_archive(file_path, path='.', archive_format='auto'):
  if archive_format is None:
    return False
  if archive_format == 'auto':
    archive_format = ['tar', 'zip']
  if isinstance(archive_format, str):
    archive_format = [archive_format]

  file_path = os.fspath(file_path)
  path = os.fspath(path)

  for archive_type in archive_format:
    if archive_type == 'tar':
      open_fn = tarfile.open
      is_match_fn = tarfile.is_tarfile
    if archive_type == 'zip':
      open_fn = zipfile.ZipFile
      is_match_fn = zipfile.is_zipfile

    if is_match_fn(file_path):
      with open_fn(file_path) as archive:
        try:
          archive.extractall(path)
        except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
          if os.path.exists(path):
            if os.path.isfile(path):
              os.remove(path)
            else:
              shutil.rmtree(path)
          raise
      return True
  return False