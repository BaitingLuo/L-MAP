import os
import urllib.request
import warnings

import gym
from gym.utils import colorize
import h5py
from tqdm import tqdm
import pickle
import numpy as np

def set_dataset_path(path):
    global DATASET_PATH
    DATASET_PATH = path
    os.makedirs(path, exist_ok=True)


set_dataset_path(os.environ.get('D4RL_DATASET_DIR', os.path.expanduser('~/.d4rl/datasets')))


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def filepath_from_url(dataset_url):
    _, dataset_name = os.path.split(dataset_url)
    dataset_filepath = os.path.join(DATASET_PATH, dataset_name)
    return dataset_filepath


def download_dataset_from_url(dataset_url):
    dataset_filepath = filepath_from_url(dataset_url)
    if not os.path.exists(dataset_filepath):
        print('Downloading dataset:', dataset_url, 'to', dataset_filepath)
        urllib.request.urlretrieve(dataset_url, dataset_filepath)
    if not os.path.exists(dataset_filepath):
        raise IOError("Failed to download dataset from %s" % dataset_url)
    return dataset_filepath


class OfflineEnv(gym.Env):
    """
    Base class for offline RL envs.

    Args:
        dataset_url: URL pointing to the dataset.
        ref_max_score: Maximum score (for score normalization)
        ref_min_score: Minimum score (for score normalization)
        deprecated: If True, will display a warning that the environment is deprecated.
    """

    def __init__(self, dataset_url=None, ref_max_score=None, ref_min_score=None, 
                       deprecated=False, deprecation_message=None, **kwargs):
        super(OfflineEnv, self).__init__(**kwargs)
        self.dataset_url = self._dataset_url = dataset_url
        self.ref_max_score = ref_max_score
        self.ref_min_score = ref_min_score
        if deprecated:
            if deprecation_message is None:
                deprecation_message = "This environment is deprecated. Please use the most recent version of this environment."
            # stacklevel=2 will bump the warning to the superclass.
            warnings.warn(colorize(deprecation_message, 'yellow'), stacklevel=2)
 

    def get_normalized_score(self, score):
        if (self.ref_max_score is None) or (self.ref_min_score is None):
            raise ValueError("Reference score not provided for env")
        return (score - self.ref_min_score) / (self.ref_max_score - self.ref_min_score)

    @property
    def dataset_filepath(self):
        return filepath_from_url(self.dataset_url)

    def load_data_from_file(self, filepath):
        # Check if the file exists
        if not os.path.exists(filepath):
            raise ValueError("File does not exist: {}".format(filepath))

        # Determine the file extension
        _, file_extension = os.path.splitext(filepath)

        data_dict = {}
        if file_extension in ['.pkl', '.pickle']:
            # Load Pickle file
            with open(filepath, 'rb') as pickle_file:
                data_dict = pickle.load(pickle_file)
        elif file_extension in ['.h5', '.hdf5']:
            # Load HDF5 file
            with h5py.File(filepath, 'r') as dataset_file:
                for k in tqdm(get_keys(dataset_file), desc="load datafile"):
                    try:  # first try loading as an array
                        data_dict[k] = dataset_file[k][:]
                    except ValueError as e:  # try loading as a scalar
                        data_dict[k] = dataset_file[k][()]
        else:
            raise ValueError("Unsupported file type: {}".format(file_extension))

        return data_dict

    def merge_datasets(self, filepaths):
        # Load each dataset into its own dictionary
        data_dicts = [self.load_data_from_file(fp) for fp in filepaths]

        # Assume that all dictionaries share the same set of keys
        merged_data = {}
        common_keys = data_dicts[0].keys()
        for key in common_keys:
            arrays = []
            for d in data_dicts:
                if key not in d:
                    raise ValueError(f"Key '{key}' not found in all datasets")
                arr = d[key]
                # If rewards or terminals come in as (N, 1), flatten them
                if arr.ndim == 2 and arr.shape[1] == 1:
                    arr = arr[:, 0]
                arrays.append(arr)
            merged_data[key] = np.concatenate(arrays, axis=0)

        return merged_data


    def get_dataset(self, h5path=None):
        if h5path is None:
            if self._dataset_url is None:
                raise ValueError("Offline env not configured with a dataset URL.")
            h5path = download_dataset_from_url(self.dataset_url)
        if "walker2d_medium_expert" in h5path:
            h5path = "/home/-/.d4rl/datasets/walker2d-high-noise-medium-expert-v0.pkl"
        elif "walker_mixed" in h5path:
            h5path = "/home/-/.d4rl/datasets/walker2d-high-noise-medium-replay-v0.pkl"
        elif "walker2d_medium" in h5path:
            h5path = "/home/-/.d4rl/datasets/walker2d-high-noise-medium-v0.pkl"
        elif "hopper_medium_expert" in h5path:
            h5path = "/home/-/.d4rl/datasets/hopper-high-noise-medium-expert-v0.pkl"
        elif "hopper_mixed" in h5path:
            h5path = "/home/-/.d4rl/datasets/hopper-high-noise-medium-replay-v0.pkl"
        elif "hopper_medium" in h5path:
            h5path = "/home/-/.d4rl/datasets/hopper-high-noise-medium-v0.pkl"
        else:
            data_dict = self.load_data_from_file(h5path)
        print(h5path)
        print("loading!!!!!", h5path)
        # Run a few quick sanity checks
        for key in ['observations', 'actions', 'rewards', 'terminals']:
            assert key in data_dict, 'Dataset is missing key %s' % key
        #print(data_dict['observations'].shape)
        N_samples = data_dict['observations'].shape[0]
        if self.observation_space.shape is not None:
            assert data_dict['observations'].shape[1:] == self.observation_space.shape, \
                'Observation shape does not match env: %s vs %s' % (
                    str(data_dict['observations'].shape[1:]), str(self.observation_space.shape))
        assert data_dict['actions'].shape[1:] == self.action_space.shape, \
            'Action shape does not match env: %s vs %s' % (
                str(data_dict['actions'].shape[1:]), str(self.action_space.shape))
        if data_dict['rewards'].shape == (N_samples, 1):
            data_dict['rewards'] = data_dict['rewards'][:, 0]
        assert data_dict['rewards'].shape == (N_samples,), 'Reward has wrong shape: %s' % (
            str(data_dict['rewards'].shape))
        if data_dict['terminals'].shape == (N_samples, 1):
            data_dict['terminals'] = data_dict['terminals'][:, 0]
        assert data_dict['terminals'].shape == (N_samples,), 'Terminals has wrong shape: %s' % (
            str(data_dict['rewards'].shape))
        return data_dict

    def get_dataset_chunk(self, chunk_id, h5path=None):
        """
        Returns a slice of the full dataset.

        Args:
            chunk_id (int): An integer representing which slice of the dataset to return.

        Returns:
            A dictionary containing observations, actions, rewards, and terminals.
        """
        if h5path is None:
            if self._dataset_url is None:
                raise ValueError("Offline env not configured with a dataset URL.")
            h5path = download_dataset_from_url(self.dataset_url)

        dataset_file = h5py.File(h5path, 'r')

        if 'virtual' not in dataset_file.keys():
            raise ValueError('Dataset is not a chunked dataset')
        available_chunks = [int(_chunk) for _chunk in list(dataset_file['virtual'].keys())]
        if chunk_id not in available_chunks:
            raise ValueError('Chunk id not found: %d. Available chunks: %s' % (chunk_id, str(available_chunks)))

        load_keys = ['observations', 'actions', 'rewards', 'terminals']
        data_dict = {k: dataset_file['virtual/%d/%s' % (chunk_id, k)][:] for k in load_keys}
        dataset_file.close()
        return data_dict


class OfflineEnvWrapper(gym.Wrapper, OfflineEnv):
    """
    Wrapper class for offline RL envs.
    """

    def __init__(self, env, **kwargs):
        gym.Wrapper.__init__(self, env)
        OfflineEnv.__init__(self, **kwargs)

    def reset(self):
        return self.env.reset()
