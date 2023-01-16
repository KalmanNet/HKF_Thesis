from Dataloaders.BaseDataLoader import BaseECGLoader
import torch
import os
import glob
from numpy import genfromtxt, array


class CardiaceSense_Loader(BaseECGLoader):

    def __init__(self, datapoints: int, samples: list, snr_dB: int):
        if samples[0] == 0:
            snr_dB = None
        super(CardiaceSense_Loader, self).__init__(datapoints, samples, snr_dB)

    def load_data(self, samples: list) -> (torch.Tensor, int):
        # Define sampling frequency

        # Get location of this file
        file_location = os.path.dirname(os.path.realpath(__file__))

        # Get path to dataset location
        path_to_dataset = file_location + '\\..\\Datasets\\CardiacSense'

        raw_data_files = glob.glob(path_to_dataset + '\\' + '*.csv')
        raw_data_files = [raw_data_files[index] for index in samples]

        samples_per_second = [int(file[-7:-4]) for file in raw_data_files]

        dataset = [genfromtxt(raw_data_file[:-4] + '.csv', delimiter='\n', skip_header=1) for raw_data_file in
                   raw_data_files]
        sps = samples_per_second[0]

        if samples[0] == 0:
            sign = -1
        else:
            sign = 1

        dataset = sign * torch.tensor(array(dataset)).reshape(len(samples), -1, 1).float()

        return dataset, sps


    def __len__(self):
        return 220
