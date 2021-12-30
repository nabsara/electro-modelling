from torchvision import transforms
from torch.utils.data import DataLoader
from electro_modelling.helpers.utils import load_pickle
from electro_modelling.datasets.techno_dataset import TechnoDatasetWav, TechnoDatasetWavtoMel


def techno_data_loader(batch_size, data_dir,operator,phase_method='griff'):
    if phase_method == 'griff':
        transform = transforms.Compose([
            transforms.Normalize((-3.76), (10.05))  #TO BE MODIFIED
        ])    
    elif phase_method =='IF':
        transform = transforms.Compose([
            transforms.Normalize((-3.76, 0), (10.05,1))  #TO BE MODIFIED
        ])
    train_set = TechnoDatasetWavtoMel(operator,phase_method = 'griff', dat_location=data_dir)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_loader
