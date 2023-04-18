#!/usr/bin/env python3
import os, sys, json, torch

from utils import get_optimizer_scheduler, get_model
from Data.IO import add_points_to_patient_data, readSplit, read_patient_data_base
from losses import get_loss_criterion

from trainer import create_trainer
from dataset import getDataloaders

def main():
    # load config file
    try:
        with open(os.path.join(sys.argv[1], 'ndl_config.json'),'r') as f:
            config = json.load(f)
    except:
        print(f'No such config file for training')
        exit()    
    
    workers = 12
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = get_model(config, device=device)

    # Data Preparation
    ## Load patients
    train_list, valid_list, _ = readSplit(config['split_file'])
    ## Load training and validation patches
    train_db = add_points_to_patient_data(read_patient_data_base(pat_list = train_list,
                                                                 volume = config['volume'],
                                                                 pts_aneurysm = config['truth file'],
                                                                 normalize = config['normalize'],
                                                                 label = "Training",
                                                                 workers = workers,
                                                                ), 
                                          config['negative patch centers'],
                                          nb = None)
    valid_db = add_points_to_patient_data(read_patient_data_base(pat_list = valid_list,
                                                                 volume = config['volume'],
                                                                 pts_aneurysm = config['truth file'],
                                                                 normalize = config['normalize'],
                                                                 label = "Validation",
                                                                 workers = workers,
                                                                ),
                                          config['negative patch centers'],
                                          nb = None) if len(valid_list) > 0 else None
    # Create datasets and data loaders
    loaders, iterations = getDataloaders(train_db = train_db, 
                                         valid_db = valid_db, 
                                         config = config, 
                                         workers = workers)
    # Create optimizer & scheduler
    warmup_steps = iterations['train'] * config["warmEpochs"] if "warmEpochs" in config else 0
    optimizer, scheduler = get_optimizer_scheduler (model = model, config=config, warmup_steps=warmup_steps)
    # Create Trainer and start/resume training
    trainer = create_trainer( config, device, model, optimizer, scheduler, get_loss_criterion(config["loss"]), loaders, iterations)
    trainer.fit()

if __name__ == '__main__':
    main()
