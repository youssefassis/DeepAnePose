import torch, random, sys
import numpy as np
from utils import get_logger

from Data.Generators import generateTransforms, getPatches
from Data.Augmentation import randomFlip
from Volume.Patch import getPatchAndAneurysms
from batchsampler import BalancedBatchSampler
import Data.IO as dio

logger = get_logger("Dataset")

def getDataloaders(train_db, valid_db, config, workers):
    '''
    This function returns PyTorch data loaders for training and validation sets, 
    given the training and validation databases, a configuration dictionary, 
    and the number of worker processes to use.
    The function first checks whether a training/validation database is provided. 
    If so, it calls the getPatches() function to obtain patches from the database 
    with the specified batch size and positive patch duplicates. Then, it creates
    a Dataset object with the obtained patches and the other configuration parameters,
    and creates a PyTorch data loader using either Balanced Batch Sampler or a standard
    random batch sampler. 
    '''
    train_generator, train_iterations = None, 0
    valid_generator, valid_iterations = None, 0

    balancedbatch = config['balancedbatch'] if 'balancedbatch' in config else False
    scales = config["scales"] if 'scales' in config else None
    
    if train_db is not None:
        train_patches, train_iterations = getPatches(pat_db = train_db, dup_ane = config['positive duplicates'], batch_size = config["batch_size"])

        train_dataset = Dataset(patches = train_patches,
                                size = [x * y for x, y in zip(config["patch_size"], config["patch_shape"])],
                                dim = config["patch_shape"],
                                neg_trans = config["negative sample shift"],
                                neg_rot = config["negative sample rotation"],
                                neg_disp = config["negative sample distortion"],
                                neg_scal = config["negative sample scaling"],
                                pos_trans = config["positive sample shift"],
                                pos_rot = config["positive sample rotation"],
                                pos_disp = config["positive sample distortion"],
                                pos_scal = config["positive sample scaling"],
                                flip_prob = config['flip probability'],
                                flip_axes = config['flip axes'],
                                scales = scales)
        
        if balancedbatch:
            sampler = BalancedBatchSampler(patches = train_dataset.patches, 
                                           batch_size = config["batch_size"], 
                                           nb_negative_samples_per_patient = config['nb neg patches per patient'],
                                           percentage_neg_patches = config['percentage of negative patches'],
                                           drop_last = config["drop_last"])
            train_iterations = sampler.get_n_batches()
            train_generator = torch.utils.data.DataLoader(dataset = train_dataset,
                                                         num_workers = workers,
                                                         pin_memory = True,
                                                         batch_sampler = sampler)
        else: 
            train_generator = torch.utils.data.DataLoader(dataset = train_dataset,
                                                         batch_size = config["batch_size"],
                                                         shuffle = True,
                                                         num_workers = workers,
                                                         pin_memory = True)

        logger.info(f"Training DataLoader : ({len(train_patches)} patches; Balanced batch : {balancedbatch}; batch size = {config['batch_size']}; shuffle = {True}; workers = {workers})")

    if valid_db is not None:
        valid_patches, valid_iterations = getPatches(pat_db = valid_db, dup_ane = config['positive duplicates'], batch_size = config["validation_batch_size"])
        valid_dataset = Dataset(patches = valid_patches, 
                                size = [x * y for x, y in zip(config["patch_size"], config["patch_shape"])],
                                dim = config["patch_shape"],
                                neg_trans = config["negative sample shift"], 
                                neg_rot = config["negative sample rotation"],
                                neg_disp = config["negative sample distortion"], 
                                neg_scal = config["negative sample scaling"],
                                pos_trans = config["positive sample shift"],
                                pos_rot = config["positive sample rotation"], 
                                pos_disp = config["positive sample distortion"],
                                pos_scal = config["positive sample scaling"],
                                flip_prob = config['flip probability'],
                                flip_axes = config['flip axes'],
                                scales = scales,
                                )
        valid_generator = torch.utils.data.DataLoader(dataset = valid_dataset,
                                             batch_size = config["validation_batch_size"],
                                             shuffle = False,
                                             num_workers = workers,
                                             pin_memory = True)

        logger.info(f"Validation DataLoader : ({len(valid_patches)} patches; Balanced batch : {balancedbatch}; batch size = {config['validation_batch_size']}; shuffle = {shuffle_val}; workers = {workers})")
    return {"train": train_generator, "valid": valid_generator}, {'train' : train_iterations, 'valid' : valid_iterations}

class Dataset(torch.utils.data.Dataset):
    '''
    This is a PyTorch Dataset class that defines the data loading pipeline for the neural network. 
    The dataset is assumed to consist of 3D images and their corresponding aneurysms (as markup points).
    The class takes several arguments such as the patches volume (i.e., the images and their aneurysms),
    patch size, dimension, transformations (translation, rotation, distorsion, scaling, flipping) for 
    positive and negative patches, and the scales size as a list (e.g. [12]).
    '''
    def __init__(self, patches, size, dim, neg_trans, neg_rot, neg_disp, neg_scal, pos_trans, pos_rot, pos_disp, pos_scal, flip_prob, flip_axes, scales):
        random.shuffle(patches)
        self.patches = patches
        self.size = size
        self.dim = dim
        self.neg_trans, self.neg_rot, self.neg_disp, self.neg_scal = neg_trans, neg_rot, neg_disp, neg_scal        
        self.pos_trans, self.pos_rot, self.pos_disp, self.pos_scal = pos_trans, pos_rot, pos_disp, pos_scal
        self.flip_prob, self.flip_axis = flip_prob, flip_axes

        self.scales = scales

        logger.info(f"Patch size = {[ round(elem, 2) for elem in self.size]}")
        logger.info(f"Patch dimension = {self.dim}")
        
        logger.info (f"Scales: {self.scales}")
        
        logger.info(f"Positive patches: Scaling: {self.pos_scal}; Translation: {self.pos_trans}; Rotation: {self.pos_rot}; Distorision: {self.pos_disp}; Flip axis: {self.flip_axis}; Flip probability {self.flip_prob}")
        logger.info(f"Negative patches: Scaling: {self.neg_scal}; Translation: {self.neg_trans}; Rotation: {self.neg_rot}; Distorision: {self.neg_disp}; Flip axis: {self.flip_axis}; Flip probability {self.flip_prob}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        '''
        This method first retrieves the patch at the given index. It then determines which set 
        of transformation parameters to use based on the value of p['status']. If p['status'] 
        is True, it uses the positive transformation parameters; otherwise, it uses the negative
        transformation parameters.
        An infinite loop is used to make sure that the augmented aneurysm points are fullt included 
        inside the patch volume and maintains a 1.2mm distance from the borders (see getPatchAndAneurysms).
        If the points are not fully covered inside the patch, new random transformations are 
        generated using the generateTransforms.
        Once it has generated a fully-covered aneurysm, the method applies a random flip to the
        volume and associated aneurysms using the randomFlip function.
        Finally, a list of targets for each scale is generated using the get_truth_from_aneurysms
        function, passing in the associated aneurysms, scales, and the patch dimension (self.dim[0]).
        It then converts each set of targets to a PyTorch tensor.
        '''
        p = self.patches[index]

        affine, disp = None, None
        if p['status']:
            trans, rot, scal, dispc = self.pos_trans, self.pos_rot, self.pos_scal, self.pos_disp
        else:
            trans, scal, rot, dispc = self.neg_trans, self.neg_scal, self.neg_rot, self.neg_disp    
        repeat_patch_gen = True
        while repeat_patch_gen:
            affine, disp = generateTransforms(trans = trans, scal = scal, rot = rot, center = p['point'], disp = dispc)
            patch_data = getPatchAndAneurysms(vol = p['data'], vox2met = p['vox2met'], center = p['point'],  size = self.size, 
                                             dim = self.dim, aneurysms = p['aneurysms'], affine=affine, disp=disp)
            v, aneurysms, fully_covered = patch_data['volume'], patch_data['aneurysms'], patch_data['fully_covered']
            repeat_patch_gen = not fully_covered
        v, aneurysms = randomFlip(data = [v, aneurysms], axes = self.flip_axis, patch_shape= self.dim, flip_probability = self.flip_prob)
        
        targets = get_truth_from_aneurysms(points = aneurysms, scales = self.scales, dim = self.dim[0])
        targets = [[torch.from_numpy(scale[0]), torch.from_numpy(scale[1]), torch.from_numpy(scale[2])] for scale in targets]

        return  {'volume': torch.from_numpy(v[np.newaxis]).float(), 'truth': targets}

def get_truth_from_aneurysms(points, scales, dim):
    '''
    This is a function that takes in a set of points (representing the two ends of an aneurysm), 
    a list of scales, and the dimensions of the input volume. It then creates a set of empty 
    target tensors for each scale in the list of scales, with dimensions equal to the input 
    volume dimensions scaled by the corresponding scale. For each pair of points, the function
    calculates the center point between them, scales it to the appropriate size for each scale,
    and uses the resulting coordinates to set the appropriate values in the corresponding target
    tensors. Specifically, it sets the confidence value to 1 (indicating the presence of an 
    aneurysm in that location), the center point to the scaled coordinates of the center point, 
    and size and orientation of aneurysms to the scaled difference between the center point and 
    one of the end points. The function returns the list of target tensors for each scale.
    '''
    points = points / dim
    targets = [[np.zeros((S, S, S, 1), dtype='float32'), np.zeros((S, S, S, 3), dtype='float32'), np.zeros((S, S, S, 3), dtype='float32') ]  for S in scales]
    for idx in range(0, points.shape[0], 2):
        p1, p2 = points[idx], points[idx + 1]
        center = (p1 + p2) / 2
        x, y, z = center
        for idx, S in enumerate(scales):
            i, j, k = int(S * x), int(S * y), int(S * z)
            x_cell, y_cell, z_cell = S * x - i, S * y - j, S * z - k

            targets[idx][0][i, j, k, 0] = 1 # confidence
            targets[idx][1][i, j, k, :] = np.array([x_cell, y_cell, z_cell]) # center
            targets[idx][2][i, j, k, :] = np.array([(center[0] - p1[0]) * S, (center[1] - p1[1]) * S, (center[2] - p1[2]) * S]) # size and orientation
                
    return targets
