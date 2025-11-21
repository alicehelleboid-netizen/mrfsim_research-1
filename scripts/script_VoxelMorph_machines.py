# imports
# third party imports
import numpy as np
import tensorflow as tf
import json
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'
from copy import copy
from mrfsim.utils_reco import unpad,format_input_voxelmorph,format_input_voxelmorph_3D,plot_deformation_map,apply_deformation_to_complex_volume
from skimage.transform import resize

print(tf.config.experimental.list_physical_devices("GPU"))

# local imports
import voxelmorph as vxm
import neurite as ne
from mrfsim import io

import matplotlib.pyplot as plt
try:
    import SimpleITK as sitk
except:
    pass
import wandb
from wandb.integration.keras import WandbMetricsLogger,WandbModelCheckpoint
try:
    import torchio as tio
    import torch
    import torchvision.transforms as T
except:
    pass

from sklearn.model_selection import train_test_split
from keras import backend

import machines as ma
from machines import Toolbox
import subprocess

def get_total_memory_mb():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,nounits,noheader"]
    )
    return int(result.decode("utf-8").strip().split("\n")[0])

total_mem = get_total_memory_mb()
fraction = 0.25

gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
if gpus:
    try:
        mem_limit = int(total_mem * fraction)
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=mem_limit)]
        )
        print(f"GPU memory limited to {mem_limit} MB")
    except RuntimeError as e:
        print("Error:", e)


DEFAULT_TRAIN_CONFIG="../config/config_train_voxelmorph.json"
DEFAULT_TRAIN_CONFIG_3D="../config/config_train_voxelmorph_3D.json"
# You should most often have this import together with all other imports at the top,
# but we include here here explicitly to show where data comes from

def vxm_data_generator(x_data_fixed, x_data_moving, batch_size=32):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """

    # preliminary sizing
    vol_shape = x_data_fixed.shape[1:]  # extract data shape
    ndims = len(vol_shape)

    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])

    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data_moving.shape[0], size=batch_size)
        moving_images = x_data_moving[idx1, ..., np.newaxis]
        # idx2 = np.random.randint(0, x_data_fixed.shape[0], size=batch_size)
        fixed_images = x_data_fixed[idx1, ..., np.newaxis]
        inputs = [moving_images, fixed_images]

        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare
        # the resulting moved image with the fixed image.
        # we also wish to penalize the deformation field.
        outputs = [fixed_images, zero_phi]

        yield (inputs, outputs)

def vxm_data_generator_3D(x_data_fixed, x_data_moving, batch_size=32):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """

    # preliminary sizing
    vol_shape = x_data_fixed.shape[1:]  # extract data shape
    ndims = len(vol_shape)

    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])

    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data_moving.shape[0], size=batch_size)
        moving_images = x_data_moving[idx1, ...,np.newaxis]
        # idx2 = np.random.randint(0, x_data_fixed.shape[0], size=batch_size)
        fixed_images = x_data_fixed[idx1, ...,np.newaxis]
        inputs = [moving_images, fixed_images]

        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare
        # the resulting moved image with the fixed image.
        # we also wish to penalize the deformation field.
        outputs = [fixed_images, zero_phi]

        yield (inputs, outputs)

def vxm_data_generator_torchio(x_data_fixed, x_data_moving, transform,batch_size=32):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """

    # preliminary sizing
    vol_shape = x_data_fixed.shape[1:]  # extract data shape
    ndims = len(vol_shape)

    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])

    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data_moving.shape[0], size=batch_size)
        moving_images=[]
        fixed_images=[]
        for j in idx1:
            #print(j)
            moving_image = x_data_moving[np.newaxis,j, ..., np.newaxis]
            # idx2 = np.random.randint(0, x_data_fixed.shape[0], size=batch_size)
            fixed_image = x_data_fixed[np.newaxis,j, ..., np.newaxis]
            subject=tio.Subject(moving=tio.ScalarImage(tensor=moving_image),fixed=tio.ScalarImage(tensor=fixed_image))
            subject_transf=transform(subject)
            moving_images.append(subject_transf.moving.data.squeeze()[...,np.newaxis].numpy())
            fixed_images.append(subject_transf.fixed.data.squeeze()[..., np.newaxis].numpy())

        inputs = [np.array(moving_images), np.array(fixed_images)]


        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare
        # the resulting moved image with the fixed image.
        # we also wish to penalize the deformation field.
        outputs = [np.array(fixed_images), zero_phi]

        yield (inputs, outputs)


def vxm_data_generator_torchvision(x_data_fixed, x_data_moving, transform_list,batch_size=32,probabilities=None):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """

    # preliminary sizing
    vol_shape = x_data_fixed.shape[1:]  # extract data shape
    ndims = len(vol_shape)

    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    transform_minmax = T.Lambda(lambda x: x - x.min() / (x.max() - x.min()))
    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data_moving.shape[0], size=batch_size)
        moving_images=[]
        fixed_images=[]



        for j in idx1:
            #print(j)
            moving_image = torch.from_numpy(x_data_moving[np.newaxis,j])
            # idx2 = np.random.randint(0, x_data_fixed.shape[0], size=batch_size)
            fixed_image = torch.from_numpy(x_data_fixed[np.newaxis,j])
            t=np.random.choice(transform_list,p=probabilities)
            #print("Transform for j {}: {}".format(j,t))

            transf= T.Compose([t, T.Lambda(lambda x : (x-x.min())/(x.max()-x.min()))])
            moving_images.append(transf(moving_image).numpy().squeeze()[...,np.newaxis])
            fixed_images.append(transf(fixed_image).numpy().squeeze()[..., np.newaxis])

        inputs = [np.array(moving_images), np.array(fixed_images)]

        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare
        # the resulting moved image with the fixed image.
        # we also wish to penalize the deformation field.
        outputs = [np.array(fixed_images), zero_phi]

        yield (inputs, outputs)

@ma.machine()
@ma.parameter("filename_volumes",str,default=None,description="Volumes for all motion phases nb_motion x nb_slices x npoint_x x npoint_y")
@ma.parameter("file_config_train",str,default=DEFAULT_TRAIN_CONFIG,description="Config training")
@ma.parameter("kept_bins",str,default=None,description="Bins to keep for training")
@ma.parameter("suffix",str,default="",description="suffix")
@ma.parameter("resolution",int,default=None,description="image resolution")
@ma.parameter("nepochs",int,default=None,description="Number of epochs (overwrites config)")
@ma.parameter("lr",float,default=None,description="Learning rate (overwrites config)")
@ma.parameter("init_weights",str,default=None,description="Weights initialization from .h5 file")
@ma.parameter("us",int,default=None,description="Select one every us slice")
@ma.parameter("excluded",int,default=5,description="Excluded slices on both extremities")
@ma.parameter("axis",int,default=None,description="Change registration axis")
def train_voxelmorph(filename_volumes,file_config_train,suffix,init_weights,resolution,nepochs,lr,kept_bins,axis,us,excluded):

    with open(file_config_train,"r") as f:
        config_train=json.load(f)

    all_volumes = np.abs(np.load(filename_volumes))
    print("Volumes shape {}".format(all_volumes.shape))
    all_volumes=all_volumes.astype("float32")


    if kept_bins is not None:
        kept_bins_list=np.array(str.split(kept_bins,",")).astype(int)
        print(kept_bins_list)
        all_volumes=all_volumes[kept_bins_list]


    nb_gr,nb_slices,npoint,npoint=all_volumes.shape


    if axis is not None:
        all_volumes=np.moveaxis(all_volumes,axis+1,1)
        # all_volumes=resize(all_volumes,(nb_gr,npoint,npoint,npoint))
        all_volumes=all_volumes[:,::int(npoint/nb_slices)]


    if us is not None:
        # all_volumes=resize(all_volumes,(nb_gr,npoint,npoint,npoint))
        all_volumes=all_volumes[:,::us]

    print(all_volumes.shape)

    if nepochs is not None:
        config_train["nb_epochs"]=nepochs

    if lr is not None:
        config_train["lr"]=lr

    if resolution is not None:
        all_volumes=resize(all_volumes,(nb_gr,nb_slices,resolution,resolution))

    if resolution is None:
        file_model=filename_volumes.split(".npy")[0]+"_vxm_model_weights{}.h5".format(suffix)
        file_checkpoint="/".join(filename_volumes.split("/")[:-1])+"/model_checkpoint.h5"
    else:
        file_model=filename_volumes.split(".npy")[0]+"_vxm_model_weights_res{}{}.h5".format(resolution,suffix)
        file_checkpoint="/".join(filename_volumes.split("/")[:-1])+"/model_checkpoint_res{}.h5".format(resolution)
    print(file_checkpoint)
    # run=wandb.init(
    #     project=str.replace(file_model.split("/")[-1].split("_FULL")[0],"raFin_3D_","")+file_model.split("/")[-1].split("_FULL")[1],
    #     config=config_train
    # )

    run=wandb.init(
        project="project_test",
        config=config_train
    )

    #pad_amount=config_train["padding"]
    loss = config_train["loss"]
    decay=config_train["lr_decay"]
    #pad_amount=tuple(tuple(l) for l in pad_amount)

    #Finding the power of 2 "closest" and longer than  x dimension
    n = np.maximum(all_volumes.shape[-1],all_volumes.shape[-2])
    pad_1=2**(int(np.log2(n))+1)-n
    pad_2= 2 ** int(np.log2(n) - 1) * 3 - n

    if pad_2<0:
        pad=int(pad_1/2)
    else:
        pad = int(pad_2 / 2)

    # if n%2==0:
    #     pad=0
    pad_x=int((2*pad+n-all_volumes.shape[-2])/2)
    pad_y=int((2*pad+n-all_volumes.shape[-1])/2)

    pad_amount = ((0,0),(pad_x,pad_x), (pad_y,pad_y))
    print(pad_amount)
    nb_features=config_train["nb_features"]
    # configure unet features
    #nb_features = [
    #    [256, 32, 32, 32],         # encoder features
    #    [32, 32, 32, 32, 32, 16]  # decoder features
    #]

    #nb_features = [
    #    [32, 64, 128, 128],         # encoder features
    #    [128, 128, 64, 64, 32, 16]  # decoder features
    #]

    optimizer=config_train["optimizer"] #"Adam"
    lambda_param = config_train["lambda"] # 0.05
    nb_epochs = config_train["nb_epochs"] # 200
    batch_size=config_train["batch_size"] # 16
    lr=config_train["lr"]

    x_train_fixed,x_train_moving=format_input_voxelmorph(all_volumes,pad_amount,sl_down=excluded,sl_top=-excluded)
    

    # configure unet input shape (concatenation of moving and fixed images)
    inshape = x_train_fixed.shape[1:]

    print(inshape)

    print(nb_features)

    vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)
    print("Model defined")
    # voxelmorph has a variety of custom loss classes

    if loss=="MSE":
        losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
    elif loss == "NCC":
        losses = [vxm.losses.NCC().loss, vxm.losses.Grad('l2').loss]
    elif loss =="MI":
        losses = [vxm.losses.MutualInformation().loss, vxm.losses.Grad('l2').loss]
    else:
        raise ValueError("Loss should be either MSE or Mutual Information (MI)")

    loss_weights = [1, lambda_param]

    print("Compiling Model")
    vxm_model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

    if lr is not None:
        backend.set_value(vxm_model.optimizer.learning_rate,lr)

    train_generator = vxm_data_generator(x_train_fixed,x_train_moving,batch_size=batch_size)


    nb_examples=x_train_fixed.shape[0]

    
    steps_per_epoch = int(nb_examples/batch_size)+1
    
    if "min_lr" in config_train:
        min_lr=config_train["min_lr"]
    else:
        min_lr=0.0002

    if "decay_start" in config_train:
        decay_start=config_train["decay_start"]
    else:
        decay_start=20

    curr_scheduler=lambda epoch,lr: scheduler(epoch,lr,decay,min_lr,decay_start)
    Schedulecallback = tf.keras.callbacks.LearningRateScheduler(curr_scheduler)

    callback_checkpoint=WandbModelCheckpoint(filepath=file_checkpoint,save_best_only=True,save_weights_only=True,monitor="vxm_dense_transformer_loss")


    if init_weights is not None:
        vxm_model.load_weights(init_weights)

    hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2,callbacks=[Schedulecallback,WandbMetricsLogger(),callback_checkpoint])

    vxm_model.save_weights(file_model)

    run.finish()

    plt.figure()
    plt.plot(hist.epoch, hist.history["loss"], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(file_model.split(".h5")[0]+"_loss.jpg")

    return


@ma.machine()
@ma.parameter("filename_volumes",str,default=None,description="Volumes for all motion phases nb_motion x nb_slices x npoint_x x npoint_y")
@ma.parameter("file_config_train",str,default=DEFAULT_TRAIN_CONFIG_3D,description="Config training")
@ma.parameter("suffix",str,default="",description="suffix")
@ma.parameter("init_weights",str,default=None,description="Weights initialization from .h5 file")
@ma.parameter("kept_bins",str,default=None,description="Bins to keep for training")
def train_voxelmorph_3D(filename_volumes,file_config_train,suffix,init_weights,kept_bins):

    with open(file_config_train,"r") as f:
        config_train=json.load(f)

    all_volumes = np.abs(np.load(filename_volumes))
    print("Volumes shape {}".format(all_volumes.shape))
    all_volumes=all_volumes.astype("float32")

    if kept_bins is not None:
        kept_bins_list=np.array(str.split(kept_bins,",")).astype(int)
        print(kept_bins_list)
        all_volumes=all_volumes[kept_bins_list]
    
    nb_gr,nb_slices,npoint,npoint=all_volumes.shape

    file_model=filename_volumes.split(".npy")[0]+"_vxm_model_weights_3D{}.h5".format(suffix)
    file_checkpoint="/".join(filename_volumes.split("/")[:-1])+"/model_checkpoint_3D.h5"
    print(file_checkpoint)
    run=wandb.init(
        project=file_model.split("/")[-1],
        config=config_train
    )

    #pad_amount=config_train["padding"]
    loss = config_train["loss"]
    decay=config_train["lr_decay"]
    #pad_amount=tuple(tuple(l) for l in pad_amount)

    #Finding the power of 2 "closest" and longer than  x dimension
    n=all_volumes.shape[-1]
    pad_1=2**(int(np.log2(n))+1)-n
    pad_2= 2 ** int(np.log2(n) - 1) * 3 - n

    if pad_2<0:
        pad=int(pad_1/2)
    else:
        pad = int(pad_2 / 2)

    n=all_volumes.shape[1]
    pad_1=2**(int(np.log2(n))+1)-n
    pad_2= 2 ** int(np.log2(n) - 1) * 3 - n

    if pad_2<0:
        pad_z=int(pad_1/2)
    else:
        pad_z = int(pad_2 / 2)

    pad_amount = ((0,0),(pad_z,pad_z),(pad,pad), (pad,pad))
    print(pad_amount)
    nb_features=config_train["nb_features"]
    # configure unet features
    #nb_features = [
    #    [256, 32, 32, 32],         # encoder features
    #    [32, 32, 32, 32, 32, 16]  # decoder features
    #]

    #nb_features = [
    #    [32, 64, 128, 128],         # encoder features
    #    [128, 128, 64, 64, 32, 16]  # decoder features
    #]

    optimizer=config_train["optimizer"] #"Adam"
    lambda_param = config_train["lambda"] # 0.05
    nb_epochs = config_train["nb_epochs"] # 200
    batch_size=config_train["batch_size"] # 16
    
    lr=config_train["lr"]
    x_train_fixed,x_train_moving=format_input_voxelmorph_3D(all_volumes,pad_amount)

    # configure unet input shape (concatenation of moving and fixed images)
    inshape = x_train_fixed.shape[1:]
    print(inshape)
    vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)

    # voxelmorph has a variety of custom loss classes

    if loss=="MSE":
        losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
    elif loss == "NCC":
        losses = [vxm.losses.NCC().loss, vxm.losses.Grad('l2').loss]
    elif loss =="MI":
        losses = [vxm.losses.MutualInformation().loss, vxm.losses.Grad('l2').loss]
    else:
        raise ValueError("Loss should be either MSE or Mutual Information (MI)")

    loss_weights = [1, lambda_param]

    print("Compiling Model")
    vxm_model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

    if lr is not None:
        backend.set_value(vxm_model.optimizer.learning_rate,lr)

    train_generator = vxm_data_generator_3D(x_train_fixed,x_train_moving,batch_size=batch_size)


    nb_examples=x_train_fixed.shape[0]

    
    steps_per_epoch = int(nb_examples/batch_size)+1
    

    if "min_lr" in config_train:
        min_lr=config_train["min_lr"]
    else:
        min_lr=0.0002


    curr_scheduler=lambda epoch,lr: scheduler(epoch,lr,decay,min_lr)
    Schedulecallback = tf.keras.callbacks.LearningRateScheduler(curr_scheduler)
    
    callback_checkpoint=WandbModelCheckpoint(filepath=file_checkpoint,save_best_only=True,save_weights_only=True,monitor="vxm_dense_transformer_loss")


    if init_weights is not None:
        vxm_model.load_weights(init_weights)

    hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2,callbacks=[Schedulecallback,WandbMetricsLogger(),callback_checkpoint])

    vxm_model.save_weights(file_model)

    run.finish()

    plt.figure()
    plt.plot(hist.epoch, hist.history["loss"], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(file_model.split(".h5")[0]+"_loss.jpg")



@ma.machine()
@ma.parameter("filename_volumes",str,default=None,description="Volumes for all motion phases nb_motion x nb_slices x npoint_x x npoint_y")
@ma.parameter("file_config_train",str,default=DEFAULT_TRAIN_CONFIG,description="Config training")
@ma.parameter("suffix",str,default="",description="suffix")
@ma.parameter("init_weights",str,default=None,description="Weights initialization from .h5 file")
def train_voxelmorph_torchio(filename_volumes,file_config_train,suffix,init_weights):
    
    with open(file_config_train,"r") as f:
        config_train=json.load(f)

    all_volumes = np.load(filename_volumes)
    print("Volumes shape {}".format(all_volumes.shape))
    all_volumes=all_volumes.astype("float32")

    file_model=filename_volumes.split(".npy")[0]+"_vxm_model_weights{}.h5".format(suffix)
    run=wandb.init(
        project=file_model.split("/")[-1],
        config=config_train
    )

    #pad_amount=config_train["padding"]
    loss = config_train["loss"]
    #pad_amount=tuple(tuple(l) for l in pad_amount)
    decay=config_train["lr_decay"]

    #Finding the power of 2 "closest" and longer than  x dimension
    n=all_volumes.shape[-1]
    pad_1=2**(int(np.log2(n))+1)-n
    pad_2= 2 ** int(np.log2(n) - 1) * 3 - n

    if pad_2<0:
        pad=int(pad_1/2)
    else:
        pad = int(pad_2 / 2)

    pad_amount = ((0,0),(pad,pad), (pad,pad))
    print(pad_amount)
    nb_features=config_train["nb_features"]
    # configure unet features
    #nb_features = [
    #    [256, 32, 32, 32],         # encoder features
    #    [32, 32, 32, 32, 32, 16]  # decoder features
    #]

    #nb_features = [
    #    [32, 64, 128, 128],         # encoder features
    #    [128, 128, 64, 64, 32, 16]  # decoder features
    #]

    optimizer=config_train["optimizer"] #"Adam"
    lambda_param = config_train["lambda"] # 0.05
    nb_epochs = config_train["nb_epochs"] # 200
    batch_size=config_train["batch_size"] # 16
    

    x_train_fixed,x_train_moving=format_input_voxelmorph(all_volumes,pad_amount,normalize=False)



    # configure unet input shape (concatenation of moving and fixed images)
    inshape = x_train_fixed.shape[1:]

    vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)

    # voxelmorph has a variety of custom loss classes

    if loss=="MSE":
        losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
    elif loss =="MI":
        losses = [vxm.losses.MutualInformation().loss, vxm.losses.Grad('l2').loss]
    else:
        raise ValueError("Loss should be either MSE or Mutual Information (MI)")

    loss_weights = [1, lambda_param]

    print("Compiling Model")
    vxm_model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

    
    noise=tio.RandomNoise(std=(0,0.05))
    blur=tio.RandomBlur(std=(0,2))
    gamma=tio.RandomGamma()
    noise_transform={
        blur:0.5,
        noise:0.5,
    }
    intensity_transform={
        gamma:1.0,
    }

    spatial_transforms = {
        #tio.RandomElasticDeformation(max_displacement=1): 0.2,
        tio.RandomAffine(): 1.0,
    }

    transf=tio.Compose([tio.OneOf(noise_transform,p=0.3),tio.OneOf(intensity_transform,p=0.3),tio.OneOf(spatial_transforms,p=0.3),tio.RescaleIntensity()])

    train_generator = vxm_data_generator_torchio(x_train_fixed,x_train_moving,batch_size=batch_size,transform=transf)

    nb_examples=x_train_fixed.shape[0]

    steps_per_epoch = int(nb_examples/batch_size)+1
    
    curr_scheduler=lambda epoch,lr: scheduler(epoch,lr,decay)
    Schedulecallback = tf.keras.callbacks.LearningRateScheduler(curr_scheduler)


    if init_weights is not None:
        vxm_model.load_weights(init_weights)

    hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2,callbacks=[Schedulecallback,WandbMetricsLogger()])

    vxm_model.save_weights(file_model)

    run.finish()

    plt.figure()
    plt.plot(hist.epoch, hist.history["loss"], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(file_model.split(".h5")[0]+"_loss.jpg")

@ma.machine()
@ma.parameter("filename_volumes",str,default=None,description="Volumes for all motion phases nb_motion x nb_slices x npoint_x x npoint_y")
@ma.parameter("file_config_train",str,default=DEFAULT_TRAIN_CONFIG,description="Config training")
@ma.parameter("suffix",str,default="",description="suffix")
@ma.parameter("init_weights",str,default=None,description="Weights initialization from .h5 file")
def train_voxelmorph_torchvision(filename_volumes,file_config_train,suffix,init_weights):
    
    with open(file_config_train,"r") as f:
        config_train=json.load(f)

    all_volumes = np.load(filename_volumes)
    print("Volumes shape {}".format(all_volumes.shape))
    all_volumes=np.abs(all_volumes).astype("float32")

    file_model=filename_volumes.split(".npy")[0]+"_vxm_model_weights_torchaugment{}.h5".format(suffix)
    file_checkpoint="/".join(filename_volumes.split("/")[:-1])+"/model_checkpoint.h5"
    print(file_checkpoint)
    run=wandb.init(
        project=file_model.split("/")[-1],
        config=config_train
    )

    #pad_amount=config_train["padding"]
    loss = config_train["loss"]
    #pad_amount=tuple(tuple(l) for l in pad_amount)
    decay=config_train["lr_decay"]

    #Finding the power of 2 "closest" and longer than  x dimension
    n=all_volumes.shape[-1]
    pad_1=2**(int(np.log2(n))+1)-n
    pad_2= 2 ** int(np.log2(n) - 1) * 3 - n

    if pad_2<0:
        pad=int(pad_1/2)
    else:
        pad = int(pad_2 / 2)

    pad_amount = ((0,0),(pad,pad), (pad,pad))
    print(pad_amount)
    nb_features=config_train["nb_features"]
    # configure unet features
    #nb_features = [
    #    [256, 32, 32, 32],         # encoder features
    #    [32, 32, 32, 32, 32, 16]  # decoder features
    #]

    #nb_features = [
    #    [32, 64, 128, 128],         # encoder features
    #    [128, 128, 64, 64, 32, 16]  # decoder features
    #]

    optimizer=config_train["optimizer"] #"Adam"
    lambda_param = config_train["lambda"] # 0.05
    nb_epochs = config_train["nb_epochs"] # 200
    batch_size=config_train["batch_size"] # 16

    all_groups_combination=config_train["all_groups_combination"]
    perc_augmentation = config_train["perc_augmentation"]
    test_size = config_train["test_size"]

    x_fixed,x_moving=format_input_voxelmorph(all_volumes,pad_amount,normalize=True,all_groups_combination=all_groups_combination)
    x_train_fixed, x_test_fixed, x_train_moving, x_test_moving = train_test_split(x_fixed, x_moving, test_size=test_size, random_state=42)

    print("Train size : {}".format(x_train_fixed.shape))
    print("Test size : {}".format(x_test_fixed.shape))

    # configure unet input shape (concatenation of moving and fixed images)
    inshape = x_train_fixed.shape[1:]

    vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)

    # voxelmorph has a variety of custom loss classes

    if loss=="MSE":
        losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
    elif loss =="MI":
        losses = [vxm.losses.MutualInformation().loss, vxm.losses.Grad('l2').loss]
    else:
        raise ValueError("Loss should be either MSE or Mutual Information (MI)")

    loss_weights = [1, lambda_param]

    print("Compiling Model")
    vxm_model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

    
    transform_color=T.Lambda(lambda x : T.functional.adjust_contrast(T.functional.adjust_brightness(x,1.5),1.5))
    identity=T.Lambda(lambda x : x)
    transform_list=[T.GaussianBlur(9,2),T.GaussianBlur(5,2),transform_color,T.Lambda(lambda x : x + 0.05*torch.randn_like(x)),T.Lambda(lambda x : x + 0.1*torch.randn_like(x)),identity]
    probabilities= [perc_augmentation/(len(transform_list)-1)]*(len(transform_list)-1)+[1-perc_augmentation]
    train_generator = vxm_data_generator_torchvision(x_train_fixed,x_train_moving,batch_size=batch_size,transform_list=transform_list,probabilities=probabilities)

    validation_generator=vxm_data_generator_torchvision(x_test_fixed,x_test_moving,batch_size=batch_size,transform_list=transform_list,probabilities=[0]*(len(transform_list)-1)+[1.0])

    nb_examples_training=x_train_fixed.shape[0]
    nb_examples_test = x_test_fixed.shape[0]
    steps_per_epoch_training = int(nb_examples_training/batch_size)+1
    steps_per_epoch_test = int(nb_examples_test / batch_size) + 1
    
    curr_scheduler=lambda epoch,lr: scheduler(epoch,lr,decay)
    Schedulecallback = tf.keras.callbacks.LearningRateScheduler(curr_scheduler)


    if init_weights is not None:
        vxm_model.load_weights(init_weights)

    callback_checkpoint=WandbModelCheckpoint(filepath=file_checkpoint,save_best_only=True,save_weights_only=True)

    hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, validation_data=validation_generator,steps_per_epoch=steps_per_epoch_training,validation_steps=steps_per_epoch_test, verbose=2,callbacks=[Schedulecallback,WandbMetricsLogger(log_freq=8),callback_checkpoint])

    vxm_model.save_weights(file_model)

    run.finish()

    plt.figure()
    plt.plot(hist.epoch, hist.history["loss"], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(file_model.split(".h5")[0]+"_loss.jpg")


@ma.machine()
@ma.parameter("file_config_train", str, default=DEFAULT_TRAIN_CONFIG, description="Config training")
@ma.parameter("kept_bins", str, default=None, description="Bins to keep for training")
@ma.parameter("suffix", str, default="", description="suffix")
@ma.parameter("init_weights", str, default=None, description="Weights initialization from .h5 file")
def train_voxelmorph_torchvision_multiple_patients(file_config_train, suffix, init_weights, kept_bins):
    with open(file_config_train,"r") as f:
        config_train=json.load(f)

    filenames = config_train["filenames"]
    folder = config_train["folder"]

    file_model = folder + "/multiple_patients_vxm_model_weights_torchaugment{}.h5".format(suffix)
    file_checkpoint = folder + "/multiple_patients_model_checkpoint.h5"
    print(file_checkpoint)
    run = wandb.init(
        project=file_model.split("/")[-1],
        config=config_train
    )

    # pad_amount=config_train["padding"]
    loss = config_train["loss"]
    # pad_amount=tuple(tuple(l) for l in pad_amount)
    decay = config_train["lr_decay"]

    n = 0
    volumes_all_patients = []
    n_all_patients = []

    for file_volume in filenames:
        all_volumes = np.load(folder + file_volume)
        print("Volumes shape {}".format(all_volumes.shape))
        all_volumes = np.abs(all_volumes).astype("float32")
        if kept_bins is not None:
            kept_bins_list = np.array(str.split(kept_bins, ",")).astype(int)
            print(kept_bins_list)
            all_volumes = all_volumes[kept_bins_list]
        n_curr = all_volumes.shape[-1]
        n_all_patients.append(n_curr)
        if n_curr > n:
            n = n_curr
        volumes_all_patients.append(all_volumes)
    pads = ((n - np.array(n_all_patients)) / 2).astype(int)
    volumes_all_patients = [np.pad(v, ((0, 0), (0, 0), (pads[i], pads[i]), (pads[i], pads[i])), "constant") for i, v in
                            enumerate(volumes_all_patients)]

    for v in volumes_all_patients:
        print(v.shape)

    # Finding the power of 2 "closest" and longer than  x dimension

    pad_1 = 2 ** (int(np.log2(n)) + 1) - n
    pad_2 = 2 ** int(np.log2(n) - 1) * 3 - n

    if pad_2 < 0:
        pad = int(pad_1 / 2)
    else:
        pad = int(pad_2 / 2)

    if n%2==0:
        pad=0

    pad_amount = ((0, 0), (pad, pad), (pad, pad))
    print(pad_amount)
    nb_features = config_train["nb_features"]
    # configure unet features
    # nb_features = [
    #    [256, 32, 32, 32],         # encoder features
    #    [32, 32, 32, 32, 32, 16]  # decoder features
    # ]

    # nb_features = [
    #    [32, 64, 128, 128],         # encoder features
    #    [128, 128, 64, 64, 32, 16]  # decoder features
    # ]

    optimizer = config_train["optimizer"]  # "Adam"
    lambda_param = config_train["lambda"]  # 0.05
    nb_epochs = config_train["nb_epochs"]  # 200
    batch_size = config_train["batch_size"]  # 16

    all_groups_combination = config_train["all_groups_combination"]
    perc_augmentation = config_train["perc_augmentation"]
    test_size = config_train["test_size"]

    lr = config_train["lr"]

    x_fixed_all = []
    x_moving_all = []

    for v in volumes_all_patients:
        x_fixed, x_moving = format_input_voxelmorph(v, pad_amount, normalize=True,
                                                    all_groups_combination=all_groups_combination)
        x_fixed_all.append(x_fixed)
        x_moving_all.append(x_moving)
        print(x_fixed.shape)

    x_fixed_all = np.concatenate(x_fixed_all, axis=0)
    x_moving_all = np.concatenate(x_moving_all, axis=0)
    x_train_fixed, x_test_fixed, x_train_moving, x_test_moving = train_test_split(x_fixed_all, x_moving_all,
                                                                                  test_size=test_size, random_state=42)

    print("Train size : {}".format(x_train_fixed.shape))
    print("Test size : {}".format(x_test_fixed.shape))

    # configure unet input shape (concatenation of moving and fixed images)
    inshape = x_train_fixed.shape[1:]

    vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)

    # voxelmorph has a variety of custom loss classes

    if loss == "MSE":
        losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
    elif loss == "NCC":
        losses = [vxm.losses.NCC().loss, vxm.losses.Grad('l2').loss]
    elif loss == "MI":
        losses = [vxm.losses.MutualInformation().loss, vxm.losses.Grad('l2').loss]
    else:
        raise ValueError("Loss should be either MSE or Mutual Information (MI)")

    loss_weights = [1, lambda_param]

    print("Compiling Model")
    vxm_model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

    if lr is not None:
        backend.set_value(vxm_model.optimizer.learning_rate, lr)

    transform_color = T.Lambda(lambda x: T.functional.adjust_contrast(T.functional.adjust_brightness(x, 1.5), 1.5))
    identity = T.Lambda(lambda x: x)
    transform_list = [T.GaussianBlur(9, 2), T.GaussianBlur(5, 2), transform_color,
                      T.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),
                      T.Lambda(lambda x: x + 0.1 * torch.randn_like(x)), identity]
    probabilities = [perc_augmentation / (len(transform_list) - 1)] * (len(transform_list) - 1) + [
        1 - perc_augmentation]
    train_generator = vxm_data_generator_torchvision(x_train_fixed, x_train_moving, batch_size=batch_size,
                                                     transform_list=transform_list, probabilities=probabilities)

    validation_generator = vxm_data_generator_torchvision(x_test_fixed, x_test_moving, batch_size=batch_size,
                                                          transform_list=transform_list,
                                                          probabilities=[0] * (len(transform_list) - 1) + [1.0])

    nb_examples_training = x_train_fixed.shape[0]
    nb_examples_test = x_test_fixed.shape[0]
    steps_per_epoch_training = int(nb_examples_training / batch_size) + 1
    steps_per_epoch_test = int(nb_examples_test / batch_size) + 1

    if "min_lr" in config_train:
        min_lr = config_train["min_lr"]

    curr_scheduler = lambda epoch, lr: scheduler(epoch, lr, decay, min_lr)
    Schedulecallback = tf.keras.callbacks.LearningRateScheduler(curr_scheduler)

    if init_weights is not None:
        vxm_model.load_weights(init_weights)

    callback_checkpoint = WandbModelCheckpoint(filepath=file_checkpoint, save_best_only=True, save_weights_only=True)

    hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, validation_data=validation_generator,
                                   steps_per_epoch=steps_per_epoch_training, validation_steps=steps_per_epoch_test,
                                   verbose=2,
                                   callbacks=[Schedulecallback, WandbMetricsLogger()])

    vxm_model.save_weights(file_model)

    run.finish()

    plt.figure()
    plt.plot(hist.epoch, hist.history["loss"], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(file_model.split(".h5")[0] + "_loss.jpg")

@ma.machine()
@ma.parameter("filename_volumes",str,default=None,description="Volumes for all motion phases nb_motion x nb_slices x npoint_x x npoint_y")
@ma.parameter("file_model",str,default=None,description="Trained Model weights")
@ma.parameter("file_config_train",str,default=DEFAULT_TRAIN_CONFIG,description="Config training")

def evaluate_model(filename_volumes,file_model,file_config_train):
    with open(file_config_train,"r") as f:
        config_train=json.load(f)

    dx=1# Might want to change that (but not that important for now given that the registration is 2D)
    dy=1
    dz=5

    all_volumes = np.load(filename_volumes)
    print("Volumes shape {}".format(all_volumes.shape))
    all_volumes=all_volumes.astype("float32")
    nb_gr=all_volumes.shape[0]
    
    
    
    #pad_amount=config_train["padding"]
    #pad_amount=tuple(tuple(l) for l in pad_amount)

    n = all_volumes.shape[-1]
    pad_1 = 2 ** (int(np.log2(n)) + 1) - n
    pad_2 = 2 ** int(np.log2(n) - 1) * 3 - n

    if pad_2 < 0:
        pad = int(pad_1 / 2)
    else:
        pad = int(pad_2 / 2)

    pad_amount = ((0, 0), (pad, pad), (pad, pad))


    nb_features=config_train["nb_features"]

    for gr in range(nb_gr-1):
        x_val_fixed,x_val_moving=format_input_voxelmorph(all_volumes[[gr,gr+1]],pad_amount)
        inshape=x_val_fixed.shape[1:]

        vxm_model=vxm.networks.VxmDense(inshape, nb_features, int_steps=0)
        vxm_model.load_weights(file_model)

        
        val_input=[x_val_moving[...,None],x_val_fixed[...,None]]
        val_pred=vxm_model.predict(val_input)

        field_array=np.zeros(shape=val_pred[1].shape[:-1]+(3,),dtype=val_pred[1].dtype)
        field_array[:,:,:,:2]=val_pred[1]
        field=sitk.GetImageFromArray(field_array,isVector=True)
        field.SetSpacing([dx,dy,dz])
        moving_3D=sitk.GetImageFromArray(val_input[0][:,:,:,0])
        moving_3D.SetSpacing([dx,dy,dz])
        fixed_3D=sitk.GetImageFromArray(val_input[1][:,:,:,0])
        fixed_3D.SetSpacing([dx,dy,dz])
        moved_3D=sitk.GetImageFromArray(val_pred[0][:,:,:,0])
        moved_3D.SetSpacing([dx,dy,dz])

        sitk.WriteImage(field,file_model.split(".h5")[0]+"_displacement_field_vm_gr{}.nii".format(gr))
        sitk.WriteImage(moving_3D,file_model.split(".h5")[0]+"_moving_vm_gr{}.mha".format(gr))
        sitk.WriteImage(fixed_3D,file_model.split(".h5")[0]+"_fixed_vm_gr{}.mha".format(gr))
        sitk.WriteImage(moved_3D,file_model.split(".h5")[0]+"_moved_vm_gr{}.mha".format(gr))


@ma.machine()
@ma.parameter("filename_volumes",str,default=None,description="Volumes for all motion phases nb_motion x nb_slices x npoint_x x npoint_y")
@ma.parameter("file_model",str,default=None,description="Trained Model weights")
@ma.parameter("file_config_train",str,default=DEFAULT_TRAIN_CONFIG,description="Config training")
@ma.parameter("file_deformation",str,default=None,description="Initial deformation if doing multiple pass of registration algo")
@ma.parameter("niter",int,default=1,description="Number of iterations for registration")
@ma.parameter("resolution",int,default=None,description="Image resolution")
@ma.parameter("metric",["abs","phase","real","imag"],default="abs",description="Metric to register")
@ma.parameter("axis",int,default=None,description="Change registration axis")
def register_allbins_to_baseline(filename_volumes,file_model,file_config_train,niter,file_deformation,resolution,metric,axis):

    with open(file_config_train,"r") as f:
        config_train=json.load(f)

    all_volumes = np.load(filename_volumes)


    if resolution is None:
        filename_registered_volumes=filename_volumes.split(".npy")[0]+"_registered.npy"
        filename_deformation = filename_volumes.split(".npy")[0] + "_deformation_map.npy"
    else:
        filename_registered_volumes=filename_volumes.split(".npy")[0]+"_registered_res{}.npy".format(resolution)
        filename_deformation = filename_volumes.split(".npy")[0] + "_deformation_map_res{}.npy".format(resolution)



    if metric=="abs":
        all_volumes = np.abs(all_volumes)
    elif metric=="phase":
        all_volumes = np.angle(all_volumes)
        filename_registered_volumes = str.replace(filename_registered_volumes,"registered","registered_phase")
        filename_deformation = str.replace(filename_deformation, "deformation_map", "deformation_map_phase")

    elif metric=="real":
        all_volumes = np.real(all_volumes)
        filename_deformation = str.replace(filename_deformation, "deformation_map", "deformation_map_real")

    elif metric=="imag":
        all_volumes = np.imag(all_volumes)
        filename_deformation = str.replace(filename_deformation, "deformation_map", "deformation_map_imag")

    else:
        raise ValueError("metric unknown - choose from abs/phase/real/imag")

    if file_deformation is not None:
        deformation_map=np.load(file_deformation)
    else:
        deformation_map=None

    print("Volumes shape {}".format(all_volumes.shape))
    all_volumes=all_volumes.astype("float32")
    nb_gr,nb_slices,npoint,npoint=all_volumes.shape

    if resolution is not None:
        all_volumes=resize(all_volumes,(nb_gr,nb_slices,resolution,resolution))
    
    if axis is not None:
        all_volumes=np.moveaxis(all_volumes,axis+1,1)
        # all_volumes=resize(all_volumes,(nb_gr,npoint,npoint,npoint))
    
    #pad_amount=config_train["padding"]
    #pad_amount=tuple(tuple(l) for l in pad_amount)
    n = np.maximum(all_volumes.shape[-1],all_volumes.shape[-2])
    pad_1 = 2 ** (int(np.log2(n)) + 1) - n
    pad_2 = 2 ** int(np.log2(n) - 1) * 3 - n

    if pad_2 < 0:
        pad = int(pad_1 / 2)
    else:
        pad = int(pad_2 / 2)

    # if n%2==0:
    #     pad=0
    pad_x=int((2*pad+n-all_volumes.shape[-2])/2)
    pad_y=int((2*pad+n-all_volumes.shape[-1])/2)

    pad_amount = ((0, 0), (pad_x, pad_x), (pad_y, pad_y))
    print(pad_amount)

    nb_features=config_train["nb_features"]
    inshape=np.pad(all_volumes[0],pad_amount,mode="constant").shape[1:]
    print(inshape)
    vxm_model=vxm.networks.VxmDense(inshape, nb_features, int_steps=0)
    vxm_model.load_weights(file_model)

    # Filtering out slices with only 0 as it seems to be buggy
    # sl_down_non_zeros = 0
    # while not (np.any(all_volumes[:, sl_down_non_zeros])):
    #     sl_down_non_zeros += 1
    #
    # sl_top_non_zeros = nb_slices
    # while not (np.any(all_volumes[:, sl_top_non_zeros-1])):
    #     sl_top_non_zeros -= 1
    #
    # print(sl_top_non_zeros)
    # print(sl_down_non_zeros)
    # all_volumes=all_volumes[:,sl_down_non_zeros:sl_top_non_zeros]
    registered_volumes=copy(all_volumes)
    mapxbase_all=np.zeros_like(all_volumes)
    mapybase_all = np.zeros_like(all_volumes)
    print(registered_volumes.shape)


    i=0
    while i<niter:
        print("Registration for iter {}".format(i+1))
        for gr in range(nb_gr):
            registered_volumes[gr],mapxbase_all[gr],mapybase_all[gr]=register_motionbin(vxm_model,all_volumes,gr,pad_amount,deformation_map)

        all_volumes=copy(registered_volumes)
        deformation_map=np.stack([mapxbase_all,mapybase_all],axis=0)
        print(deformation_map.shape)
        i+=1

    if axis is not None:
        registered_volumes=np.moveaxis(registered_volumes,1,axis+1)
        # registered_volumes=resize(registered_volumes,(nb_gr,nb_slices,npoint,npoint))
        # registered_volumes=registered_volumes[:,::int(npoint/nb_slices)]
        deformation_map=np.moveaxis(deformation_map,2,axis+2)
        # deformation_map=resize(deformation_map,(2,nb_gr,nb_slices,npoint,npoint))


    if resolution is not None:
        deformation_map=resize(deformation_map,(2,nb_gr,nb_slices,npoint,npoint),order=3)
    np.save(filename_registered_volumes,registered_volumes)
    np.save(filename_deformation, deformation_map)
    
@ma.machine()
@ma.parameter("file_deformation",str,default=None,description="deformation")
@ma.parameter("gr",int,default=None,description="Respiratory bin")
@ma.parameter("sl",int,default=None,description="Slice")
@ma.parameter("axis",int,default=None,description="Change registration axis")
def plot_deformation_flow(file_deformation,gr,sl,axis):
    deformation_map=np.load(file_deformation)
    if gr is None:
        gr=deformation_map.shape[1]-1
    if axis is not None:
        deformation_map=np.moveaxis(deformation_map,axis+2,2)
    file_plot=file_deformation.split(".npy")[0]+"_gr{}sl{}.jpg".format(gr,sl)
    print(file_plot)
    plot_deformation_map(deformation_map[:,gr,sl],us=4,save_file=file_plot)


@ma.machine()
@ma.parameter("filename_volumes",str,default=None,description="Volumes for all motion phases nb_motion x nb_slices x npoint_x x npoint_y")
@ma.parameter("file_deformation",str,default=None,description="Initial deformation if doing multiple pass of registration algo")
@ma.parameter("axis",int,default=None,description="Registration axis")
def apply_deformation_map(filename_volumes,file_deformation,axis):
    all_volumes = np.load(filename_volumes)
    filename_registered_volumes = filename_volumes.split(".npy")[0] + "_registered_by_deformation.npy"

    # if metric=="abs":
    #     all_volumes = np.abs(all_volumes)
    # elif metric=="phase":
    #     all_volumes = np.angle(all_volumes)
    #     filename_registered_volumes = str.replace(filename_registered_volumes,"registered","registered_phase")
    #
    # elif metric=="real":
    #     all_volumes = np.real(all_volumes)
    #     filename_registered_volumes = str.replace(filename_registered_volumes,"registered","registered_real")
    #
    # elif metric=="imag":
    #     all_volumes = np.imag(all_volumes)
    #     filename_registered_volumes = str.replace(filename_registered_volumes,"registered","registered_imag")
    #
    # else:
    #     raise ValueError("metric unknown - choose from abs/phase/real/imag")

    deformation_map=np.load(file_deformation)

    # if axis is not None:
    #     all_volumes=np.moveaxis(all_volumes,axis+1,1)
    #     deformation_map=np.moveaxis(deformation_map,axis+2,2)

    deformed_volumes = np.zeros_like(all_volumes)
    nb_gr=all_volumes.shape[0]
    print(deformed_volumes.dtype)
    print(all_volumes.dtype)

    

    for gr in range(nb_gr):
        deformed_volumes[gr]= apply_deformation_to_complex_volume(all_volumes[gr], deformation_map[:,gr],axis=axis)

    # if axis is not None:
    #     deformed_volumes=np.moveaxis(deformed_volumes,1,axis+1)
        

    np.save(filename_registered_volumes,deformed_volumes)

    return

@ma.machine()
@ma.parameter("filename_volumes",str,default=None,description="Volumes for all motion phases nb_motion x nb_slices x npoint_x x npoint_y")
@ma.parameter("file_model",str,default=None,description="Trained Model weights")
@ma.parameter("file_config_train",str,default=DEFAULT_TRAIN_CONFIG_3D,description="Config training")
@ma.parameter("file_deformation",str,default=None,description="Initial deformation if doing multiple pass of registration algo")
@ma.parameter("niter",int,default=1,description="Number of iterations for registration")
def register_allbins_to_baseline_3D(filename_volumes,file_model,file_config_train,niter,file_deformation):

    with open(file_config_train,"r") as f:
        config_train=json.load(f)

    all_volumes = np.abs(np.load(filename_volumes))
    filename_registered_volumes=filename_volumes.split(".npy")[0]+"_registered_3D.npy"
    filename_deformation = filename_volumes.split(".npy")[0] + "_deformation_map_3D.npy"

    if file_deformation is not None:
        deformation_map=np.load(file_deformation)
    else:
        deformation_map=None

    print("Volumes shape {}".format(all_volumes.shape))
    all_volumes=all_volumes.astype("float32")
    nb_gr=all_volumes.shape[0]
    
    
    
    #pad_amount=config_train["padding"]
    #pad_amount=tuple(tuple(l) for l in pad_amount)
    n = all_volumes.shape[-1]
    pad_1 = 2 ** (int(np.log2(n)) + 1) - n
    pad_2 = 2 ** int(np.log2(n) - 1) * 3 - n

    if pad_2 < 0:
        pad = int(pad_1 / 2)
    else:
        pad = int(pad_2 / 2)

    n = all_volumes.shape[1]

    pad_1 = 2 ** (int(np.log2(n)) + 1) - n
    pad_2 = 2 ** int(np.log2(n) - 1) * 3 - n

    if pad_2 < 0:
        pad_z = int(pad_1 / 2)
    else:
        pad_z = int(pad_2 / 2)

    if (2 ** (int(np.log2(n)))-n)==0:
        pad_z=0

    pad_amount = ((pad_z, pad_z), (pad, pad), (pad, pad))
    print(pad_amount)

    nb_features=config_train["nb_features"]
    inshape=np.pad(all_volumes[0],pad_amount,mode="constant").shape
    #print(inshape)
    #print(inshape)
    vxm_model=vxm.networks.VxmDense(inshape, nb_features, int_steps=0)
    vxm_model.load_weights(file_model)
    
    registered_volumes=copy(all_volumes)
    mapxbase_all=np.zeros_like(all_volumes)
    mapybase_all = np.zeros_like(all_volumes)
    mapzbase_all = np.zeros_like(all_volumes)
    i=0
    while i<niter:
        print("Registration for iter {}".format(i+1))
        for gr in range(nb_gr):
            registered_volumes[gr],mapzbase_all[gr],mapxbase_all[gr],mapybase_all[gr]=register_motionbin_3D(vxm_model,all_volumes,gr,pad_amount,deformation_map)

        all_volumes=copy(registered_volumes)
        deformation_map=np.stack([mapzbase_all,mapxbase_all,mapybase_all],axis=0)
        print(deformation_map.shape)
        i+=1
    np.save(filename_registered_volumes,registered_volumes)
    np.save(filename_deformation, deformation_map)
    

def register_motionbin(vxm_model,all_volumes,gr,pad_amount,deformation_map=None):
    curr_gr=gr
    moving_volume=np.pad(all_volumes[curr_gr],pad_amount,mode="constant")
    nb_slices=all_volumes.shape[1]

    print(all_volumes.shape)
    

    if deformation_map is None:
        # print("Here")
        mapx_base, mapy_base = np.meshgrid(np.arange(all_volumes.shape[-1]), np.arange(all_volumes.shape[-2]))
        mapx_base=np.tile(mapx_base,reps=(nb_slices,1,1))
        mapy_base = np.tile(mapy_base, reps=(nb_slices, 1, 1))
        # print("Here 2")
    else:
        #print("Applying existing deformation map")
        mapx_base=deformation_map[0,gr]
        mapy_base=deformation_map[1,gr]

    #print("mapx_base shape: {}".format(mapx_base.shape))
    #print("mapy_base shape: {}".format(mapy_base.shape))
    #print(mapx_base.shape)
    while curr_gr>0:
        # print(all_volumes[curr_gr-1].shape)
        # print(moving_volume.shape)

        input=np.stack([np.pad(all_volumes[curr_gr-1],pad_amount,mode="constant"),moving_volume],axis=0)
        # print(input.shape)
        x_val_fixed,x_val_moving=format_input_voxelmorph(input,((0,0),(0,0),(0,0)),sl_down=0,sl_top=nb_slices,exclude_zero_slices=False)
        val_input=[x_val_moving[...,None],x_val_fixed[...,None]]
        #print(val_input.shape)


        val_pred=vxm_model.predict(val_input)
        moving_volume=val_pred[0][:,:,:,0]
        #print(val_pred[1][:,:,:].shape)
        
        mapx_base=mapx_base+unpad(val_pred[1][:,:,:,1],pad_amount)
        mapy_base=mapy_base+unpad(val_pred[1][:,:,:,0],pad_amount)

        curr_gr=curr_gr-1
    print("Moving volume shape : {}".format(moving_volume.shape))

    if gr==0:
        moving_volume/=np.max(moving_volume,axis=(1,2),keepdims=True)
    unpadded_moving_volume=unpad(moving_volume,pad_amount)
    # print("Unpadded Moving volume shape : {}".format(unpadded_moving_volume.shape))
    # print(mapx_base.shape)
    # print(mapy_base.shape)
    print("Norm unpadded_moving_volume : {}".format(np.linalg.norm(unpadded_moving_volume)))
    print("Max unpadded_moving_volume : {}".format(np.max(unpadded_moving_volume)))
    print("Min unpadded_moving_volume : {}".format(np.min(unpadded_moving_volume)))


    
          
    return unpadded_moving_volume,mapx_base,mapy_base


def register_motionbin_3D(vxm_model,all_volumes,gr,pad_amount,deformation_map=None):
    curr_gr=gr
    moving_volume=np.pad(all_volumes[curr_gr],pad_amount,mode="constant")
    nb_slices=all_volumes.shape[1]

    #print(all_volumes.shape)

    if deformation_map is None:
        mapz_base,mapx_base, mapy_base = np.meshgrid(np.arange(all_volumes.shape[1]),np.arange(all_volumes.shape[2]), np.arange(all_volumes.shape[3]),indexing="ij")
        
    else:
        #print("Applying existing deformation map")
        mapz_base=deformation_map[0,gr]
        mapx_base=deformation_map[1,gr]
        mapy_base=deformation_map[2,gr]

    #print("mapx_base shape: {}".format(mapx_base.shape))
    #print("mapy_base shape: {}".format(mapy_base.shape))
    #print("mapz_base shape: {}".format(mapz_base.shape))
    #print(mapx_base.shape)
    while curr_gr>0:
        #print("curr_gr {}".format(curr_gr))
        #print(all_volumes[curr_gr-1].shape)
        #print(moving_volume.shape)
        #print("all_volumes[curr_gr-1].shape {}".format(all_volumes[curr_gr-1].shape))
        #print("moving_volume.shape {}".format(moving_volume.shape))
        input=np.stack([np.pad(all_volumes[curr_gr-1],pad_amount,mode="constant"),moving_volume],axis=0)
        #print(input.shape)
        x_val_fixed,x_val_moving=format_input_voxelmorph_3D(input,((0,0),(0,0),(0,0),(0,0)),sl_down=0,sl_top=nb_slices)
        #print(x_val_fixed.shape)
        val_input=[x_val_moving,x_val_fixed]
        #print(val_input.shape)

        val_pred=vxm_model.predict(val_input)
        moving_volume=val_pred[0][0,:,:,:,0]
        #print("Moving volume shape after registration {}".format(moving_volume.shape))
        #print("val_pred shape {}".format(val_pred[1][:,:,:].shape))
        mapz_base=mapz_base+unpad(val_pred[1][0,:,:,:,1],pad_amount)
        mapx_base=mapx_base+unpad(val_pred[1][0,:,:,:,0],pad_amount)
        mapy_base=mapy_base+unpad(val_pred[1][0,:,:,:,2],pad_amount)

        curr_gr=curr_gr-1
    #print("Moving volume shape {}:".format(moving_volume.shape))
    return unpad(moving_volume,pad_amount),mapz_base,mapx_base,mapy_base




def scheduler(epoch, lr,decay=0.005,min_lr=None,decay_start=20):
  if epoch < decay_start:
    return lr
  else:
    if min_lr is None:
        return lr * tf.math.exp(-decay)
    else:
        return np.maximum(lr * tf.math.exp(-decay),min_lr)



toolbox = Toolbox("script_VoxelMorph_machines.", description="Volume registration with Voxelmorph")
toolbox.add_program("train_voxelmorph", train_voxelmorph)
toolbox.add_program("train_voxelmorph_3D", train_voxelmorph_3D)
toolbox.add_program("train_voxelmorph_torchio", train_voxelmorph_torchio)
toolbox.add_program("train_voxelmorph_torchvision", train_voxelmorph_torchvision)
toolbox.add_program("train_voxelmorph_torchvision_multiple_patients", train_voxelmorph_torchvision_multiple_patients)
toolbox.add_program("evaluate_model", evaluate_model)
toolbox.add_program("register_allbins_to_baseline", register_allbins_to_baseline)
toolbox.add_program("register_allbins_to_baseline_3D", register_allbins_to_baseline_3D)
toolbox.add_program("apply_deformation_map", apply_deformation_map)
toolbox.add_program("plot_deformation_flow", plot_deformation_flow)

if __name__ == "__main__":
    toolbox.cli()

