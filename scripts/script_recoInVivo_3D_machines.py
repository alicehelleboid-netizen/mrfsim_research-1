
import matplotlib
# matplotlib.use("Agg")
from mrfsim.image_series import *
from mrfsim.dictoptimizers import SimpleDictSearch

from pathlib import Path

from PIL import Image
from mrfsim.utils_simu import *
#from utils_reco import calculate_sensitivity_map_3D,kdata_aggregate_center_part,calculate_displacement,build_volume_singular_2Dplus1_cc_allbins_registered,simulate_nav_images_multi,calculate_displacement_ml,estimate_weights_bins,calculate_displacements_singlechannel,calculate_displacements_singlechannel,calculate_displacements_allchannels,coil_compression_2Dplus1,build_volume_2Dplus1_cc_allbins
from mrfsim.utils_reco import *
from mrfsim.utils_mrf import *
from mrfsim.trajectory import *
from mrfsim import io

import math
import nibabel as nib
try :
    import cupy as cp
except:
    pass
# machines

import machines as ma


from machines import Toolbox
import glob


import sys
os.environ['PATH'] = os.environ['TOOLBOX_PATH'] + ":" + os.environ['PATH']
sys.path.append(os.environ['TOOLBOX_PATH'] + "/python/")
import cfl
from bart import bart

DEFAULT_OPT_CONFIG_2STEPS={
    "type": "CF_iterative_2Dplus1",
    "pca":10,
    "split":200,
    "useGPU": True,
    "niter": 0,
    "mu":1,
    "mu_TV":0.5,
    "weights_TV":[1.0,0.1,0.1],
    "volumes_type":"singular",
    "nspoke":1400,
    "return_matched_signals":False,
    "return_cost":True,
    "clustering":True,
    "mu_bins":None,
    "log":False
}


@ma.machine()
@ma.parameter("filename", str, default=None, description="Siemens K-space data .dat file")
@ma.parameter("index", int, default=-1, description="Header index")
def getTR(filename,index):
    # twix = twixtools.read_twix(filename, optional_additional_maps=["sWipMemBlock"],
                                #    optional_additional_arrays=["SliceThickness"])


    hdr = io_twixt.parse_twixt_header(filename)
    
    adFree = get_specials(hdr, type="adFree",index=index)
    
        
    minTE = hdr[index]["alTE[0]"] / 1e3
    echoSpacing = adFree[1]
    dTR = echoSpacing - minTE
    total_TR = hdr[index]["alTR[0]"] / 1e6
    invTime = adFree[0]

    print(adFree)

    print("dTR is {} ms".format(dTR))
    print("Total TR is {} s".format(total_TR))
    print("TI is {} ms".format(invTime))
    return


@ma.machine()
@ma.parameter("filepickle", str, default=None, description=".pkl")
def showPickle(filepickle):
    with open(filepickle,"rb") as file:
        dico=pickle.load(file)
    print(dico)

    return

@ma.machine()
@ma.parameter("filename", str, default=None, description="Filename for getting metatada")
@ma.parameter("filemha", str, default=None, description="Map (.mha)")
@ma.parameter("suffix",str,default="")
@ma.parameter("nifti",bool,default=False)
def getGeometry(filename,filemha,suffix,nifti):
    if filename is None:
        filename=str.split(filemha,"_MRF_map")[0]+".dat"
    if nifti:
        filemha_adjusted=str.replace(filemha,".mha","{}.nii".format(suffix))
    else:
        filemha_adjusted=str.replace(filemha,".mha","{}.mha".format(suffix))
    geom,is3D,orientation=get_volume_geometry(filename)
    data=np.array(io.read(filemha))
    print(data.shape)
    if is3D:
        #data=np.flip(np.moveaxis(data,0,-1),axis=(1,2))
        if orientation=="coronal":
            data=np.flip(np.moveaxis(data,(0,1,2),(1,2,0)),axis=(1))
            #data=np.moveaxis(data,0,2)
        elif orientation=="transversal":
            data=np.moveaxis(data,0,2)[:,::-1]
    else:
        data=np.moveaxis(data,0,2)[:,::-1]
    
    
    vol = io.Volume(data, **geom)
    io.write(filemha_adjusted,vol)
    return

@ma.machine()
@ma.parameter("filedico", str, default=None, description="Dictionary .pkl containing parameters")
@ma.parameter("filevolume", str, default=None, description="Volume (.npy)")
@ma.parameter("suffix",str,default="")
@ma.parameter("nifti",bool,default=False)
@ma.parameter("apply_offset",bool,default=False,description="Apply offset (for having the absolute position in space) - should be false for unwarping volumes")
@ma.parameter("reorient",bool,default=True,description="Reorient input volumes")
@ma.parameter("gr",int,default=None,description="Bin number if dynamic data")
def convertArrayToImage(filedico,filevolume,suffix,nifti,apply_offset,reorient,gr):


    extension=str.split(filevolume,".")[-1]


    print(extension)
    if ("nii" in extension) or (extension=="mha"):
        func_load=io.read
    elif (extension=="npy"):
        func_load=np.load
    else:
        raise ValueError("Unknown extension {}".format(extension))

    if nifti:
        new_extension="nii"
    else:
        new_extension="mha"

    filemha_adjusted=str.replace(filevolume,".{}".format(extension),"{}.{}".format(suffix,new_extension))


    if gr is not None:
        filemha_adjusted=str.replace(filemha_adjusted,".{}".format(new_extension),"_gr{}.{}".format(gr,new_extension))

    print(filemha_adjusted)
    
    with open(filedico,"rb") as file:
        dico=pickle.load(file)
    
    spacing=dico["spacing"]
    origin=dico["origin"]
    orientation=dico["orientation"]
    is3D=dico["is3D"]

    if apply_offset:
        offset=dico["offset"]
        print("Applying offset {}".format(offset))
        
        origin=np.array(origin)
        origin[-1]=origin[-1]+offset
        origin=tuple(origin)
        

    geom={"origin":origin,"spacing":spacing}
    print(geom)

    if gr is not None:
        data=np.abs(np.array(func_load(filevolume))[gr].squeeze())
    else:
        data=np.abs(np.array(func_load(filevolume)).squeeze())
    print(data.shape)
    if reorient:
        print("Reorienting input volume")
        if is3D:
            #data=np.flip(np.moveaxis(data,0,-1),axis=(1,2))
            offset=data.ndim-3
            if orientation=="coronal":
                
                data=np.flip(np.moveaxis(data,(offset,offset+1,offset+2),(offset+1,offset+2,offset)),axis=(offset,offset+1))
                #data=np.moveaxis(data,0,2)
            elif orientation=="transversal":
                # data=np.moveaxis(data,offset,offset+2)
                data=np.flip(np.moveaxis(data,offset,offset+2),axis=(offset+1,offset+2))

            elif orientation=="sagittal":
                # data=np.moveaxis(data,offset,offset+2)
                data=np.flip(np.moveaxis(data,(offset,offset+1,offset+2),(offset,offset+2,offset+1)))
        else:
            data=np.moveaxis(data,0,2)[:,::-1]
    
    
    vol = io.Volume(data, **geom)
    io.write(filemha_adjusted,vol)
    return

@ma.machine()
@ma.parameter("folder", str, default=None, description="Folder containing the .mha")
@ma.parameter("key", str, default=None, description="Substring for matching volumes")
@ma.parameter("suffix",str,default="_corrected_offset.nii")
@ma.parameter("spacing",[float, float, float],default=[2,2,2],description="Target spacing")
@ma.parameter("overlap",[float, float, float],default=[0,0,0],description="Overlap between regions")
def concatenateVolumes(folder,key,spacing,suffix,overlap):
    print(folder+"/*{}{}".format(key,suffix))
    files_list=glob.glob(folder+"/*{}{}".format(key,suffix))
    print(files_list)
    image_list=[io.read(f) for f in files_list]
    whole=concatenate_images(image_list,spacing=tuple(spacing),overlap=tuple(overlap))
    io.write(folder+"/whole_{}{}".format(key,suffix),whole)
    return






@ma.machine()
@ma.parameter("filename", str, default=None, description="Siemens K-space data .dat file")
@ma.parameter("dens_adj", bool, default=True, description="Radial density adjustment")
@ma.parameter("save", bool, default=False, description="save intermediary npy file")
@ma.parameter("suffix",str,default="")
@ma.parameter("select_first_rep", bool, default=False, description="Select the first central partition repetition")
@ma.parameter("index", int, default=-1, description="Header index")
@ma.parameter("nb_rep", int, default=None, description="nb rep selection for kushball undersampling simulation")
def build_kdata(filename,suffix,dens_adj,nb_rep,select_first_rep,index,save):

    if dens_adj:
        filename_kdata = str.split(filename, ".dat")[0] + suffix + "_kdata.npy"
    #filename_kdata_no_densadj = str.split(filename, ".dat")[0] + suffix + "_no_densadj_kdata.npy"
    else:
         filename_kdata =str.split(filename, ".dat")[0] + suffix + "_no_densadj_kdata.npy"
    filename_save = str.split(filename, ".dat")[0] + "{}.npy".format(suffix)
    filename_nav_save = str.split(filename, ".dat")[0] + "{}_nav.npy".format(suffix)
    filename_seqParams = str.split(filename, ".dat")[0] + "_seqParams.pkl"

    folder = "/".join(str.split(filename, "/")[:-1])


    if str.split(filename_seqParams, "/")[-1] not in os.listdir(folder):

       
        dico_seqParams = build_dico_seqParams(filename,index=index)

        

        file = open(filename_seqParams, "wb")
        pickle.dump(dico_seqParams, file)
        file.close()

    else:
        file = open(filename_seqParams, "rb")
        dico_seqParams = pickle.load(file)
        file.close()

    print(dico_seqParams)

    use_navigator_dll = dico_seqParams["use_navigator_dll"]

    if "use_kushball_dll" in dico_seqParams.keys():
        use_kushball_dll=dico_seqParams["use_kushball_dll"]
    else:
        use_kushball_dll=False

    nb_segments = dico_seqParams["alFree"][4]


    if use_kushball_dll:
        meas_sampling_mode = dico_seqParams["alFree"][16]
    elif use_navigator_dll:
        meas_sampling_mode = dico_seqParams["alFree"][15]
    else:
        meas_sampling_mode = dico_seqParams["alFree"][12]

    if use_navigator_dll:
        
        nb_gating_spokes = dico_seqParams["alFree"][6]
        if not(nb_gating_spokes==0) and (int(nb_segments/nb_gating_spokes)<(nb_segments/nb_gating_spokes)):
            print("Nb segments not divisible by nb_gating_spokes - adjusting nb_gating_spokes")
            nb_gating_spokes+=1
    else:
        nb_gating_spokes = 0

    
    undersampling_factor=dico_seqParams["alFree"][9]

    
    nb_part = dico_seqParams["nb_part"]
    nb_slices=int(nb_part)
    dummy_echos = dico_seqParams["alFree"][5]
    nb_rep_center_part = dico_seqParams["alFree"][11]

    if ("Spherical" in dico_seqParams)and(dico_seqParams["Spherical"]):
        nb_part = dico_seqParams["alFree"][12]
    else:
        nb_part = math.ceil(nb_part / undersampling_factor)

    nb_part_center=int(nb_part/2)
    nb_part = nb_part + nb_rep_center_part - 1
    del dico_seqParams

    if meas_sampling_mode==1:
        incoherent=False
    elif meas_sampling_mode==2:
        incoherent = True
    elif meas_sampling_mode==3:
        incoherent = True
    elif meas_sampling_mode==4:
        incoherent=True


    if incoherent:
        print("Non Stack-Of-Stars acquisition - 2Dplus1 reconstruction should not be used")

    if str.split(filename_save, "/")[-1] not in os.listdir(folder):
        if 'twix' not in locals():
            print("Re-loading raw data")
            twix = twixtools.read_twix(filename)

        mdb_list = twix[-1]['mdb']
        if nb_gating_spokes == 0:
            data = []

            for i, mdb in enumerate(mdb_list):
                if mdb.is_image_scan():
                    data.append(mdb)

        else:
            print("Reading Navigator Data....")
            data_for_nav = []
            data = []
            # k = 0
            for i, mdb in enumerate(mdb_list):
                if mdb.is_image_scan():
                    if not (mdb.mdh.Counter.Ida):
                        mdb_data_shape = mdb.data.shape
                        mdb_dtype = mdb.data.dtype
                        break

            for i, mdb in enumerate(mdb_list):
                if mdb.is_image_scan():
                    if not (mdb.mdh.Counter.Ida):
                        data.append(mdb)
                    else:
                        data_for_nav.append(mdb)
                        data.append(np.zeros(mdb_data_shape, dtype=mdb_dtype))
                    # print("i : {} / k : {} / Line : {} / Part : {}".format(i, k, mdb.cLin, mdb.cPar))
                    # k += 1
            data_for_nav = np.array([mdb.data for mdb in data_for_nav])
            data_for_nav = data_for_nav.reshape(
                (int(nb_part + dummy_echos), int(nb_gating_spokes)) + data_for_nav.shape[1:])

            if data_for_nav.ndim == 3:
                data_for_nav = np.expand_dims(data_for_nav, axis=-2)
            data_for_nav = data_for_nav[dummy_echos:]
            data_for_nav = np.moveaxis(data_for_nav, -2, 0)
            
            
            if select_first_rep:
                data_for_nav_select_first=np.zeros((data_for_nav.shape[0],nb_part-nb_rep_center_part+1,int(nb_gating_spokes),data_for_nav.shape[-1]),dtype=data_for_nav.dtype)
                data_for_nav_select_first[:,:(nb_part_center+1)]=data_for_nav[:,:(nb_part_center+1)]
                data_for_nav_select_first[:,(nb_part_center+1):]=data_for_nav[:,(nb_part_center+nb_rep_center_part):]
                data_for_nav=data_for_nav_select_first

            if nb_rep is not None:
                data_for_nav=data_for_nav[:,:nb_rep]


            np.save(filename_nav_save, data_for_nav)

        data = np.array([mdb.data for mdb in data])
        data = data.reshape((-1, int(nb_segments)) + data.shape[1:])
        data = data[dummy_echos:]
        data = np.moveaxis(data, 2, 0)
        data = np.moveaxis(data, 2, 1)

        if (undersampling_factor > 1)and(not(incoherent)):
            print("Filling kdata for undersampling {}".format(undersampling_factor))
            data_zero_filled=np.zeros((data.shape[0],int(nb_segments),nb_slices,data.shape[-1]),dtype=data.dtype)
            data_zero_filled_shape=data_zero_filled.shape
            data_zero_filled=data_zero_filled.reshape(data.shape[0],-1,8,nb_slices,data.shape[-1])
            data = data.reshape(data.shape[0], -1, 8, nb_part, data.shape[-1])

            curr_start=0

            for sl in range(nb_slices):
                data_zero_filled[:,curr_start::undersampling_factor, :,sl,:] = data[:,curr_start::undersampling_factor, :,int(sl/undersampling_factor),:]
                curr_start = curr_start + 1
                curr_start = curr_start % undersampling_factor

            data=data_zero_filled.reshape(data_zero_filled_shape)

            filename_us_weights = str.split(filename, ".dat")[0] + "_us_weights.npy"
            us_weights=np.zeros((1,int(nb_segments),nb_slices,1))
            us_weights_shape = us_weights.shape
            us_weights = us_weights.reshape(1, -1, 8, nb_slices, 1)
            curr_start = 0
            for sl in range(nb_slices):
                us_weights[:,curr_start::undersampling_factor, :,sl,:] = 1
                curr_start = curr_start + 1
                curr_start = curr_start % undersampling_factor

            us_weights=us_weights.reshape(us_weights_shape)
            np.save(filename_us_weights,us_weights)



        if select_first_rep:
                data_select_first=np.zeros((data.shape[0],int(nb_segments),nb_part-nb_rep_center_part+1,data.shape[-1]),dtype=data.dtype)
                data_select_first[:,:,:(nb_part_center+1)]=data[:,:,:(nb_part_center+1)]
                data_select_first[:,:,(nb_part_center+1):]=data[:,:,(nb_part_center+nb_rep_center_part):]
                data=data_select_first


        if nb_rep is not None:
            data=data[:,:,:nb_rep]
        


        del mdb_list

        ##################################################
        try:
            del twix
        except:
            pass
        
        if save:
            np.save(filename_save, data)

    else:
        data = np.load(filename_save)
        if nb_gating_spokes > 0:
            data_for_nav = np.load(filename_nav_save)

    npoint = data.shape[-1]
    #image_size = (nb_slices, int(npoint / 2), int(npoint / 2))

    print(data.shape)
    if nb_gating_spokes>0:
        print(data_for_nav.shape)

    if dens_adj:
        print("Performing Density Adjustment....")
        density = np.abs(np.linspace(-1, 1, npoint))
        density = np.expand_dims(density, tuple(range(data.ndim - 1)))

        if meas_sampling_mode==4:
            # phi1 = 0.4656
            # phi = np.arccos(np.mod(np.arange(nb_segments * nb_part) * phi1, 1))
            data *= density**2 #*np.sin(phi.reshape(nb_part, nb_segments).T[None, :, :, None])
        else:
            
            data *= density
    np.save(filename_kdata, data)

    return


@ma.machine()
@ma.parameter("filename_kdata", str, default=None, description="Saved K-space data .npy file")
@ma.parameter("filename_traj", str, default=None, description="Saved traj data .npy file (useful for traj not covered by Trajectory object e.g. Grappa rebuilt data")
@ma.parameter("sampling_mode", ["stack","incoherent_old","incoherent_new"], default="stack", description="Radial sampling strategy over partitions")
@ma.parameter("undersampling_factor", int, default=1, description="Kz undersampling factor")
@ma.parameter("dens_adj", bool, default=False, description="Memory usage")
@ma.parameter("suffix",str,default="")
@ma.parameter("nb_rep_center_part", int, default=1, description="Center partition repetitions")
def build_coil_sensi(filename_kdata,filename_traj,sampling_mode,undersampling_factor,dens_adj,suffix,nb_rep_center_part):

    kdata_all_channels_all_slices = np.load(filename_kdata)
    filename_b1 = ("_b1"+suffix).join(str.split(filename_kdata, "_kdata"))

    filename_seqParams =filename_kdata.split("_kdata.npy")[0] + "_seqParams.pkl"

    file = open(filename_seqParams, "rb")
    dico_seqParams = pickle.load(file)
    file.close()

    use_navigator_dll = dico_seqParams["use_navigator_dll"]

    if use_navigator_dll:
        meas_sampling_mode = dico_seqParams["alFree"][15]
    else:
        meas_sampling_mode = dico_seqParams["alFree"][12]

    undersampling_factor=dico_seqParams["alFree"][9]

    nb_slices = int(dico_seqParams["nb_part"])

    if meas_sampling_mode==1:
        incoherent=False
        mode = None
    elif meas_sampling_mode==2:
        incoherent = True
        mode = "old"
    elif meas_sampling_mode==3:
        incoherent = True
        mode = "new"


    if nb_rep_center_part>1:
        kdata_all_channels_all_slices=kdata_aggregate_center_part(kdata_all_channels_all_slices,nb_rep_center_part)

    # if sampling_mode_list[0]=="stack":
    #     incoherent=False
    # else:
    #     incoherent=True

    # if len(sampling_mode_list)>1:
    #     mode=sampling_mode_list[1]
    # else:
    #     mode="old"

    data_shape = kdata_all_channels_all_slices.shape
    nb_allspokes = data_shape[1]
    npoint = data_shape[-1]
    #image_size = (nb_slices, int(npoint / 2), int(npoint / 2))

    if filename_traj is None:
        radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=undersampling_factor, npoint=npoint,
                               nb_slices=nb_slices, incoherent=incoherent, mode=mode)
    else:
        curr_traj_completed_all_ts = np.load(filename_traj)
        radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=1, npoint=npoint,
                               nb_slices=nb_slices, incoherent=incoherent, mode=mode)
        radial_traj.traj = curr_traj_completed_all_ts

    res = 16
    image_size = (nb_slices, int(npoint / 2), int(npoint / 2))
    b1_all_slices = calculate_sensitivity_map_3D(kdata_all_channels_all_slices, radial_traj, res, image_size,
                                                 useGPU=False, light_memory_usage=True,density_adj=dens_adj,hanning_filter=True)

    image_file=str.split(filename_b1, ".npy")[0] + suffix + ".jpg"

    sl = int(b1_all_slices.shape[1]/2)

    list_images=list(np.abs(b1_all_slices[:,sl,:,:]))
    plot_image_grid(list_images,(6,6),title="Sensivitiy map for slice".format(sl),save_file=image_file)

    np.save(filename_b1, b1_all_slices)

    return


@ma.machine()
@ma.parameter("filename_kdata", str, default=None, description="Saved K-space data .npy file")
@ma.parameter("filename_traj", str, default=None, description="Saved traj data .npy file (useful for traj not covered by Trajectory object e.g. Grappa rebuilt data")
@ma.parameter("sampling_mode", ["stack","incoherent_old","incoherent_new"], default="stack", description="Radial sampling strategy over partitions")
@ma.parameter("undersampling_factor", int, default=1, description="Kz undersampling factor")
@ma.parameter("ntimesteps", int, default=175, description="Number of timesteps for the image serie")
@ma.parameter("use_GPU", bool, default=True, description="Use GPU")
@ma.parameter("light_mem", bool, default=True, description="Memory usage")
@ma.parameter("dens_adj", bool, default=False, description="Density adjustment for radial data")
@ma.parameter("suffix",str,default="")
def build_volumes(filename_kdata,filename_traj,sampling_mode,undersampling_factor,ntimesteps,use_GPU,light_mem,dens_adj,suffix):

    kdata_all_channels_all_slices = np.load(filename_kdata)
    filename_b1 = ("_b1" + suffix).join(str.split(filename_kdata, "_kdata"))

    b1_all_slices=np.load(filename_b1)
    filename_volume = ("_volumes"+suffix).join(str.split(filename_kdata, "_kdata"))
    print(filename_volume)
    sampling_mode_list = str.split(sampling_mode, "_")

    if sampling_mode_list[0] == "stack":
        incoherent = False
    else:
        incoherent = True

    if len(sampling_mode_list) > 1:
        mode = sampling_mode_list[1]
    else:
        mode = "old"

    data_shape = kdata_all_channels_all_slices.shape
    nb_allspokes = data_shape[1]
    npoint = data_shape[-1]
    nb_slices = data_shape[2]*undersampling_factor
    image_size = (nb_slices, int(npoint / 2), int(npoint / 2))

    if filename_traj is None:
        radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=undersampling_factor, npoint=npoint,
                               nb_slices=nb_slices, incoherent=incoherent, mode=mode)
    else:
        curr_traj_completed_all_ts=np.load(filename_traj)
        radial_traj=Radial3D(total_nspokes=nb_allspokes, undersampling_factor=1, npoint=npoint,
                                  nb_slices=nb_slices, incoherent=incoherent, mode=mode)
        radial_traj.traj = curr_traj_completed_all_ts

    volumes_all = simulate_radial_undersampled_images_multi(kdata_all_channels_all_slices, radial_traj, image_size,
                                                            b1=b1_all_slices, density_adj=dens_adj, ntimesteps=ntimesteps,
                                                            useGPU=use_GPU, normalize_kdata=False, memmap_file=None,
                                                            light_memory_usage=light_mem,normalize_volumes=True,normalize_iterative=True)
    np.save(filename_volume, volumes_all)


    gif=[]
    sl=int(volumes_all.shape[1]/2)
    volume_for_gif = np.abs(volumes_all[:,sl,:,:])
    for i in range(volume_for_gif.shape[0]):
        img = Image.fromarray(np.uint8(volume_for_gif[i]/np.max(volume_for_gif[i])*255), 'L')
        img=img.convert("P")
        gif.append(img)

    filename_gif = str.split(filename_volume,".npy") [0]+".gif"
    gif[0].save(filename_gif,save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)

    return




@ma.machine()
@ma.parameter("file_deformation", str, default=None, description="Deformation map file")
@ma.parameter("gr", int, default=4, description="Motion state")
@ma.parameter("sl", int, default=None, description="Slice")
def plot_deformation(file_deformation,gr,sl):
    deformation_map=np.load(file_deformation)[:,gr,sl]
    file_deformation_plot=str.split(file_deformation,".npy")[0]+"_gr{}sl{}.jpg".format(gr,sl)
    plot_deformation_map(deformation_map,save_file=file_deformation_plot)


@ma.machine()
@ma.parameter("filename_kdata", str, default=None, description="Saved K-space data .npy file")
@ma.parameter("filename_traj", str, default=None, description="Saved traj data .npy file (useful for traj not covered by Trajectory object e.g. Grappa rebuilt data")
@ma.parameter("sampling_mode", ["stack","incoherent_old","incoherent_new"], default="stack", description="Radial sampling strategy over partitions")
@ma.parameter("undersampling_factor", int, default=1, description="Kz undersampling factor")
@ma.parameter("dens_adj", bool, default=False, description="Memory usage")
@ma.parameter("threshold", float, default=None, description="Threshold for mask")
@ma.parameter("suffix",str,default="")
@ma.parameter("nb_rep_center_part", int, default=1, description="Center partition repetitions")
def build_mask(filename_kdata,filename_traj,sampling_mode,undersampling_factor,dens_adj,threshold,suffix,nb_rep_center_part):
    kdata_all_channels_all_slices = np.load(filename_kdata)

    if nb_rep_center_part>1:
        kdata_all_channels_all_slices=kdata_aggregate_center_part(kdata_all_channels_all_slices,nb_rep_center_part)


    filename_b1 = ("_b1" + suffix).join(str.split(filename_kdata, "_kdata"))

    b1_all_slices=np.load(filename_b1)

    filename_mask =("_mask"+suffix).join(str.split(filename_kdata, "_kdata"))
    print(filename_mask)

    sampling_mode_list = str.split(sampling_mode, "_")

    if sampling_mode_list[0] == "stack":
        incoherent = False
    else:
        incoherent = True

    if len(sampling_mode_list) > 1:
        mode = sampling_mode_list[1]
    else:
        mode = "old"

    data_shape = kdata_all_channels_all_slices.shape
    nb_allspokes = data_shape[1]
    npoint = data_shape[-1]
    nb_slices = data_shape[2]*undersampling_factor
    image_size = (nb_slices, int(npoint / 2), int(npoint / 2))

    if filename_traj is None:
        radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=undersampling_factor, npoint=npoint,
                               nb_slices=nb_slices, incoherent=incoherent, mode=mode)
    else:
        curr_traj_completed_all_ts = np.load(filename_traj)
        radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=1, npoint=npoint,
                               nb_slices=nb_slices, incoherent=incoherent, mode=mode)
        radial_traj.traj = curr_traj_completed_all_ts

    selected_spokes=np.r_[10:400] 
    selected_spokes=None 
    mask = build_mask_single_image_multichannel(kdata_all_channels_all_slices, radial_traj, image_size,
                                                b1=b1_all_slices, density_adj=dens_adj, threshold_factor=threshold,
                                                normalize_kdata=False, light_memory_usage=True,selected_spokes=selected_spokes,normalize_volumes=True)


    np.save(filename_mask, mask)

    gif = []

    for i in range(mask.shape[0]):
        img = Image.fromarray(np.uint8(mask[i] / np.max(mask[i]) * 255), 'L')
        img = img.convert("P")
        gif.append(img)

    filename_gif = str.split(filename_mask, ".npy")[0] + ".gif"
    gif[0].save(filename_gif, save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)

    return


@ma.machine()
@ma.parameter("filename_volume", str, default=None, description="Singular volumes")
@ma.parameter("l", int, default=0, description="Singular volume number for mask calculation")
@ma.parameter("threshold", float, default=None, description="Threshold for mask")
@ma.parameter("it", int, default=3, description="Binary closing iterations")
def build_mask_from_singular_volume(filename_volume,l,threshold,it):
    filename_mask="".join(filename_volume.split(".npy"))+"_l{}_mask.npy".format(l)
    volumes=np.load(filename_volume)
    print(volumes.shape)
    if volumes.ndim==4:
        volume=volumes[l]
        mask=build_mask_from_volume(volume,threshold,it)
    elif volumes.ndim==5:
        print("Aggregating mask from all respiratory motions")
        volume_allbins=volumes[:,l]
        nb_bins=volume_allbins.shape[0]
        mask=False

        for gr in range(nb_bins):
            volume=volume_allbins[gr]
            current_mask=build_mask_from_volume(volume,threshold,it)
            mask=mask|current_mask



    elif volumes.ndim==3:
        print("Singular volume number l not used - input dim was 3")
        filename_mask="".join(filename_volume.split(".npy"))+"_mask.npy"
        mask=build_mask_from_volume(volume,threshold,it)
    else:
        raise ValueError("Volume number of dimensions should be 3 or 4 or 5")

    np.save(filename_mask,mask)

    gif = []

    for i in range(mask.shape[0]):
        img = Image.fromarray(np.uint8(mask[i] / np.max(mask[i]) * 255), 'L')
        img = img.convert("P")
        gif.append(img)

    filename_gif = str.split(filename_mask, ".npy")[0] + ".gif"
    gif[0].save(filename_gif, save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)

    return


@ma.machine()
@ma.parameter("filename_mask", str, default=None, description="Mask")
def build_mask_full_from_mask(filename_mask):
    filename_mask_full=filename_mask.split("_mask.npy")[0]+"_mask_full.npy"
    mask=np.load(filename_mask)

    new_mask=np.zeros_like(mask)
    # for sl in range(mask.shape[0]):
    #     x_min=np.min(np.argwhere(mask[sl]>0)[:,0])
    #     y_min=np.min(np.argwhere(mask[sl]>0)[:,1])

    #     x_max=np.max(np.argwhere(mask[sl]>0)[:,0])
    #     y_max=np.max(np.argwhere(mask[sl]>0)[:,1])

    #     new_mask[sl,x_min:(x_max+1),y_min:(y_max+1)]=1

    x_min=np.min(np.argwhere(mask>0)[:,1])
    y_min=np.min(np.argwhere(mask>0)[:,2])

    x_max=np.max(np.argwhere(mask>0)[:,1])
    y_max=np.max(np.argwhere(mask>0)[:,2])

    new_mask[:,x_min:(x_max+1),y_min:(y_max+1)]=1
        
    np.save(filename_mask_full,new_mask)

    gif = []
    for i in range(new_mask.shape[0]):
        img = Image.fromarray(np.uint8(new_mask[i] / np.max(new_mask[i]) * 255), 'L')
        img = img.convert("P")
        gif.append(img)

    filename_gif = str.split(filename_mask_full, ".npy")[0] + ".gif"
    gif[0].save(filename_gif, save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)

    return


@ma.machine()
@ma.parameter("filename_volume", str, default=None, description="MRF time series")
@ma.parameter("filename_mask", str, default=None, description="Mask")
@ma.parameter("dico_full_file", str, default=None, description="Dictionary file")
@ma.parameter("optimizer_config",type=ma.Config(),default=DEFAULT_OPT_CONFIG_2STEPS,description="Optimizer parameters")
@ma.parameter("slices",str,default=None,description="Slices to consider for pattern matching")
@ma.parameter("L0",int,default=6,description="Number of singular volumes")
@ma.parameter("force_pca",bool,default=False,description="Force recalculation of the PCA on the dictionary")
@ma.parameter("config_clustering",str,default=None,description=".json file with clustering windows for 2 steps matching")
def build_maps(filename_volume,filename_mask,dico_full_file,optimizer_config,slices,config_clustering,L0,force_pca):
    '''
    builds MRF maps using bi-component dictionary matching (Slioussarenko et al. MRM 2024)
    inputs:
    filename_volume - .npy file containing time serie of undersampled volumes size ntimesteps x nb_slices x npoint/2 x npoint/2 (numpy array)
    filename_mask - .npy file containing mask of size nb_slices x npoint/2 x npoint/2 (numpy array)
    dico_full_file - light and full dictionaries with headers (.pkl)
    optimizer_config - optimization options
    slices - list of slices to consider for pattern matching (e.g. "0,1,2,3")
    volumes_type - "raw" or "singular" - depending on the input volumes ("raw" time serie of undersampled volumes / "singular" singular volumes)
    

    outputs:
    all_maps: tuple containing for all iterations 
            (maps - dictionary with parameter maps for all keys
             mask - numpy array
             cost map (OPTIONAL)
             phase map - numpy array (OPTIONAL)
             proton density map - numpy array (OPTIONAL)
             matched_signals - numpy array  (OPTIONAL))

    '''

    opt_type = optimizer_config["type"]
    print(opt_type)
    useGPU = optimizer_config["useGPU"]
    threshold_pca=optimizer_config["pca"]
    split=optimizer_config["split"]
    volumes_type=optimizer_config["volumes_type"]
    return_cost = optimizer_config["return_cost"]
    clustering_windows=config_clustering

    try:
        import cupy
    except:
        print("Could not import cupy - not using gpu")
        useGPU=False
    
    
    if volumes_type=="singular":
        threshold_pca=float(L0)
        
            

    file_map = "".join(filename_volume.split(".npy")) + "_{}_MRF_map.pkl".format(opt_type)
    volumes_all_slices = np.load(filename_volume)
    masks_all_slices=np.load(filename_mask)

    if slices is not None:
        sl = np.array(slices.split(",")).astype(int)
        if not(len(sl)==0):
            mask_slice = np.zeros(mask.shape, dtype=mask.dtype)
            mask_slice[sl] = 1
            mask *= mask_slice
            sl=[str(s) for s in sl]
            file_map = "".join(filename_volume.split(".npy")) + "_sl{}_{}_MRF_map.pkl".format("_".join(sl),opt_type)


    

    optimizer = SimpleDictSearch(mask=masks_all_slices, split=split, pca=True,
                                                threshold_pca=threshold_pca,threshold_ff=1.1,return_cost=return_cost,useGPU_dictsearch=useGPU,volumes_type=volumes_type,clustering_windows=clustering_windows,force_pca=force_pca)
                
    all_maps=optimizer.search_patterns_test_multi_2_steps_dico(dico_full_file,volumes_all_slices)

    save_pickle(file_map,all_maps)
    
    return


@ma.machine()
@ma.parameter("filename_map", str, default=None, description="Maps .pkl")
@ma.parameter("filename_seqParams", str, default=None, description="Seq Params .pkl")
@ma.parameter("params", str, default=None, description="Parameters to output (e.g. 'ff_wT1_df_att' )")
def generate_image_maps(filename_map,filename_seqParams,params):

    if params is None:
        params="ff_wT1_df_att"

    params=str.split(params,"_")
    print(params)

    print(filename_seqParams)
    print(filename_map)
    with open(filename_seqParams,"rb") as file:
        dico=pickle.load(file)
    
    dx,dy,dz=dico["spacing"]

    print("dx,dy,dz : ",dx,dy,dz)
    with open(filename_map,"rb") as filemap:
        all_maps=pickle.load(filemap)

    print(list(all_maps[0][0].keys()))
    for iter in list(all_maps.keys()):

        map_rebuilt=all_maps[iter][0]
        mask=all_maps[iter][1]

        map_rebuilt["wT1"][map_rebuilt["ff"] > 0.7] = 0.0

        keys_simu = list(map_rebuilt.keys())
        values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
        map_for_sim = dict(zip(keys_simu, values_simu))

        #map_Python = MapFromDict3D("RebuiltMapFromParams_iter{}".format(iter), paramMap=map_for_sim)
        #map_Python.buildParamMap()


        for key in params:
            file_mha = "/".join(["/".join(str.split(filename_map,"/")[:-1]),"_".join(str.split(str.split(filename_map,"/")[-1],".")[:-1])]) + "_it{}_{}.mha".format(iter,key)
            io.write(file_mha,map_for_sim[key],tags={"spacing":[dz,dx,dy]})
    return

@ma.machine()
@ma.parameter("file_map",str,default=None,description="map file (.pkl)")
@ma.parameter("file_ref",str,default=None,description="reference file for spacing (.mha)")
def build_additional_maps(file_map,file_ref):

    curr_file=file_map
    file = open(curr_file, "rb")
    all_maps = pickle.load(file)
    file.close()

    print(all_maps.keys)

    ref_map=io.read(file_ref)
    spacing=ref_map.spacing

    mask=all_maps[0][1]
    map_norm=makevol(all_maps[0][4], mask > 0)
    map_phase=makevol(all_maps[0][3], mask > 0)
    map_J=makevol(all_maps[0][2], mask > 0)
    

    file_mha_norm = "/".join(["/".join(str.split(curr_file,"/")[:-1]),"_".join(str.split(str.split(curr_file,"/")[-1],".")[:-1])]) + "_it0_norm.mha"
    io.write(file_mha_norm,map_norm,tags={"spacing":spacing})

    file_mha_phase = "/".join(["/".join(str.split(curr_file,"/")[:-1]),"_".join(str.split(str.split(curr_file,"/")[-1],".")[:-1])]) + "_it0_phase.mha"
    io.write(file_mha_phase,map_phase,tags={"spacing":spacing})

    file_mha_J = "/".join(["/".join(str.split(curr_file,"/")[:-1]),"_".join(str.split(str.split(curr_file,"/")[-1],".")[:-1])]) + "_it0_J.mha"
    io.write(file_mha_J,map_J,tags={"spacing":spacing})

    return





# @ma.machine()
# @ma.parameter("file_map",str,default=None,description="map file (.pkl)")
# @ma.parameter("config_image_maps",type=ma.Config(),default=None,description="Image Config")
# @ma.parameter("suffix", str, default="", description="suffix")
# def generate_image_maps(file_map,config_image_maps,suffix):
#     return_cost=config_image_maps["return_cost"]
#     return_matched_signals=config_image_maps["return_matched_signals"]
#     #keys=list(all_maps.keys())
#     keys=config_image_maps["keys"]
#     list_l=config_image_maps["singular_volumes_outputted"]

#     print(keys)

#     distances=config_image_maps["image_distances"]
#     dx=distances[0]
#     dy=distances[1]
#     dz=distances[2]

#     #file_map = filename.split(".dat")[0] + "{}_MRF_map.pkl".format("")
#     curr_file=file_map
#     file = open(curr_file, "rb")
#     all_maps = pickle.load(file)
#     file.close()

#     if not(keys):
#         keys=list(all_maps.keys())

#     for iter in keys:

#         map_rebuilt=all_maps[iter][0]
#         mask=all_maps[iter][1]

#         map_rebuilt["wT1"][map_rebuilt["ff"] > 0.7] = 0.0

#         keys_simu = list(map_rebuilt.keys())
#         values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
#         map_for_sim = dict(zip(keys_simu, values_simu))

#         #map_Python = MapFromDict3D("RebuiltMapFromParams_iter{}".format(iter), paramMap=map_for_sim)
#         #map_Python.buildParamMap()


#         for key in ["ff","wT1","df","att"]:
#             file_mha = "/".join(["/".join(str.split(curr_file,"/")[:-1]),"_".join(str.split(str.split(curr_file,"/")[-1],".")[:-1])]) + "{}_it{}_{}.mha".format(suffix,iter,key)
#             if file_mha.startswith("/"):
#                 file_mha=file_mha[1:]
#             io.write(file_mha,map_for_sim[key],tags={"spacing":[dz,dx,dy]})



#         if return_matched_signals:
#             for l in list_l:
#                 matched_volumes=makevol(all_maps[iter][-1][l],mask>0)
#                 #print(matched_volumes)
#                 file_mha = "/".join(["/".join(str.split(curr_file, "/")[:-1]),
#                                      "_".join(str.split(str.split(curr_file, "/")[-1], ".")[:-1])]) + "{}_it{}_l{}_{}.mha".format(suffix,
#                                                                                                                                   iter,l, "matchedvolumes")
#                 if file_mha.startswith("/"):
#                     file_mha=file_mha[1:]
#                 io.write(file_mha, np.abs(matched_volumes), tags={"spacing": [dz, dx, dy]})


#         if return_cost:
#             try:
#                 file_mha = "/".join(["/".join(str.split(curr_file, "/")[:-1]),
#                                      "_".join(str.split(str.split(curr_file, "/")[-1], ".")[:-1])]) + "{}_it{}_{}.mha".format(suffix,
#                                                                                                                               iter, "correlation")
#                 if file_mha.startswith("/"):
#                     file_mha=file_mha[1:]
#                 io.write(file_mha, makevol(all_maps[iter][2],mask>0), tags={"spacing": [dz, dx, dy]})

#                 file_mha = "/".join(["/".join(str.split(curr_file, "/")[:-1]),
#                                      "_".join(str.split(str.split(curr_file, "/")[-1], ".")[:-1])]) + "{}_it{}_{}.mha".format(suffix,
#                                                                                                                               iter, "phase")
#                 if file_mha.startswith("/"):
#                     file_mha=file_mha[1:]
#                 io.write(file_mha, makevol(all_maps[iter][3],mask>0), tags={"spacing": [dz, dx, dy]})
#             except:
#                 continue
#     return


@ma.machine()
@ma.parameter("file_volume",str,default=None,description="volume file (.mha)")
@ma.parameter("l", int, default=0, description="Singular volume")
def extract_singular_volume_allbins(file_volume,l):
    file_volume_target=str.replace(file_volume,"volumes_singular","volume_singular_l{}".format(l))
    volumes=np.load(file_volume)
    print(volumes.dtype)
    np.save(file_volume_target,volumes[:,l])
    return


@ma.machine()
@ma.parameter("file_volume",str,default=None,description="volume file (.mha)")
@ma.parameter("gr", int, default=0, description="Motion bin state")
def extract_allsingular_volumes_bin(file_volume,gr):
    file_volume_target=str.replace(file_volume,"volumes_singular_allbins","volumes_singular_gr{}".format(gr))
    volumes=np.load(file_volume)
    print(volumes.dtype)
    np.save(file_volume_target,volumes[gr,:])
    return

@ma.machine()
@ma.parameter("file_volume",str,default=None,description="volume file (.mha)")
@ma.parameter("nb_gr",int,default=4,description="number of respiratory bins")
@ma.parameter("sl", int, default=None, description="Slice")
@ma.parameter("x", int, default=None, description="x")
@ma.parameter("y", int, default=None, description="y")
@ma.parameter("slice_res_factor", int, default=5, description="Factor between slice thickness and in plane resolution")
@ma.parameter("l", int, default=0, description="Singular volume")
@ma.parameter("metric", ["abs","phase","real","imag"], default="abs", description="Metric to plot")
@ma.parameter("single_volume", bool, default=False, description="One single volume - No bin or singular volume")
def generate_movement_gif(file_volume,nb_gr,sl,x,y,l,metric,slice_res_factor,single_volume):
    if sl is not None:
        filename_gif = str.split(file_volume.format(nb_gr), ".mha")[0] + "_sl{}_moving_singular.gif".format(sl)
    elif x is not None:
        filename_gif = str.split(file_volume.format(nb_gr), ".mha")[0] + "_x{}_moving_singular.gif".format(x)
    elif y is not None:
        filename_gif = str.split(file_volume.format(nb_gr), ".mha")[0] + "_y{}_moving_singular.gif".format(y)
    elif single_volume:
        filename_gif = str.split(file_volume, ".npy")[0] + "_moving_slices.gif"
    
    if file_volume.find(".mha")>0:
        def load(file):
            return io.read(file)
    else:
        def load(file):
            return np.load(file)
    
    test_volume=load(file_volume.format(0)).squeeze()
    print(test_volume.shape)

    if single_volume:
        print("Single volume - The GIF will be a movie navigating along the slices")
        test_volume=np.expand_dims(test_volume,axis=1)
    print(test_volume.shape)
    if test_volume.ndim==3:# each file contains only one motion phase
        all_matched_volumes=[]
        #print(file_volume)
        for gr in np.arange(nb_gr):
            file_mha= file_volume.format(gr)
            matched_volume=load(file_mha)
            matched_volume=np.array(matched_volume)
            all_matched_volumes.append(matched_volume)

        all_matched_volumes=np.array(all_matched_volumes)

    elif test_volume.ndim==4:#file contains all phases for one singular volume
        all_matched_volumes=test_volume


    elif test_volume.ndim==5:#file contains all phases and all singular volumes
        all_matched_volumes=test_volume[:,l]
        filename_gif=filename_gif.replace("moving_singular.gif","moving_singular_l{}.gif".format(l))

    print(all_matched_volumes.shape)
    if sl is not None:
        moving_image=np.concatenate([all_matched_volumes[:,sl],all_matched_volumes[1:-1,sl][::-1]],axis=0)
    elif x is not None:
        moving_image=np.concatenate([all_matched_volumes[:,:,x],all_matched_volumes[1:-1,:,x][::-1]],axis=0)
    elif y is not None:
        all_matched_volumes=np.repeat(all_matched_volumes,slice_res_factor,axis=1)
        moving_image=np.concatenate([all_matched_volumes[:,:,:,y],all_matched_volumes[1:-1,:,:,y][::-1]],axis=0)
    elif single_volume:
        moving_image=np.concatenate([all_matched_volumes[:,0],all_matched_volumes[1:-1,0][::-1]],axis=0)
    animate_images(moving_image,interval=10)

    from PIL import Image
    gif=[]
    print(moving_image.shape)
    if metric=="abs":
        volume_for_gif = np.abs(moving_image)
    elif metric=="phase":
        volume_for_gif = np.angle(moving_image)
        filename_gif = str.replace(filename_gif,"moving_singular","moving_singular_phase")

    elif metric=="real":
        volume_for_gif = np.real(moving_image)
        filename_gif = str.replace(filename_gif,"moving_singular","moving_singular_real")

    elif metric=="imag":
        volume_for_gif = np.imag(moving_image)
        filename_gif = str.replace(filename_gif,"moving_singular","moving_singular_imag")

    else:
        raise ValueError("metric unknown - choose from abs/phase/real/imag")

    for i in range(volume_for_gif.shape[0]):
        min_value=np.min(volume_for_gif[i])
        max_value=np.max(volume_for_gif[i])
        img = Image.fromarray(np.uint8((volume_for_gif[i]-min_value)/(max_value-min_value)*255), 'L')
        img=img.convert("P")
        gif.append(img)


    gif[0].save(filename_gif,save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)
    print(filename_gif)
    return

@ma.machine()
@ma.parameter("file_map",str,default=None,description="map file (.pkl)")
@ma.parameter("config_image_maps",type=ma.Config(),default=None,description="Image Config")
@ma.parameter("suffix", str, default="", description="suffix")
def generate_matchedvolumes_allgroups(file_map,config_image_maps,suffix):
    curr_file=file_map
    file = open(curr_file, "rb")
    all_maps = pickle.load(file)
    file.close()

    distances=config_image_maps["image_distances"]
    dx=distances[0]
    dy=distances[1]
    dz=distances[2]


    keys=config_image_maps["keys"]

    matched_signals=all_maps[keys[0]][-1]
    nb_singular_images=matched_signals.shape[0]
    mask=all_maps[keys[0]][1]
    nb_signals=mask.sum()

    matched_signals=matched_signals.reshape(nb_singular_images,-1,nb_signals)
    nb_gr=matched_signals.shape[1]


    l_list=config_image_maps["singular_volumes_outputted"]


    for gr in range(nb_gr):
        for l in l_list:
            for iter in keys:
                matched_signals=all_maps[iter][-1]
                matched_signals=matched_signals.reshape(nb_singular_images,-1,nb_signals)
                matched_volumes=makevol(matched_signals[l][gr],mask>0)
                file_mha = "/".join(["/".join(str.split(curr_file, "/")[:-1]),
                                            "_".join(str.split(str.split(curr_file, "/")[-1], ".")[:-1])]) + "{}_gr{}_it{}_l{}_{}.mha".format(suffix,gr,
                            iter,l, "matchedvolumes")
                if file_mha.startswith("/"):
                    file_mha=file_mha[1:]
                io.write(file_mha, np.abs(matched_volumes), tags={"spacing": [dz, dx, dy]})



@ma.machine()
@ma.parameter("files_config",type=ma.Config(),default=None,description="Files to consider for aggregate kdata reconstruction")
@ma.parameter("disp_config",type=ma.Config(),default=None,description="Parameters for movement identification")
@ma.parameter("undersampling_factor", int, default=1, description="Kz undersampling factor")
def build_data_nacq(files_config,disp_config,undersampling_factor):

    base_folder = "./data/InVivo/3D"
    files=files_config["files"]
    shifts=list(range(disp_config["shifts"][0],disp_config["shifts"][1]))
    ch=disp_config["channel"]
    bin_width=disp_config["bin_width"]

    folder = base_folder + "/".join(str.split(files[0], "/")[:-1])


    filename_categories_global = folder + "/categories_global_bw{}.npy".format(bin_width)
    filename_df_groups_global = folder + "/df_groups_global_bw{}.pkl".format(bin_width)

    categories_global = []
    df_groups_global = pd.DataFrame()

    for localfile in files:

        filename = base_folder + localfile
        filename_nav_save = str.split(filename, ".dat")[0] + "_nav.npy"
        folder = "/".join(str.split(filename, "/")[:-1])
        filename_kdata = str.split(filename, ".dat")[0] + "_kdata{}.npy".format("")
        filename_disp_image=str.split(filename, ".dat")[0] + "_nav_image.jpg".format("")


        dico_seqParams = build_dico_seqParams(filename)

        use_navigator_dll = dico_seqParams["use_navigator_dll"]

        if use_navigator_dll:
            meas_sampling_mode = dico_seqParams["alFree"][14]
            nb_gating_spokes = dico_seqParams["alFree"][6]
        else:
            meas_sampling_mode = dico_seqParams["alFree"][12]
            nb_gating_spokes = 0


        nb_segments = dico_seqParams["alFree"][4]

        del dico_seqParams

        if meas_sampling_mode == 1:
            incoherent = False
            mode = None
        elif meas_sampling_mode == 2:
            incoherent = True
            mode = "old"
        elif meas_sampling_mode == 3:
            incoherent = True
            mode = "new"

        data, data_for_nav = build_data(filename, folder, nb_segments, nb_gating_spokes)

        data_shape = data.shape

        nb_allspokes = data_shape[-3]
        npoint = data_shape[-1]
        nb_slices = data_shape[-2]

        if str.split(filename_kdata, "/")[-1] in os.listdir(folder):
            del data

        if str.split(filename_kdata, "/")[-1] not in os.listdir(folder):
            # Density adjustment all slices

            density = np.abs(np.linspace(-1, 1, npoint))
            density = np.expand_dims(density, tuple(range(data.ndim - 1)))

            print("Performing Density Adjustment....")
            data *= density
            np.save(filename_kdata, data)
            del data



        print("Calculating Coil Sensitivity....")

        radial_traj = Radial3D(total_nspokes=nb_allspokes, undersampling_factor=undersampling_factor, npoint=npoint,
                               nb_slices=nb_slices, incoherent=incoherent, mode=mode)

        nb_segments = radial_traj.get_traj().shape[0]

        if nb_gating_spokes > 0:
            print("Processing Nav Data...")
            data_for_nav = np.load(filename_nav_save)

            nb_allspokes = nb_segments
            nb_slices = data_for_nav.shape[1]
            nb_channels = data_for_nav.shape[0]
            npoint_nav = data_for_nav.shape[-1]

            all_timesteps = np.arange(nb_allspokes)
            nav_timesteps = all_timesteps[::int(nb_allspokes / nb_gating_spokes)]

            nav_traj = Navigator3D(direction=[0, 0, 1], npoint=npoint_nav, nb_slices=nb_slices,
                                   applied_timesteps=list(nav_timesteps))

            nav_image_size = (int(npoint_nav),)

            print("Building nav image for channel {}...".format(ch))
            # b1_nav = calculate_sensitivity_map_3D_for_nav(data_for_nav, nav_traj, res=16, image_size=nav_image_size)
            # b1_nav_mean = np.mean(b1_nav, axis=(1, 2))

            image_nav_ch = simulate_nav_images_multi(np.expand_dims(data_for_nav[ch], axis=0), nav_traj, nav_image_size)

            plt.figure()
            plt.plot(np.abs(image_nav_ch.reshape(-1, int(npoint / 2)))[5*nb_gating_spokes, :])
            plt.savefig(filename_disp_image)

            print("Estimating Movement...")

            bottom = -shifts[0]
            top = nav_image_size[0]-shifts[-1]


            displacements = calculate_displacement(image_nav_ch, bottom, top, shifts, 0.001)

            displacement_for_binning = displacements

            max_bin = np.max(displacement_for_binning)
            min_bin = shifts[0]

            bins = np.arange(min_bin, max_bin + bin_width, bin_width)
            # print(bins)
            categories = np.digitize(displacement_for_binning, bins)
            df_cat = pd.DataFrame(data=np.array([displacement_for_binning, categories]).T,
                                  columns=["displacement", "cat"])
            df_groups = df_cat.groupby("cat").count()
            curr_max = df_groups.displacement.max()

            if df_groups_global.empty:
                df_groups_global = df_groups
            else:
                df_groups_global += df_groups

            categories_global.append(categories)

    #################################################################################################################################"

    categories_global = np.array(categories_global)

    np.save(filename_categories_global, categories_global)
    df_groups_global.to_pickle(filename_df_groups_global)



    return

@ma.machine()
@ma.parameter("filename_nav_save", str, default=None, description="Navigator data")
@ma.parameter("seasonal_adj", bool, default=False, description="Seasonal adjustement")
def build_navigator_images(filename_nav_save,seasonal_adj):
    filename_image_nav= filename_nav_save.split("_nav.npy")[0] + "_image_nav.npy"
    filename_image_nav_plot = filename_nav_save.split("_nav.npy")[0] + "_image_nav.jpg"
    filename_image_nav_diff_plot = filename_nav_save.split("_nav.npy")[0] + "_image_nav_diff.jpg"

    data_for_nav = np.load(filename_nav_save)
    nb_channels = data_for_nav.shape[0]
    npoint_nav = data_for_nav.shape[-1]
    nb_gating_spokes = data_for_nav.shape[-2]
    nb_slices = data_for_nav.shape[1]

    nav_traj = Navigator3D(direction=[0, 0, 1], npoint=npoint_nav, nb_slices=nb_slices,
                           nb_gating_spokes=nb_gating_spokes)
    nav_image_size = (int(npoint_nav ),)

    image_nav_all_channels = []

    # for j in tqdm(range(nb_channels)):
    for j in tqdm(range(nb_channels)):
        images_series_rebuilt_nav_ch = simulate_nav_images_multi(np.expand_dims(data_for_nav[j], axis=0), nav_traj,
                                                                 nav_image_size, b1=None)
        image_nav_ch = np.abs(images_series_rebuilt_nav_ch)
        image_nav_all_channels.append(image_nav_ch)

    image_nav_all_channels = np.array(image_nav_all_channels)
    if seasonal_adj:
        from statsmodels.tsa.seasonal import seasonal_decompose

        image_reshaped = image_nav_all_channels.reshape(-1, npoint_nav)
        decomposition = seasonal_decompose(image_reshaped,
                                           model='multiplicative', period=nb_gating_spokes)
        image=image_reshaped/decomposition.seasonal
        image=image.reshape(-1,nb_gating_spokes,npoint_nav)
        image_nav_all_channels=image
        print(image.shape)


    np.save(filename_image_nav,image_nav_all_channels)

    plot_image_grid(
        np.moveaxis(image_nav_all_channels.reshape(nb_channels, -1, int(npoint_nav)), -1, -2)[:, :, :100],
        nb_row_col=(6, 6),save_file=filename_image_nav_plot)

    plot_image_grid(
        np.moveaxis(np.diff(image_nav_all_channels.reshape(nb_channels, -1, int(npoint_nav)), axis=-1), -1, -2)[:,
        :, :100], nb_row_col=(6, 6),save_file=filename_image_nav_diff_plot)

    print("Navigator images plot file: {}".format(filename_image_nav_diff_plot))

    return


@ma.machine()
@ma.parameter("filename_nav_save", str, default=None, description="Navigator data")
@ma.parameter("filename_displacement", str, default=None, description="Displacement data")
@ma.parameter("nb_segments", int, default=1400, description="MRF Total Spoke number")
@ma.parameter("bottom", int, default=-30, description="Lowest displacement for displacement estimation")
@ma.parameter("top", int, default=30, description="Highest displacement for displacement estimation")
@ma.parameter("ntimesteps", int, default=175, description="Number of MRF images")
@ma.parameter("nspoke_per_z", int, default=8, description="number of spokes before partition jump when undersampling")
@ma.parameter("us", int, default=1, description="undersampling_factor")
@ma.parameter("incoherent", bool, default=True, description="3D sampling type")
@ma.parameter("lambda_tv", float, default=0.001, description="Temporal regularization for displacement estimation")
@ma.parameter("ch", int, default=None, description="channel if single channel estimation")
@ma.parameter("filename_bins", str, default=None, description="bins file if data is binned according to another scan")
@ma.parameter("filename_disp_respi", str, default=None, description="source displacement for distribution matching")
@ma.parameter("retained_categories", str, default=None, description="retained bins")
@ma.parameter("nbins", int, default=5, description="Number of motion states")
@ma.parameter("gating_only", bool, default=False, description="Weights for gating only and not for density compensation")
@ma.parameter("pad", int, default=10, description="Navigator images padding")
@ma.parameter("randomize", bool, default=False, description="Randomization for baseline navigator image for displacement calc")
@ma.parameter("equal_spoke_per_bin", bool, default=False, description="Distribute evenly the number of spokes per bin")
@ma.parameter("use_ml", bool, default=False, description="Use segment anything for motion estimation")
@ma.parameter("useGPU", bool, default=True, description="Use GPU")
@ma.parameter("force_recalc_disp", bool, default=True, description="Force calculation of displacement")
@ma.parameter("dct_frequency_filter", int, default=None, description="DCT filtering for displacement smoothing")
@ma.parameter("seasonal_adj", bool, default=False, description="Seasonal adjustement")
@ma.parameter("hard_interp", bool, default=False, description="Hard interpolation for inversion")
@ma.parameter("nb_rep_center_part", int, default=1, description="Central partition repetitions")
@ma.parameter("sim_us", int, default=1, description="Undersampling simulation")
@ma.parameter("us_file", str, default=None, description="Undersampling simulation from file ")
@ma.parameter("interp_bad_correl", bool, default=False, description="Interpolate displacements with neighbours when poorly correlated")
@ma.parameter("nav_res_factor", int, default=None, description="bins rescaling if resolution of binning navigator different from current input")
@ma.parameter("soft_weight", bool, default=False, description="use soft weight for full inspiration")
@ma.parameter("stddisp", float, default=None, description="outlier exclusion for irregular breathing")

def calculate_displacement_weights(filename_nav_save,filename_displacement,nb_segments,bottom,top,ntimesteps,us,incoherent,lambda_tv,ch,filename_bins,retained_categories,nbins,gating_only,pad,randomize,equal_spoke_per_bin,use_ml,useGPU,force_recalc_disp,dct_frequency_filter,seasonal_adj,hard_interp,nb_rep_center_part,sim_us,us_file,interp_bad_correl,nspoke_per_z,nav_res_factor,soft_weight,stddisp,filename_disp_respi):

    '''
    Displacement calculation from raw navigator K-space data
    Remark:
    If use_ml is True, bottom, top, randomize, lambda_TV are not useful
    '''

    print(seasonal_adj)

    if filename_nav_save is None:
        nav_file=False
    else:
        nav_file=True

    if filename_displacement is None:
        filename_displacement=filename_nav_save.split("_nav.npy")[0] + "_displacement.npy"
        filename_weights=filename_nav_save.split("_nav.npy")[0] + "_weights.npy"
        filename_retained_ts=filename_nav_save.split("_nav.npy")[0] + "_retained_ts.pkl"
        filename_bins_output = filename_nav_save.split("_nav.npy")[0] + "_bins.npy"

        folder = "/".join(str.split(filename_nav_save, "/")[:-1])

    else:
        print("Displacement file given")
        filename_weights = filename_displacement.split("_displacement")[0] + "_weights.npy"
        filename_retained_ts = filename_displacement.split("_displacement")[0] + "_retained_ts.pkl"
        filename_bins_output = filename_displacement.split("_displacement")[0] + "_bins.npy"

        folder = "/".join(str.split(filename_displacement, "/")[:-1])
        print(folder)

    if retained_categories is not None:
        retained_categories = np.array(retained_categories.split(",")).astype(int)

    if nav_file:
        data_for_nav=np.load(filename_nav_save)


    if filename_disp_respi is not None:
        disp_respi=np.load(filename_disp_respi)
    else:
        disp_respi=None


    if ((str.split(filename_displacement, "/")[-1] not in os.listdir(folder)) or (force_recalc_disp)):
        if use_ml:
            if useGPU:
                device="cuda"
            else:
                device="cpu"
            displacements=calculate_displacement_ml(data_for_nav,nb_segments,ch=ch,device=device)

        else:
            if ch is None:
                displacements=calculate_displacements_allchannels(data_for_nav,nb_segments,shifts = list(range(bottom, top)),lambda_tv=lambda_tv,pad=pad,randomize=randomize)
            else:
                displacements = calculate_displacements_singlechannel(data_for_nav, nb_segments, shifts=list(range(bottom, top)),
                                                                    lambda_tv=lambda_tv,ch=ch,pad=pad,randomize=randomize,dct_frequency_filter=dct_frequency_filter,seasonal_adj=seasonal_adj,interp_bad_correl=interp_bad_correl)
        np.save(filename_displacement, displacements)

    else:
        displacements=np.load(filename_displacement)

    if nav_file:
        nb_slices=data_for_nav.shape[1]
        nb_gating_spokes=data_for_nav.shape[2]
    else:
        nb_slices=displacements.shape[0]
        nb_gating_spokes=displacements.shape[1]

    if hard_interp:
        disp_interp=copy(displacements).reshape(-1,nb_gating_spokes)
        disp_interp[:, :8] = ((disp_interp[:, 7] - disp_interp[:, 0]) / (7 - 0))[:, None] * np.arange(8)[None,
                                                                                            :] + disp_interp[:, 0][:,
                                                                                                 None]
        displacements = disp_interp.flatten()
        np.save(filename_displacement, displacements)

    filename_displacement_plot = filename_displacement.replace(".npy",".jpg")
    plt.plot(displacements.flatten())
    plt.savefig(filename_displacement_plot)

    radial_traj=Radial3D(total_nspokes=nb_segments,undersampling_factor=1,npoint=800,nb_slices=nb_slices*us,incoherent=incoherent,mode="old",nspoke_per_z_encoding=nspoke_per_z)

    if filename_bins is None:
        dico_traj_retained,dico_retained_ts,bins=estimate_weights_bins(displacements,nb_slices,nb_segments,nb_gating_spokes,ntimesteps,radial_traj,nb_bins=nbins,retained_categories=retained_categories,equal_spoke_per_bin=equal_spoke_per_bin,sim_us=sim_us,us_file=us_file,us=us,soft_weight_for_full_inspi=soft_weight,nb_rep_center_part=nb_rep_center_part,std_disp=stddisp,disp_respi=disp_respi)

        np.save(filename_bins_output,bins)
    else:
        bins=np.load(filename_bins)
        if nav_res_factor is not None:
            bins=nav_res_factor*bins
            print("Rescaled bins {}".format(bins))
        nb_bins=len(bins)+1
        print(nb_bins)
        dico_traj_retained,dico_retained_ts,_=estimate_weights_bins(displacements,nb_slices,nb_segments,nb_gating_spokes,ntimesteps,radial_traj,nb_bins=nb_bins,retained_categories=retained_categories,bins=bins,sim_us=sim_us,us_file=us_file,us=us,soft_weight_for_full_inspi=soft_weight,nb_rep_center_part=nb_rep_center_part,disp_respi=disp_respi)

    weights=[]
    for gr in dico_traj_retained.keys():
        weights.append(np.expand_dims(dico_traj_retained[gr],axis=-1))
    weights=np.array(weights)
    if gating_only:
        weights=(weights>0)*1
    print(weights.shape)
    np.save(filename_weights, weights)

    file = open(filename_retained_ts, "wb")
    pickle.dump(dico_retained_ts, file)
    file.close()
    return


@ma.machine()
@ma.parameter("filename_kdata", str, default=None, description="MRF raw data")
@ma.parameter("dens_adj", bool, default=False, description="Radial density adjustment")
@ma.parameter("n_comp", int, default=None, description="Number of virtual coils")
@ma.parameter("nb_rep_center_part", int, default=1, description="Number of center partition repetition")
@ma.parameter("invert_dens_adj", bool, default=False, description="Remove Radial density adjustment")
@ma.parameter("res",int,default=16,description="central points kept for coil sensitivity calc")
@ma.parameter("res_kz",int,default=None,description="central partitions kept for coil sensitivity calc when no coil compression")
@ma.parameter("cc_res",int,default=None,description="central points kept for coil compression")
@ma.parameter("cc_res_kz",int,default=None,description="central partitions kept for coil compression")
def coil_compression(filename_kdata, dens_adj,n_comp,nb_rep_center_part,invert_dens_adj,res,cc_res,res_kz,cc_res_kz):
    kdata_all_channels_all_slices = np.load(filename_kdata)
    
    nb_channels=kdata_all_channels_all_slices.shape[0]
    print("Nb Channels : {}".format(nb_channels))

    filename_virtualcoils = str.split(filename_kdata, "_kdata.npy")[0] + "_virtualcoils_{}.pkl".format(n_comp)
    filename_b12Dplus1 = str.split(filename_kdata, "_kdata.npy")[0] + "_b12Dplus1_{}.npy".format(n_comp)

    if nb_rep_center_part>1:
        kdata_all_channels_all_slices=kdata_aggregate_center_part(kdata_all_channels_all_slices,nb_rep_center_part)

    if dens_adj:
        npoint=kdata_all_channels_all_slices.shape[-1]
        density = np.abs(np.linspace(-1, 1, npoint))
        density = np.expand_dims(density, tuple(range(kdata_all_channels_all_slices.ndim - 1)))

        print("Performing Density Adjustment....")
        kdata_all_channels_all_slices *= density


    if n_comp>=nb_channels:#no coil compression
        n_comp=nb_channels
        print("No Coil Compression")
        data_shape=kdata_all_channels_all_slices.shape
        print(data_shape)
        nb_allspokes = data_shape[-3]
        npoint = data_shape[-1]
        nb_slices = data_shape[-2]
        image_size=(nb_slices,int(npoint/2),int(npoint/2))
        radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=1,npoint=npoint,nb_slices=nb_slices,incoherent=False,nspoke_per_z_encoding=nb_allspokes)
        print("Here")
        b1_all_slices_2Dplus1_pca=calculate_sensitivity_map_3D(kdata_all_channels_all_slices,radial_traj,res,image_size,useGPU=False,light_memory_usage=True,hanning_filter=True,res_kz=res_kz)
        pca_dict={}
        for sl in range(nb_slices):
            pca=PCAComplex(n_components_=nb_channels)
            pca.explained_variance_ratio_=[1]
            pca.components_=np.eye(nb_channels)
            pca_dict[sl]=deepcopy(pca)
        
        print("B1 shape {}".format(b1_all_slices_2Dplus1_pca.shape))


    else:
        pca_dict,b1_all_slices_2Dplus1_pca=coil_compression_2Dplus1(kdata_all_channels_all_slices, n_comp=n_comp,invert_dens_adj=invert_dens_adj,res=res,cc_res=cc_res,cc_res_kz=cc_res_kz)


    image_file=str.split(filename_b12Dplus1, ".npy")[0] + ".jpg"

    sl = int(b1_all_slices_2Dplus1_pca.shape[1]/2)

    list_images=list(np.abs(b1_all_slices_2Dplus1_pca[:,sl,:,:]))
    plot_image_grid(list_images,(6,6),title="Sensivitiy map for slice".format(sl),save_file=image_file)


    with open(filename_virtualcoils, "wb") as file:
        pickle.dump(pca_dict, file)

    np.save(filename_b12Dplus1, b1_all_slices_2Dplus1_pca)

    return




@ma.machine()
@ma.parameter("filename_kdata", str, default=None, description="MRF raw data")
@ma.parameter("dens_adj", bool, default=False, description="Radial density adjustment")
def build_coil_images(filename_kdata,dens_adj):
    kdata_all_channels_all_slices = np.load(filename_kdata)

    nb_channels,nb_segments,nb_slices,npoint=kdata_all_channels_all_slices.shape

    filename_coilimg = str.split(filename_kdata, "_kdata.npy")[0] + "_bart_coilimg.pkl"

    
    if dens_adj:
        density = np.abs(np.linspace(-1, 1, npoint))
        density = np.expand_dims(density, tuple(range(kdata_all_channels_all_slices.ndim - 1)))

        print("Performing Density Adjustment....")
        kdata_all_channels_all_slices *= density
    
    kdata_bart=np.moveaxis(kdata_all_channels_all_slices,-1,1)
    kdata_bart=np.moveaxis(kdata_bart,0,-1)
    kdata_bart=kdata_bart[None,:]
    kdata_bart=kdata_bart.reshape(1,npoint,-1,nb_channels)

    incoherent=False
    radial_traj = Radial3D(total_nspokes=nb_segments, undersampling_factor=1, npoint=npoint,
                                nb_slices=nb_slices, incoherent=incoherent, mode=None,nspoke_per_z_encoding=nb_segments,)

    traj_python = radial_traj.get_traj()
    traj_python=traj_python.reshape(nb_segments,nb_slices,-1,3)
    traj_python=traj_python.T
    traj_python=np.moveaxis(traj_python,-1,-2)
    
    traj_python[0] = traj_python[0] / np.max(np.abs(traj_python[0])) * int(
        npoint / 4)
    traj_python[1] = traj_python[1] / np.max(np.abs(traj_python[1])) * int(
            npoint / 4)
    traj_python[2] = traj_python[2] / np.max(np.abs(traj_python[2])) * int(
            nb_slices / 2)
        
    print("Building coil images")
    traj_python_bart=traj_python.reshape(3,npoint,-1)
    coil_img = bart(1,'nufft -a -t', traj_python_bart, kdata_bart)

    np.save(filename_coilimg,coil_img)

    return

@ma.machine()
@ma.parameter("file_bart", str, default=None, description="bart file")
@ma.parameter("sl", int, default=30, description="Slice number")
def plot_image_grid_bart(file_bart,sl):
    file_plot=file_bart+".jpg"
    img_LLR=cfl.readcfl(file_bart)
    plot_image_grid(np.abs(np.moveaxis(img_LLR.squeeze()[sl],-1,0)),nb_row_col=(3,2),same_range=True)
    plt.savefig(file_plot)
    return






@ma.machine()
@ma.parameter("filename_kdata", str, default=None, description="MRF raw data")
@ma.parameter("filename_seqParams", str, default=None, description="Seq params")
@ma.parameter("dens_adj", bool, default=False, description="Radial density adjustment")
@ma.parameter("lowmem", bool, default=False, description="Low memory nufft in bart")
@ma.parameter("n_comp", int, default=None, description="Number of virtual coils")
@ma.parameter("filename_cc", str, default=None, description="Filename for coil compression")
@ma.parameter("calc_sensi", bool, default=True, description="Calculate coil sensitivities")
@ma.parameter("iskushball", bool, default=False, description="3D Kushball sampling")
@ma.parameter("spoke_start", int, default=None, description="Starting segment to avoid inversion")
@ma.parameter("us", int, default=1, description="undersampling partitions")
def coil_compression_bart(filename_kdata,dens_adj,n_comp,filename_cc,calc_sensi,iskushball,filename_seqParams,spoke_start,us,lowmem):
    kdata_all_channels_all_slices = np.load(filename_kdata)

    nb_channels,nb_segments,nb_part,npoint=kdata_all_channels_all_slices.shape

    print(kdata_all_channels_all_slices.shape)


    filename_virtualcoils = str.split(filename_kdata, "_kdata.npy")[0] + "_bart{}_virtualcoils_{}.pkl".format(n_comp,n_comp)
    filename_b12Dplus1 = str.split(filename_kdata, "_kdata.npy")[0] + "_bart{}_b12Dplus1_{}.npy".format(n_comp,n_comp)
    filename_coilimg = str.split(filename_kdata, "_kdata.npy")[0] + "_bart{}_coil_img_{}.npy".format(n_comp,n_comp)
    filename_kdata_compressed = str.split(filename_kdata, "_kdata.npy")[0] + "_bart{}_kdata.npy".format(n_comp)
    coil_image_file=str.split(filename_b12Dplus1, ".npy")[0] + "_coilimg.jpg"

    if filename_seqParams is None:
        filename_seqParams =filename_kdata.split("_kdata.npy")[0] + "_seqParams.pkl"

    file = open(filename_seqParams, "rb")
    dico_seqParams = pickle.load(file)
    file.close()

    print(dico_seqParams)

    if "Spherical" in dico_seqParams.keys():
        iskushball=dico_seqParams["Spherical"]
    else:
        iskushball=False

    nb_slices = int(dico_seqParams["nb_part"])
    
    if dens_adj:
        print("Performing Density Adjustment....")
        density = np.abs(np.linspace(-1, 1, npoint))
        density = np.expand_dims(density, tuple(range(kdata_all_channels_all_slices.ndim - 1)))
        if iskushball:
            # phi1 = 0.4656
            # phi = np.arccos(np.mod(np.arange(nb_segments * nb_slices) * phi1, 1))
            kdata_all_channels_all_slices *= density**2#*np.sin(phi.reshape(nb_slices, nb_segments).T[None, :, :, None])
        else:
            kdata_all_channels_all_slices *= density
    
    kdata_bart=np.moveaxis(kdata_all_channels_all_slices,-1,1)
    kdata_bart=np.moveaxis(kdata_bart,0,-1)
    kdata_bart=kdata_bart[None,:]

    kdata_bart=kdata_bart.reshape(1,npoint,-1,nb_channels)

    if lowmem:
        suffix_lowmem="--lowmem --no-precomp"
    else:
        suffix_lowmem=""
    
    if (filename_cc is None) or calc_sensi:
        

        incoherent=False
        if iskushball:
            print("Kushball reco")
            radial_traj=Radial3D(total_nspokes=nb_segments,undersampling_factor=1,npoint=npoint,nb_slices=nb_part,mode="Kushball")
        else:
            radial_traj = Radial3D(total_nspokes=nb_segments, undersampling_factor=1, npoint=npoint,
                                nb_slices=nb_part, incoherent=incoherent, mode=None,nspoke_per_z_encoding=nb_segments,)

        traj_python = radial_traj.get_traj()
        traj_python=traj_python.reshape(nb_segments,nb_part,-1,3)
        if spoke_start is not None:
            traj_python=traj_python[spoke_start:]
        
        traj_python=traj_python.T
        traj_python=np.moveaxis(traj_python,-1,-2)

        traj_python[0] = traj_python[0] / np.max(np.abs(traj_python[0])) * int(
        npoint / 4)
        traj_python[1] = traj_python[1] / np.max(np.abs(traj_python[1])) * int(
            npoint / 4)
        # if iskushball:
        #     traj_python[2] = traj_python[2] / np.max(np.abs(traj_python[2])) * int(
        #     npoint / 4)
        
        # else:
        #     traj_python[2] = traj_python[2] / np.max(np.abs(traj_python[2])) * int(
        #     nb_slices / 2)
        
        
        traj_python[2] = traj_python[2] / np.max(np.abs(traj_python[2])) * int(
            nb_slices / 2) # TO CHECK
        
        print(traj_python.shape)

        traj_python=traj_python[:,:,:,::us]
        traj_python_bart=traj_python.reshape(3,npoint,-1)

        print(traj_python_bart.shape)
        print(kdata_bart.shape)

        if spoke_start is not None:
                coil_img=bart(1,'nufft {} -a -t'.format(suffix_lowmem), traj_python_bart, (kdata_bart.reshape(1,npoint,nb_segments,nb_part,nb_channels)[:,:,spoke_start:,::us]).reshape(1,npoint,-1,nb_channels))
        else:
            coil_img = bart(1,'nufft {} -a -t'.format(suffix_lowmem), traj_python_bart, kdata_bart)

        print(coil_img.shape)
        
        coil_img_plot=np.moveaxis(coil_img,-2,0)
        coil_img_plot=np.moveaxis(coil_img_plot,-1,0)
        np.save(filename_coilimg,coil_img_plot)

        sl = int(coil_img_plot.shape[1]/2)

        list_images=list(np.abs(coil_img_plot[:,sl,:,:]))
        plot_image_grid(list_images,(6,6),title="BART Coil Images for slice".format(sl),save_file=coil_image_file)
        
        kdata_cart = bart(1,'fft -u 7', coil_img)

    if filename_cc is None:
        print("Calculating coil compression")
        filename_cc = str.split(filename_kdata, "_kdata.npy")[0] + "_bart_cc.cfl"
        cc=bart(1,"cc -M",kdata_cart)
        cfl.writecfl(filename_cc,cc)
    else:
        print("Loading Coil compression")
        cc=cfl.readcfl(filename_cc)

    print("Applying coil compression to k-space data")
    kdata_bart_cc = bart(1, 'ccapply -p {}'.format(n_comp), kdata_bart, cc)
    kdata_python_cc=kdata_bart_cc.squeeze().reshape(npoint,nb_segments,nb_part,n_comp)
    kdata_python_cc=np.moveaxis(kdata_python_cc,-1,0)
    kdata_python_cc=np.moveaxis(kdata_python_cc,1,-1)
    np.save(filename_kdata_compressed,kdata_python_cc)

    if calc_sensi:
        print("Calculating coil sensi")
        kdata_cart_cc = bart(1, 'ccapply -p {}'.format(n_comp), kdata_cart, cc)
        b1_bart_cc=bart(1,"ecalib -m1",kdata_cart_cc)
        b1_python_cc=np.moveaxis(b1_bart_cc,-2,0)
        b1_python_cc=np.moveaxis(b1_python_cc,-1,0)
        np.save(filename_b12Dplus1,b1_python_cc)

        image_file=str.split(filename_b12Dplus1, ".npy")[0] + ".jpg"

        sl = int(b1_python_cc.shape[1]/2)

        list_images=list(np.abs(b1_python_cc[:,sl,:,:]))
        plot_image_grid(list_images,(6,6),title="BART Coil Sensitivity map for slice".format(sl),save_file=image_file)

        pca_dict={}
        for sl in range(nb_part):
            pca=PCAComplex(n_components_=n_comp)
            pca.explained_variance_ratio_=[1]
            pca.components_=np.eye(n_comp)
            pca_dict[sl]=deepcopy(pca)

        with open(filename_virtualcoils, "wb") as file:
            pickle.dump(pca_dict, file)





@ma.machine()
@ma.parameter("filename_volume", str, default=None, description="Volume time serie")
@ma.parameter("sl", str, default=None, description="Slices to select")
def select_slices_volume(filename_volume, sl):
    filename_volume_new=filename_volume.split(".npy")[0]+"_{}.npy".format(sl)
    
    volume=np.load(filename_volume)
    slices=np.array(sl.split(",")).astype(int)
    volume_new=volume[:,slices]

    np.save(filename_volume_new, volume_new)

    return


@ma.machine()
@ma.parameter("filename_kdata", str, default=None, description="MRF raw data")
@ma.parameter("filename_b1", str, default=None, description="B1")
@ma.parameter("filename_pca", str, default=None, description="filename for storing coil compression components")
@ma.parameter("filename_weights", str, default=None, description="Motion bin weights")
@ma.parameter("n_comp", int, default=None, description="Virtual coils components to load b1 and pca file")
@ma.parameter("gating_only", bool, default=False, description="Use weights only for gating")
@ma.parameter("dens_adj", bool, default=False, description="Radial density adjustment")
@ma.parameter("in_phase", bool, default=False, description="Select only in phase spokes from original MRF sequence")
@ma.parameter("out_phase", bool, default=False, description="Select only out of phase spokes from original MRF sequence")
@ma.parameter("full_volume", bool, default=False, description="Build one volume with all spokes (weights are not used)")
@ma.parameter("nb_rep_center_part", int, default=1, description="Center partition repetitions")
@ma.parameter("us", int, default=1, description="Undersampling")
def build_volumes_allbins(filename_kdata,filename_b1,filename_pca,filename_weights,n_comp,gating_only,dens_adj,in_phase,out_phase,full_volume,nb_rep_center_part,us):
    '''
    Build single volume for each motion phase with all spokes (for motion deformation field estimation)
    '''
    if not(gating_only):
        filename_volumes=filename_kdata.split("_kdata.npy")[0] + "_volumes_allbins.npy"
    else:
        filename_volumes = filename_kdata.split("_kdata.npy")[0] + "_no_dcomp_volumes_allbins.npy"

    if full_volume:
        filename_volumes=str.replace(filename_volumes,"volumes_allbins","full_volume")

    print("Loading Kdata")
    kdata_all_channels_all_slices=np.load(filename_kdata)
    if dens_adj:
        npoint = kdata_all_channels_all_slices.shape[-1]
        density = np.abs(np.linspace(-1, 1, npoint))
        density = np.expand_dims(density, tuple(range(kdata_all_channels_all_slices.ndim - 1)))

        print("Performing Density Adjustment....")
        kdata_all_channels_all_slices *= density

    if ((filename_b1 is None)or(filename_pca is None))and(n_comp is None):
        raise ValueError('n_comp should be provided when B1 or PCA files are missing')

    if filename_b1 is None:
        filename_b1=(str.split(filename_kdata, "_kdata.npy")[0] + "_b12Dplus1_{}.npy".format(n_comp)).replace("_no_densadj","").replace("_no_dens_adj","")

    if filename_pca is None:
        filename_pca = (str.split(filename_kdata, "_kdata.npy")[0] + "_virtualcoils_{}.pkl".format(n_comp)).replace("_no_densadj","")

    if filename_weights is None:
        filename_weights = (str.split(filename_kdata, "_kdata.npy")[0] + "_weights.npy").replace("_no_densadj","")
    
    b1_all_slices_2Dplus1_pca=np.load(filename_b1)

    if not(full_volume):
        all_weights=np.load(filename_weights)
    else:
        all_weights=np.ones(shape=(1,1,kdata_all_channels_all_slices.shape[1],kdata_all_channels_all_slices.shape[2],1))
    if gating_only:
        all_weights=(all_weights>0)*1

    if us >1:
        weights_us = np.zeros_like(all_weights)
        nb_slices = all_weights.shape[3]
        nspoke_per_part = 8
        weights_us = weights_us.reshape((weights_us.shape[0], 1, -1, nspoke_per_part, nb_slices, 1))


        curr_start = 0

        for sl in range(nb_slices):
            weights_us[:, :, curr_start::us, :, sl] = 1
            curr_start = curr_start + 1
            curr_start = curr_start % us

        weights_us=weights_us.reshape(all_weights.shape)
        all_weights *= weights_us

    file = open(filename_pca, "rb")
    pca_dict = pickle.load(file)
    file.close()

    print("Kdata shape {}".format(kdata_all_channels_all_slices.shape))
    print("PCA components shape {}".format(pca_dict[0].components_.shape))
    print("Weights shape {}".format(all_weights.shape))
    print("B1 shape {}".format(b1_all_slices_2Dplus1_pca.shape))

    if in_phase:
        selected_spokes=np.r_[300:800,1200:1400]
        #selected_spokes=np.r_[280:580]
    elif out_phase:
        selected_spokes = np.r_[800:1200]
    else:
        selected_spokes=None

    volumes_allbins=build_volume_2Dplus1_cc_allbins(kdata_all_channels_all_slices,b1_all_slices_2Dplus1_pca,pca_dict,all_weights,selected_spokes,nb_rep_center_part)
    np.save(filename_volumes,volumes_allbins)

    return


@ma.machine()
@ma.parameter("filename_kdata", str, default=None, description="MRF raw data")
@ma.parameter("filename_b1", str, default=None, description="B1")
@ma.parameter("filename_pca", str, default=None, description="filename for storing coil compression components")
@ma.parameter("filename_weights", str, default=None, description="Motion bin weights")
@ma.parameter("dictfile", str, default=None, description="MRF dictionary file for temporal basis")
@ma.parameter("L0", int, default=10, description="Number of retained temporal basis functions")
@ma.parameter("file_deformation_map", str, default=None, description="Deformation map from bin 0 to other bins")
@ma.parameter("n_comp", int, default=None, description="Virtual coils components to load b1 and pca file")
@ma.parameter("useGPU", bool, default=True, description="Use GPU")
@ma.parameter("nb_rep_center_part", int, default=1, description="Number of center partition repetition")
@ma.parameter("index_ref", int, default=0, description="Reference bin for deformation")
@ma.parameter("interp", str, default=None, description="Registration interpolation")
@ma.parameter("suffix", str, default="", description="Suffix")
@ma.parameter("gating_only", bool, default=False, description="Use weights only for gating")
@ma.parameter("select_first_rep", bool, default=False, description="Select firt repetition of central partition only")
@ma.parameter("axis", int, default=None, description="Registration axis")
def build_volumes_singular_allbins_registered(filename_kdata, filename_b1, filename_pca, filename_weights,dictfile,L0,file_deformation_map,n_comp,useGPU,nb_rep_center_part,index_ref,interp,gating_only,suffix,select_first_rep,axis):
    '''
    Build singular volumes for MRF registered to the same motion phase and averaged (first iteration of the gradient descent for motion-corrected MRF)
    Output shape L0 x nz x nx x ny
    '''
    filename_volumes = filename_kdata.split("_kdata.npy")[0] + "_volumes_singular_allbins_registered_gr{}{}.npy".format(index_ref,suffix)
    print("Loading Kdata")
    
    kdata_all_channels_all_slices = np.load(filename_kdata)

    if ((filename_b1 is None)or(filename_pca is None))and(n_comp is None):
        raise ValueError('n_comp should be provided when B1 or PCA files are missing')

    if filename_b1 is None:
        filename_b1=str.split(filename_kdata, "_kdata.npy")[0] + "_b12Dplus1_{}.npy".format(n_comp)

    if filename_pca is None:
        filename_pca = str.split(filename_kdata, "_kdata.npy")[0] + "_virtualcoils_{}.pkl".format(n_comp)

    if filename_weights is None:
        filename_weights = str.split(filename_kdata, "_kdata.npy")[0] + "_weights.npy"



    print("Loading B1")
    b1_all_slices_2Dplus1_pca = np.load(filename_b1)
    print("Loading Weights")
    all_weights = np.load(filename_weights)

    if gating_only:
        all_weights=(all_weights>0)*1

    print("Loading Coil Compression weights")
    file = open(filename_pca, "rb")
    pca_dict = pickle.load(file)
    file.close()

    print("Loading Time Basis")
    mrf_dict = load_pickle(dictfile)
    if "phi" not in mrf_dict.keys():
        mrf_dict=add_temporal_basis(mrf_dict,L0)
        save_pickle(mrf_dict,mrf_dict)
    phi=mrf_dict["phi"]
    print("Loading Deformation Map")
    deformation_map=np.load(file_deformation_map)
    if not(index_ref==0):
        deformation_map=change_deformation_map_ref(deformation_map,index_ref,axis)

    print("Kdata shape {}".format(kdata_all_channels_all_slices.shape))
    print("virtual coils components shape {}".format(pca_dict[0].components_.shape))
    print("Weights shape {}".format(all_weights.shape))
    print("phi shape {}".format(phi.shape))
    print("B1 shape {}".format(b1_all_slices_2Dplus1_pca.shape))


    if interp is None:
        interp=cv2.INTER_LINEAR

    elif interp=="nearest":
        interp=cv2.INTER_NEAREST
    
    elif interp=="cubic":
        interp=cv2.INTER_CUBIC



    volumes_allbins_registered=build_volume_singular_2Dplus1_cc_allbins_registered(kdata_all_channels_all_slices, b1_all_slices_2Dplus1_pca, pca_dict,
                                               all_weights, phi, L0, deformation_map,useGPU,nb_rep_center_part,interp,select_first_rep,axis)
    np.save(filename_volumes, volumes_allbins_registered)

    return




@ma.machine()
@ma.parameter("filename_kdata", str, default=None, description="MRF raw data")
@ma.parameter("filename_b1", str, default=None, description="B1")
@ma.parameter("filename_pca", str, default=None, description="filename for storing coil compression components")
@ma.parameter("filename_weights", str, default=None, description="Motion bin weights")
@ma.parameter("dictfile", str, default=None, description="MRF dictionary file for temporal basis")
@ma.parameter("L0", int, default=10, description="Number of retained temporal basis functions")
@ma.parameter("n_comp", int, default=None, description="Virtual coils components to load b1 and pca file")
@ma.parameter("useGPU", bool, default=True, description="Use GPU")
@ma.parameter("dens_adj", bool, default=False, description="Radial density adjustment")
@ma.parameter("nb_rep_center_part", int, default=1, description="Number of center partition repetition")
@ma.parameter("gating_only", bool, default=False, description="Use weights only for gating")
@ma.parameter("us", int, default=1, description="Undersampling")


def build_volumes_singular_allbins(filename_kdata, filename_b1, filename_pca, filename_weights,dictfile,L0,n_comp,useGPU,dens_adj,nb_rep_center_part,gating_only,us):
    '''
    Build singular volumes for MRF for all motion bins
    Output shape nb_motion_bins x L0 x nz x nx x ny
    '''
    


    filename_volumes = filename_kdata.split("_kdata.npy")[0] + "_volumes_singular_allbins.npy"
    print("Loading Kdata")
    print(filename_kdata)
    kdata_all_channels_all_slices = np.load(filename_kdata)
    print(filename_kdata)

    if dens_adj:
        npoint = kdata_all_channels_all_slices.shape[-1]
        density = np.abs(np.linspace(-1, 1, npoint))
        density = np.expand_dims(density, tuple(range(kdata_all_channels_all_slices.ndim - 1)))

        print("Performing Density Adjustment....")
        kdata_all_channels_all_slices *= density


    if ((filename_b1 is None)or(filename_pca is None))and(n_comp is None):
        raise ValueError('n_comp should be provided when B1 or PCA files are missing')

    if filename_b1 is None:
        filename_b1=str.split(filename_kdata, "_kdata.npy")[0] + "_b12Dplus1_{}.npy".format(n_comp)

    if filename_pca is None:
        filename_pca = str.split(filename_kdata, "_kdata.npy")[0] + "_virtualcoils_{}.pkl".format(n_comp)

    if filename_weights is None:
        filename_weights = str.split(filename_kdata, "_kdata.npy")[0] + "_weights.npy"

    b1_all_slices_2Dplus1_pca = np.load(filename_b1)

    nb_slices_b1=b1_all_slices_2Dplus1_pca.shape[1]
    npoint_b1=b1_all_slices_2Dplus1_pca.shape[-1]
    print(kdata_all_channels_all_slices.shape)
    print(nb_slices_b1)
    nb_slices=kdata_all_channels_all_slices.shape[-2]-nb_rep_center_part+1
    npoint=int(kdata_all_channels_all_slices.shape[-1]/2)

    nb_channels=b1_all_slices_2Dplus1_pca.shape[0]
    
    file = open(filename_pca, "rb")
    pca_dict = pickle.load(file)
    file.close()

    # if nb_slices>nb_slices_b1:
    #     us_b1 = int(nb_slices / nb_slices_b1)
    #     print("B1 map on x{} coarser grid. Interpolating B1 map on a finer grid".format(us_b1))
    #     b1_all_slices_2Dplus1_pca=interp_b1(b1_all_slices_2Dplus1_pca,us=us_b1,start=0)

    #     print("Warning: pca_dict can only be interpolated when no coil compression for the moment")
    #     pca_dict={}
    #     for sl in range(nb_slices):
    #         pca=PCAComplex(n_components_=nb_channels)
    #         pca.explained_variance_ratio_=[1]
    #         pca.components_=np.eye(nb_channels)
    #         pca_dict[sl]=deepcopy(pca)

    if (nb_slices>nb_slices_b1)or(npoint>npoint_b1):
        
        print("Regridding b1")
        new_shape=(nb_slices,npoint,npoint)
        b1_all_slices_2Dplus1_pca=interp_b1_resize(b1_all_slices_2Dplus1_pca,new_shape)

        print("Warning: pca_dict can only be interpolated when no coil compression for the moment")
        pca_dict={}
        for sl in range(nb_slices):
            pca=PCAComplex(n_components_=nb_channels)
            pca.explained_variance_ratio_=[1]
            pca.components_=np.eye(nb_channels)
            pca_dict[sl]=deepcopy(pca)

    all_weights = np.load(filename_weights)

    if gating_only:
        print("Using weights for gating only")
        all_weights=(all_weights>0)*1


    if us >1:
        weights_us = np.zeros_like(all_weights)
        nb_slices = all_weights.shape[3]
        nspoke_per_part = 8
        weights_us = weights_us.reshape((weights_us.shape[0], 1, -1, nspoke_per_part, nb_slices, 1))


        curr_start = 0

        for sl in range(nb_slices):
            weights_us[:, :, curr_start::us, :, sl] = 1
            curr_start = curr_start + 1
            curr_start = curr_start % us

        weights_us=weights_us.reshape(all_weights.shape)
        all_weights *= weights_us


    mrf_dict = load_pickle(dictfile)
    if "phi" not in mrf_dict.keys():
        mrf_dict=add_temporal_basis(mrf_dict,L0)
        save_pickle(mrf_dict,mrf_dict)
    phi=mrf_dict["phi"]
    
    print("Kdata shape {}".format(kdata_all_channels_all_slices.shape))
    print("virtual coils components shape {}".format(pca_dict[0].components_.shape))
    print("Weights shape {}".format(all_weights.shape))
    print("phi shape {}".format(phi.shape))
    print("B1 shape {}".format(b1_all_slices_2Dplus1_pca.shape))




    volumes_allbins=build_volume_singular_2Dplus1_cc_allbins(kdata_all_channels_all_slices, b1_all_slices_2Dplus1_pca, pca_dict,
                                               all_weights, phi, L0,useGPU,nb_rep_center_part=nb_rep_center_part)
    np.save(filename_volumes, volumes_allbins)

    return


@ma.machine()
@ma.parameter("filename_kdata", str, default=None, description="MRF raw data")
@ma.parameter("filename_weights", str, default=None, description="Weights")
@ma.parameter("filename_seqParams", str, default=None, description="Seq params")
def calculate_dcomp_voronoi_3D(filename_kdata,filename_weights,filename_seqParams):
    '''
    Build singular volumes for MRF for all motion bins
    Output shape nb_motion_bins x L0 x nz x nx x ny
    '''
    
    filename_dcomp = filename_kdata.split("_kdata.npy")[0] + "_dcomp.npy"
    print("Loading Kdata")
    print(filename_kdata)
    kdata_all_channels_all_slices = np.load(filename_kdata)

    nb_channels,nb_segments,nb_part,npoint=kdata_all_channels_all_slices.shape

    all_weights=np.load(filename_weights)
    nbins=all_weights.shape[0]

    if filename_seqParams is not None:
        file = open(filename_seqParams, "rb")
        dico_seqParams = pickle.load(file)
        file.close()

        use_navigator_dll = dico_seqParams["use_navigator_dll"]

        if "use_kushball_dll" in dico_seqParams.keys():
            use_kushball_dll=dico_seqParams["use_kushball_dll"]
        else:
            use_kushball_dll=False


        if use_kushball_dll:
            meas_sampling_mode = dico_seqParams["alFree"][16]
        elif (use_navigator_dll):
            meas_sampling_mode = dico_seqParams["alFree"][15]
        else:
            meas_sampling_mode = dico_seqParams["alFree"][12]

        print(meas_sampling_mode)



        undersampling_factor = dico_seqParams["alFree"][9]



        if meas_sampling_mode == 1:
            incoherent = False
            mode = None
        elif meas_sampling_mode == 2:
            incoherent = True
            mode = "old"
        elif meas_sampling_mode == 3:
            incoherent = True
            mode = "new"

        elif meas_sampling_mode == 4:
            incoherent = True
            mode = "Kushball"

    else:
        incoherent=False
        mode="old"
        undersampling_factor=1

    radial_traj = Radial3D(total_nspokes=nb_segments,npoint=npoint,nb_slices=nb_part,undersampling_factor=undersampling_factor,incoherent=incoherent,mode=mode)
    traj_python = radial_traj.get_traj()
    # traj_python = traj_python.reshape(nb_segments, nb_part, -1, 3)
    traj_python=traj_python.reshape(-1,npoint,3)


    dcomp=np.zeros((nbins,1)+(nb_segments*nb_part,npoint))

    for gr in tqdm(range(nbins)):
        print("Calculating Voronoi for bin {}".format(gr))
        weights_for_traj_current_bin=all_weights[gr].squeeze().flatten()
        
        traj_python_current_bin=traj_python[weights_for_traj_current_bin>0]
        traj_python_current_bin=traj_python_current_bin.reshape(-1,3)
        density_voronoi=voronoi_volumes_freud(traj_python_current_bin)
        dcomp[gr,0,weights_for_traj_current_bin>0,:]=density_voronoi.reshape((-1,npoint))

    dcomp=dcomp.reshape((nbins,1,nb_segments,nb_part,npoint))
    np.save(filename_dcomp,dcomp)

    return





@ma.machine()
@ma.parameter("filename_kdata", str, default=None, description="MRF raw data")
@ma.parameter("filename_weights", str, default=None, description="Weights")
@ma.parameter("filename_seqParams", str, default=None, description="Seq params")
def calculate_dcomp_pysap_3D(filename_kdata,filename_weights,filename_seqParams):
    '''
    Build singular volumes for MRF for all motion bins
    Output shape nb_motion_bins x L0 x nz x nx x ny
    '''
    
    filename_dcomp = filename_kdata.split("_kdata.npy")[0] + "_dcomp.npy"
    print("Loading Kdata")
    print(filename_kdata)
    kdata_all_channels_all_slices = np.load(filename_kdata)

    nb_channels,nb_segments,nb_part,npoint=kdata_all_channels_all_slices.shape

    all_weights=np.load(filename_weights)
    nbins=all_weights.shape[0]

    if filename_seqParams is not None:
        file = open(filename_seqParams, "rb")
        dico_seqParams = pickle.load(file)
        file.close()

        use_navigator_dll = dico_seqParams["use_navigator_dll"]

        if "use_kushball_dll" in dico_seqParams.keys():
            use_kushball_dll=dico_seqParams["use_kushball_dll"]
        else:
            use_kushball_dll=False


        if use_kushball_dll:
            meas_sampling_mode = dico_seqParams["alFree"][16]
        elif (use_navigator_dll):
            meas_sampling_mode = dico_seqParams["alFree"][15]
        else:
            meas_sampling_mode = dico_seqParams["alFree"][12]

        print(meas_sampling_mode)



        undersampling_factor = dico_seqParams["alFree"][9]



        if meas_sampling_mode == 1:
            incoherent = False
            mode = None
        elif meas_sampling_mode == 2:
            incoherent = True
            mode = "old"
        elif meas_sampling_mode == 3:
            incoherent = True
            mode = "new"

        elif meas_sampling_mode == 4:
            incoherent = True
            mode = "Kushball"

    else:
        incoherent=False
        mode="old"
        undersampling_factor=1

    radial_traj = Radial3D(total_nspokes=nb_segments,npoint=npoint,nb_slices=nb_part,undersampling_factor=undersampling_factor,incoherent=incoherent,mode=mode)
    traj_python = radial_traj.get_traj()
    # traj_python = traj_python.reshape(nb_segments, nb_part, -1, 3)
    traj_python=traj_python.reshape(-1,npoint,3)


    dcomp=np.zeros((nbins,1)+(nb_segments*nb_part,npoint))

    for gr in tqdm(range(nbins)):
        print("Calculating Voronoi for bin {}".format(gr))
        weights_for_traj_current_bin=all_weights[gr].squeeze().flatten()
        
        traj_python_current_bin=traj_python[weights_for_traj_current_bin>0]
        traj_python_current_bin=traj_python_current_bin.reshape(-1,3)
        density_voronoi=voronoi_volumes_freud(traj_python_current_bin)
        dcomp[gr,0,weights_for_traj_current_bin>0,:]=density_voronoi.reshape((-1,npoint))

    dcomp=dcomp.reshape((nbins,1,nb_segments,nb_part,npoint))
    np.save(filename_dcomp,dcomp)

    return

@ma.machine()
@ma.parameter("filename_kdata", str, default=None, description="MRF raw data")
@ma.parameter("filename_b1", str, default=None, description="B1")
@ma.parameter("filename_weights", str, default=None, description="Motion bin weights")
@ma.parameter("filename_seqParams", str, default=None, description="Seq params")
@ma.parameter("filename_dcomp", str, default=None, description="Seq params")
@ma.parameter("dictfile", str, default=None, description="MRF dictionary file for temporal basis")
@ma.parameter("L0", int, default=10, description="Number of retained temporal basis functions")
@ma.parameter("n_comp", int, default=None, description="Virtual coils components to load b1 and pca file")
@ma.parameter("useGPU", bool, default=True, description="Use GPU")
@ma.parameter("dens_adj", bool, default=False, description="Radial density adjustment")
@ma.parameter("nb_rep_center_part", int, default=1, description="Number of center partition repetition")
@ma.parameter("gating_only", bool, default=False, description="Use weights only for gating")
@ma.parameter("incoherent", bool, default=False, description="Use GPU")

def build_volumes_singular_allbins_3D(filename_kdata, filename_b1, filename_weights,filename_dcomp,filename_seqParams,dictfile,L0,n_comp,useGPU,dens_adj,nb_rep_center_part,gating_only,incoherent):
    '''
    Build singular volumes for MRF for all motion bins
    Output shape nb_motion_bins x L0 x nz x nx x ny
    '''
    print(incoherent)
    filename_volumes = filename_kdata.split("_kdata.npy")[0] + "_volumes_singular_allbins.npy"
    print("Loading Kdata")
    print(filename_kdata)
    kdata_all_channels_all_slices = np.load(filename_kdata)
    print(filename_kdata)

    nb_channels,nb_allspokes,nb_part,npoint=kdata_all_channels_all_slices.shape

    if filename_seqParams is not None:
        file = open(filename_seqParams, "rb")
        dico_seqParams = pickle.load(file)
        file.close()

        use_navigator_dll = dico_seqParams["use_navigator_dll"]

        if "use_kushball_dll" in dico_seqParams.keys():
            use_kushball_dll=dico_seqParams["use_kushball_dll"]
        else:
            use_kushball_dll=False


        if use_kushball_dll:
            meas_sampling_mode = dico_seqParams["alFree"][16]
        elif (use_navigator_dll):
            meas_sampling_mode = dico_seqParams["alFree"][15]
        else:
            meas_sampling_mode = dico_seqParams["alFree"][12]

        print(meas_sampling_mode)



        undersampling_factor = dico_seqParams["alFree"][9]

        nb_slices=int(dico_seqParams["nb_part"])


        if meas_sampling_mode == 1:
            incoherent = False
            mode = None
        elif meas_sampling_mode == 2:
            incoherent = True
            mode = "old"
        elif meas_sampling_mode == 3:
            incoherent = True
            mode = "new"

        elif meas_sampling_mode == 4:
            incoherent = True
            mode = "Kushball"

    else:
        incoherent="False"
        mode="old"
        undersampling_factor=1
        nb_slices=nb_part



    if dens_adj:
        npoint = kdata_all_channels_all_slices.shape[-1]
        density = np.abs(np.linspace(-1, 1, npoint))
        density = np.expand_dims(density, tuple(range(kdata_all_channels_all_slices.ndim - 1)))

        print("Performing Density Adjustment....")
        if mode=="Kushball":
            kdata_all_channels_all_slices *= density**2
        else:
            kdata_all_channels_all_slices *= density


    if ((filename_b1 is None))and(n_comp is None):
        raise ValueError('n_comp should be provided when B1 or PCA files are missing')

    if filename_b1 is None:
        filename_b1=str.split(filename_kdata, "_kdata.npy")[0] + "_b12Dplus1_{}.npy".format(n_comp)

    if filename_weights is None:
        filename_weights = str.split(filename_kdata, "_kdata.npy")[0] + "_weights.npy"


    b1_all_slices_2Dplus1_pca = np.load(filename_b1)

    nb_slices_b1=b1_all_slices_2Dplus1_pca.shape[1]
    npoint_b1=b1_all_slices_2Dplus1_pca.shape[-1]

    npoint_image=int(npoint/2)
    
    if mode =="Kushball":
        print("Assuming isotropic image size for kushball sampling")
        image_size=(npoint_image,npoint_image,npoint_image)
    else:
        image_size=(nb_slices,npoint_image,npoint_image)

    

    # if nb_slices>nb_slices_b1:
    #     us_b1 = int(nb_slices / nb_slices_b1)
    #     print("B1 map on x{} coarser grid. Interpolating B1 map on a finer grid".format(us_b1))
    #     b1_all_slices_2Dplus1_pca=interp_b1(b1_all_slices_2Dplus1_pca,us=us_b1,start=0)

    #     print("Warning: pca_dict can only be interpolated when no coil compression for the moment")
    #     pca_dict={}
    #     for sl in range(nb_slices):
    #         pca=PCAComplex(n_components_=nb_channels)
    #         pca.explained_variance_ratio_=[1]
    #         pca.components_=np.eye(nb_channels)
    #         pca_dict[sl]=deepcopy(pca)

    if (nb_slices>nb_slices_b1)or(npoint_image>npoint_b1):
        
        print("Regridding b1")
        new_shape=(nb_slices,npoint_image,npoint_image)
        b1_all_slices_2Dplus1_pca=interp_b1_resize(b1_all_slices_2Dplus1_pca,new_shape)


    all_weights = np.load(filename_weights)

    if gating_only:
        print("Using weights for gating only")
        all_weights=(all_weights>0)*1

    if filename_dcomp is not None:
        print("Weighing by density compensation file")
        dcomp=np.load(filename_dcomp)
        print(dcomp.shape)
        dcomp[:,:,:,:,-1]=dcomp[:,:,:,:,-2]
        dcomp[:,:,:,:,0]=dcomp[:,:,:,:,1]
        
        all_weights=all_weights*dcomp


    # if filename_phi not in os.listdir():
    #     phi = build_phi(dictfile, L0)
    # else:
    #     phi = np.load(filename_phi)

    mrf_dict = load_pickle(dictfile)
    if "phi" not in mrf_dict.keys():
        mrf_dict=add_temporal_basis(mrf_dict,L0)
        save_pickle(mrf_dict,mrf_dict)
    phi=mrf_dict["phi"]

    print("Kdata shape {}".format(kdata_all_channels_all_slices.shape))
    print("Weights shape {}".format(all_weights.shape))
    print("phi shape {}".format(phi.shape))
    print("B1 shape {}".format(b1_all_slices_2Dplus1_pca.shape))

    print("nb_part {}".format(nb_part))
    print("image_size {}".format(image_size))

    radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=1,npoint=npoint,nb_slices=nb_part,incoherent=incoherent,mode=mode)
    volumes_allbins=build_volume_singular_3D_allbins(kdata_all_channels_all_slices, b1_all_slices_2Dplus1_pca, radial_traj,
                                               all_weights, phi, L0,useGPU,nb_rep_center_part=nb_rep_center_part,image_size=image_size)
    np.save(filename_volumes, volumes_allbins)

    return



@ma.machine()
@ma.parameter("bart_command", str, default=None, description="bart pics command")
@ma.parameter("filename_kdata", str, default=None, description="MRF raw data")
@ma.parameter("filename_b1", str, default=None, description="B1")
@ma.parameter("filename_weights", str, default=None, description="Motion bin weights")
@ma.parameter("filename_phi", str, default=None, description="MRF temporal basis components")
@ma.parameter("filename_seqParams", str, default=None, description="Seq params")
@ma.parameter("filename_dcomp", str, default=None, description="Seq params")
@ma.parameter("dictfile", str, default=None, description="MRF dictionary file for temporal basis")
@ma.parameter("L0", int, default=10, description="Number of retained temporal basis functions")
@ma.parameter("n_comp", int, default=None, description="Virtual coils components to load b1 and pca file")
@ma.parameter("useGPU", bool, default=True, description="Use GPU")
@ma.parameter("log", bool, default=True, description="log bin results")
@ma.parameter("dens_adj", bool, default=False, description="Radial density adjustment")
@ma.parameter("nb_rep_center_part", int, default=1, description="Number of center partition repetition")
@ma.parameter("gating_only", bool, default=False, description="Use weights only for gating")
def build_volumes_singular_allbins_3D_BART(bart_command,filename_kdata, filename_b1, filename_weights,filename_dcomp,filename_phi,filename_seqParams,dictfile,L0,n_comp,useGPU,dens_adj,nb_rep_center_part,gating_only,log):
    '''
    Build singular volumes for MRF for all motion bins
    Output shape nb_motion_bins x L0 x nz x nx x ny
    '''
    filename_volumes = filename_kdata.split("_kdata.npy")[0] + "_volumes_singular_allbins_bart.npy"
    filename_volumes_log = filename_kdata.split("_kdata.npy")[0] + "_volumes_singular_allbins_bart_gr{}.npy"
    print("Loading Kdata")
    print(filename_kdata)
    kdata_all_channels_all_slices = np.load(filename_kdata)
    print(filename_kdata)
    print(np.linalg.norm(kdata_all_channels_all_slices))

    nb_channels,nb_segments,nb_part,npoint=kdata_all_channels_all_slices.shape

    if filename_seqParams is not None:
        file = open(filename_seqParams, "rb")
        dico_seqParams = pickle.load(file)
        file.close()

        use_navigator_dll = dico_seqParams["use_navigator_dll"]

        if "use_kushball_dll" in dico_seqParams.keys():
            use_kushball_dll=dico_seqParams["use_kushball_dll"]
        else:
            use_kushball_dll=False


        if use_kushball_dll:
            meas_sampling_mode = dico_seqParams["alFree"][16]
        elif (use_navigator_dll):
            meas_sampling_mode = dico_seqParams["alFree"][15]
        else:
            meas_sampling_mode = dico_seqParams["alFree"][12]

        print(meas_sampling_mode)



        undersampling_factor = dico_seqParams["alFree"][9]

        nb_slices=int(dico_seqParams["nb_part"])


        if meas_sampling_mode == 1:
            incoherent = False
            mode = None
        elif meas_sampling_mode == 2:
            incoherent = True
            mode = "old"
        elif meas_sampling_mode == 3:
            incoherent = True
            mode = "new"

        elif meas_sampling_mode == 4:
            incoherent = True
            mode = "Kushball"



    else:
        incoherent=False
        mode="old"
        undersampling_factor=1
        nb_slices=nb_part
        image_size=(100,100,100)




    if dens_adj:


        npoint = kdata_all_channels_all_slices.shape[-1]
        if mode=="Kushball": 
            density = np.abs(np.linspace(-1, 1, npoint))**2
        else:
            density=np.abs(np.linspace(-1, 1, npoint))

        

        # bart_command=str.replace(bart_command,"-t","-p density -t")
        # print(bart_command)
        


    if ((filename_b1 is None))and(n_comp is None):
        raise ValueError('n_comp should be provided when B1 or PCA files are missing')

    if filename_b1 is None:
        filename_b1=str.split(filename_kdata, "_kdata.npy")[0] + "_b12Dplus1_{}.npy".format(n_comp)

    if filename_weights is None:
        filename_weights = str.split(filename_kdata, "_kdata.npy")[0] + "_weights.npy"

    if ((filename_phi is None) and (dictfile is None)):
        raise ValueError('Either dictfile or filename_phi should be provided for temporal projection')

    if dictfile is not None:
        filename_phi = str.split(dictfile, ".dict")[0] + "_phi_L0_{}.npy".format(L0)

    b1_all_slices_2Dplus1_pca = np.load(filename_b1)
    
    image_size=b1_all_slices_2Dplus1_pca.shape[1:]

    b1_bart=np.moveaxis(b1_all_slices_2Dplus1_pca,0,-1)
    b1_bart=np.moveaxis(b1_bart,0,-2)



    all_weights = np.load(filename_weights)
    nbins=all_weights.shape[0]

    if gating_only:
        print("Using weights for gating only")
        all_weights=(all_weights>0)*1

    if filename_dcomp is not None:
        print("Weighing by density compensation file")
        dcomp=np.load(filename_dcomp)
        print(dcomp.shape)
        dcomp[:,:,:,:,-1]=dcomp[:,:,:,:,-2]
        dcomp[:,:,:,:,0]=dcomp[:,:,:,:,1]
        
        all_weights=all_weights*dcomp


    # if filename_phi not in os.listdir():
    #     phi = build_phi(dictfile, L0)
    # else:
    #     phi = np.load(filename_phi)

    mrf_dict = load_pickle(dictfile)
    if "phi" not in mrf_dict.keys():
        mrf_dict=add_temporal_basis(mrf_dict,L0)
        save_pickle(mrf_dict,mrf_dict)
    phi=mrf_dict["phi"]

    print("Kdata shape {}".format(kdata_all_channels_all_slices.shape))
    print("Weights shape {}".format(all_weights.shape))
    print("phi shape {}".format(phi.shape))
    print("B1 shape {}".format(b1_all_slices_2Dplus1_pca.shape))

    print("nb_part {}".format(nb_part))
    print("image_size {}".format(image_size))

    radial_traj=Radial3D(total_nspokes=nb_segments,undersampling_factor=1,npoint=npoint,nb_slices=nb_part,incoherent=incoherent,mode=mode)
    
        
    traj_python = radial_traj.get_traj()
    traj_python = traj_python.reshape(nb_segments, nb_part, -1, 3)
    traj_python = traj_python.T
    traj_python = np.moveaxis(traj_python, -1, -2)

    traj_python[0] = traj_python[0] / np.max(np.abs(traj_python[0])) * int(image_size[0]/2)
    traj_python[1] = traj_python[1] / np.max(np.abs(traj_python[1])) * int(
        image_size[1]/2)
    traj_python[2] = traj_python[2] / np.max(np.abs(traj_python[2])) * int(
        image_size[2] / 2)

    traj_python_bart = traj_python.reshape(3, npoint, -1)
    
    window=8
    ntimesteps=int(nb_segments/window)
    

    kdata_singular = np.zeros((nb_channels,ntimesteps, nb_part*npoint*window) + (L0,), dtype="complex64")
    data=kdata_all_channels_all_slices.reshape(nb_channels,ntimesteps,-1)

    for ch in tqdm(range(nb_channels)):
        for ts in tqdm(range(ntimesteps)):
            kdata_singular[ch, ts, :, :] = data[ch, ts, :, None] @ (phi[:L0].conj().T[ts][None, :])

    print(np.linalg.norm(kdata_singular))

    kdata_singular = np.moveaxis(kdata_singular, -1, 1)
    kdata_singular = kdata_singular.reshape(nb_channels,L0, -1,npoint)
    # kdata_singular*=1000000


    kdata_singular_bart=np.moveaxis(kdata_singular,-1,1)
    kdata_singular_bart=np.moveaxis(kdata_singular,0,-1)[None,...,None]
    kdata_singular_bart=np.moveaxis(kdata_singular_bart,1,-1)
    kdata_singular_bart=np.moveaxis(kdata_singular_bart,2,1)
    volumes_singular_allbins=[]
    for gr in tqdm(range(nbins)):
        print("Building singular volumes for bin {}".format(gr))
        curr_weights_bart=all_weights[gr]

        curr_weights_bart=(curr_weights_bart.squeeze().flatten()>0)*1
        traj_python_bart_gr=traj_python_bart[:,:,curr_weights_bart>0]
        kdata_bart_gr=kdata_singular_bart[:,:,curr_weights_bart>0]
        # 'pics -i1 -e -S -d3 -b 5 -RL:$(bart bitmask 5):$(bart bitmask 0 1 2):0.01 -t'
        if dens_adj:
            tile_shape=list(kdata_bart_gr.shape[1:])
            tile_shape[0]=1
            tile_shape=tuple(tile_shape)
            density_curr=np.expand_dims(density,tuple(range(1,len(tile_shape))))
            density_curr=np.tile(density_curr,reps=tile_shape)
            density_curr=density_curr[None,:]
            cfl.writecfl("density",np.sqrt(density_curr))
            volume_singular_gr=bart(1,bart_command+" -p density -t",traj_python_bart_gr,kdata_bart_gr,b1_bart)
        else:
            volume_singular_gr=bart(1,bart_command+" -t",traj_python_bart_gr,kdata_bart_gr,b1_bart)
        volumes_singular_allbins.append(volume_singular_gr.squeeze())

        if log:
            np.save(filename_volumes_log.format(gr),np.moveaxis(volume_singular_gr.squeeze(),-1,0))

    volumes_singular_allbins=np.array(volumes_singular_allbins)
    volumes_singular_allbins=np.moveaxis(volumes_singular_allbins,-1,1)
    np.save(filename_volumes,volumes_singular_allbins)

    return


@ma.machine()
@ma.parameter("bart_command", str, default="nlinv -w1. -d5 -i9", description="bart pics command")
@ma.parameter("filename_kdata", str, default=None, description="MRF raw data")
@ma.parameter("filename_weights", str, default=None, description="Motion bin weights")
@ma.parameter("filename_phi", str, default=None, description="MRF temporal basis components")
@ma.parameter("filename_seqParams", str, default=None, description="Seq params")
@ma.parameter("filename_dcomp", str, default=None, description="Seq params")
@ma.parameter("dictfile", str, default=None, description="MRF dictionary file for temporal basis")
@ma.parameter("L0", int, default=10, description="Number of retained temporal basis functions")
@ma.parameter("n_comp", int, default=None, description="Virtual coils components to load b1 and pca file")
@ma.parameter("useGPU", bool, default=True, description="Use GPU")
@ma.parameter("log", bool, default=True, description="log bin results")
@ma.parameter("gating_only", bool, default=False, description="Use weights only for gating")
def build_volumes_singular_allbins_3D_BART_inv(bart_command,filename_kdata, filename_weights,filename_dcomp,filename_phi,filename_seqParams,dictfile,L0,n_comp,useGPU,gating_only,log):
    '''
    Build singular volumes for MRF for all motion bins
    Output shape nb_motion_bins x L0 x nz x nx x ny
    '''
    filename_volumes = filename_kdata.split("_kdata.npy")[0] + "_volumes_singular_allbins_bart_nlinv.npy"
    filename_volumes_log = filename_kdata.split("_kdata.npy")[0] + "_volumes_singular_allbins_bart_nlinv_gr{}.npy"
    print("Loading Kdata")
    print(filename_kdata)
    kdata_all_channels_all_slices = np.load(filename_kdata)
    print(filename_kdata)
    kdata_all_channels_all_slices=kdata_all_channels_all_slices/np.max(np.abs(kdata_all_channels_all_slices))*2000
    print(np.linalg.norm(kdata_all_channels_all_slices))

    nb_channels,nb_segments,nb_part,npoint=kdata_all_channels_all_slices.shape

    if filename_seqParams is not None:
        file = open(filename_seqParams, "rb")
        dico_seqParams = pickle.load(file)
        file.close()

        use_navigator_dll = dico_seqParams["use_navigator_dll"]

        if "use_kushball_dll" in dico_seqParams.keys():
            use_kushball_dll=dico_seqParams["use_kushball_dll"]
        else:
            use_kushball_dll=False


        if use_kushball_dll:
            meas_sampling_mode = dico_seqParams["alFree"][16]
        elif (use_navigator_dll):
            meas_sampling_mode = dico_seqParams["alFree"][15]
        else:
            meas_sampling_mode = dico_seqParams["alFree"][12]

        print(meas_sampling_mode)



        undersampling_factor = dico_seqParams["alFree"][9]

        nb_slices=int(dico_seqParams["nb_part"])


        if meas_sampling_mode == 1:
            incoherent = False
            mode = None
        elif meas_sampling_mode == 2:
            incoherent = True
            mode = "old"
        elif meas_sampling_mode == 3:
            incoherent = True
            mode = "new"

        elif meas_sampling_mode == 4:
            incoherent = True
            mode = "Kushball"


        npoint_image=int(npoint/2)
    
        if mode =="Kushball":
            print("Assuming isotropic image size for kushball sampling")
            image_size=(npoint_image,npoint_image,npoint_image)
        else:
            image_size=(nb_slices,npoint_image,npoint_image)




    else:
        incoherent=False
        mode="old"
        undersampling_factor=1
        nb_slices=nb_part
        image_size=(100,100,100)


        



    if filename_weights is None:
        filename_weights = str.split(filename_kdata, "_kdata.npy")[0] + "_weights.npy"

    if ((filename_phi is None) and (dictfile is None)):
        raise ValueError('Either dictfile or filename_phi should be provided for temporal projection')

    if dictfile is not None:
        filename_phi = str.split(dictfile, ".dict")[0] + "_phi_L0_{}.npy".format(L0)



    all_weights = np.load(filename_weights)
    nbins=all_weights.shape[0]

    if gating_only:
        print("Using weights for gating only")
        all_weights=(all_weights>0)*1

    if filename_dcomp is not None:
        print("Weighing by density compensation file")
        dcomp=np.load(filename_dcomp)
        print(dcomp.shape)
        dcomp[:,:,:,:,-1]=dcomp[:,:,:,:,-2]
        dcomp[:,:,:,:,0]=dcomp[:,:,:,:,1]
        
        all_weights=all_weights*dcomp


    # if filename_phi not in os.listdir():
    #     phi = build_phi(dictfile, L0)
    # else:
    #     phi = np.load(filename_phi)
    #     L0=phi.shape[0]

    mrf_dict = load_pickle(dictfile)
    if "phi" not in mrf_dict.keys():
        mrf_dict=add_temporal_basis(mrf_dict,L0)
        save_pickle(mrf_dict,mrf_dict)
    phi=mrf_dict["phi"]

    print("Kdata shape {}".format(kdata_all_channels_all_slices.shape))
    print("Weights shape {}".format(all_weights.shape))
    print("phi shape {}".format(phi.shape))

    print("nb_part {}".format(nb_part))
    print("image_size {}".format(image_size))

    radial_traj=Radial3D(total_nspokes=nb_segments,undersampling_factor=1,npoint=npoint,nb_slices=nb_part,incoherent=incoherent,mode=mode)
    
        
    traj_python = radial_traj.get_traj()
    traj_python = traj_python.reshape(nb_segments, nb_part, -1, 3)
    traj_python = traj_python.T
    traj_python = np.moveaxis(traj_python, -1, -2)

    traj_python[0] = traj_python[0] / np.max(np.abs(traj_python[0])) * int(image_size[0]/2)
    traj_python[1] = traj_python[1] / np.max(np.abs(traj_python[1])) * int(
        image_size[1]/2)
    traj_python[2] = traj_python[2] / np.max(np.abs(traj_python[2])) * int(
        image_size[2] / 2)

    traj_python_bart = traj_python.reshape(3, npoint, -1)
    
    window=8
    ntimesteps=int(nb_segments/window)
    

    kdata_singular = np.zeros((nb_channels,ntimesteps, nb_part*npoint*window) + (L0,), dtype="complex64")
    data=kdata_all_channels_all_slices.reshape(nb_channels,ntimesteps,-1)

    for ch in tqdm(range(nb_channels)):
        for ts in tqdm(range(ntimesteps)):
            kdata_singular[ch, ts, :, :] = data[ch, ts, :, None] @ (phi[:L0].conj().T[ts][None, :])

    print(np.linalg.norm(kdata_singular))

    kdata_singular = np.moveaxis(kdata_singular, -1, 1)
    kdata_singular = kdata_singular.reshape(nb_channels,L0, -1,npoint)
    # kdata_singular*=1000000


    kdata_singular_bart=np.moveaxis(kdata_singular,-1,1)
    kdata_singular_bart=np.moveaxis(kdata_singular,0,-1)[None,...,None]
    kdata_singular_bart=np.moveaxis(kdata_singular_bart,1,-1)
    kdata_singular_bart=np.moveaxis(kdata_singular_bart,2,1)
    volumes_singular_allbins=[]
    for gr in tqdm(range(nbins)):
        print("Building singular volumes for bin {}".format(gr))
        curr_weights_bart=all_weights[gr]

        curr_weights_bart=(curr_weights_bart.squeeze().flatten()>0)*1
        traj_python_bart_gr=traj_python_bart[:,:,curr_weights_bart>0]
        kdata_bart_gr=kdata_singular_bart[:,:,curr_weights_bart>0]
        # 'pics -i1 -e -S -d3 -b 5 -RL:$(bart bitmask 5):$(bart bitmask 0 1 2):0.01 -t'
        
        volume_singular_gr=bart(1,bart_command+" -t",traj_python_bart_gr,kdata_bart_gr)
        volumes_singular_allbins.append(volume_singular_gr.squeeze())

        if log:
            np.save(filename_volumes_log.format(gr),np.moveaxis(volume_singular_gr.squeeze(),-1,0))

    volumes_singular_allbins=np.array(volumes_singular_allbins)
    volumes_singular_allbins=np.moveaxis(volumes_singular_allbins,-1,1)
    np.save(filename_volumes,volumes_singular_allbins)

    return

@ma.machine()
@ma.parameter("bart_command", str, default=None, description="bart pics command")
@ma.parameter("filename_kdata", str, default=None, description="MRF raw data")
@ma.parameter("filename_b1", str, default=None, description="B1")
@ma.parameter("filename_weights", str, default=None, description="Motion bin weights")
@ma.parameter("filename_phi", str, default=None, description="MRF temporal basis components")
@ma.parameter("filename_seqParams", str, default=None, description="Seq params")
@ma.parameter("filename_dcomp", str, default=None, description="Seq params")
@ma.parameter("dictfile", str, default=None, description="MRF dictionary file for temporal basis")
@ma.parameter("L0", int, default=10, description="Number of retained temporal basis functions")
@ma.parameter("n_comp", int, default=None, description="Virtual coils components to load b1 and pca file")
@ma.parameter("useGPU", bool, default=True, description="Use GPU")
@ma.parameter("log", bool, default=True, description="log bin results")
@ma.parameter("dens_adj", bool, default=False, description="Radial density adjustment")
@ma.parameter("nb_rep_center_part", int, default=1, description="Number of center partition repetition")
@ma.parameter("gating_only", bool, default=False, description="Use weights only for gating")
def build_volumes_singular_allbins_3D_BART_v2(bart_command,filename_kdata, filename_b1, filename_weights,filename_dcomp,filename_phi,filename_seqParams,dictfile,L0,n_comp,useGPU,dens_adj,nb_rep_center_part,gating_only,log):
    '''
    Build singular volumes for MRF for all motion bins
    Output shape nb_motion_bins x L0 x nz x nx x ny
    '''
    filename_volumes = filename_kdata.split("_kdata.npy")[0] + "_volumes_singular_allbins_bart.npy"
    filename_volumes_log = filename_kdata.split("_kdata.npy")[0] + "_volumes_singular_allbins_bart_gr{}.npy"
    print("Loading Kdata")
    print(filename_kdata)
    kdata_all_channels_all_slices = np.load(filename_kdata)
    print(filename_kdata)
    print(np.linalg.norm(kdata_all_channels_all_slices))

    nb_channels,nb_segments,nb_part,npoint=kdata_all_channels_all_slices.shape

    if filename_seqParams is not None:
        file = open(filename_seqParams, "rb")
        dico_seqParams = pickle.load(file)
        file.close()

        use_navigator_dll = dico_seqParams["use_navigator_dll"]

        if "use_kushball_dll" in dico_seqParams.keys():
            use_kushball_dll=dico_seqParams["use_kushball_dll"]
        else:
            use_kushball_dll=False


        if use_kushball_dll:
            meas_sampling_mode = dico_seqParams["alFree"][16]
        elif (use_navigator_dll):
            meas_sampling_mode = dico_seqParams["alFree"][15]
        else:
            meas_sampling_mode = dico_seqParams["alFree"][12]

        print(meas_sampling_mode)



        undersampling_factor = dico_seqParams["alFree"][9]

        nb_slices=int(dico_seqParams["nb_part"])


        if meas_sampling_mode == 1:
            incoherent = False
            mode = None
        elif meas_sampling_mode == 2:
            incoherent = True
            mode = "old"
        elif meas_sampling_mode == 3:
            incoherent = True
            mode = "new"

        elif meas_sampling_mode == 4:
            incoherent = True
            mode = "Kushball"



    else:
        incoherent=False
        mode="old"
        undersampling_factor=1
        nb_slices=nb_part
        image_size=(100,100,100)




    if dens_adj:


        npoint = kdata_all_channels_all_slices.shape[-1]
        if mode=="Kushball": 
            density = np.abs(np.linspace(-1, 1, npoint))**2
        else:
            density=np.abs(np.linspace(-1, 1, npoint))

        

        # bart_command=str.replace(bart_command,"-t","-p density -t")
        # print(bart_command)
        


    if ((filename_b1 is None))and(n_comp is None):
        raise ValueError('n_comp should be provided when B1 or PCA files are missing')

    if filename_b1 is None:
        filename_b1=str.split(filename_kdata, "_kdata.npy")[0] + "_b12Dplus1_{}.npy".format(n_comp)

    if filename_weights is None:
        filename_weights = str.split(filename_kdata, "_kdata.npy")[0] + "_weights.npy"

    if ((filename_phi is None) and (dictfile is None)):
        raise ValueError('Either dictfile or filename_phi should be provided for temporal projection')

    if dictfile is not None:
        filename_u = str.split(dictfile, ".dict")[0] + "_u_bart.npy"

    b1_all_slices_2Dplus1_pca = np.load(filename_b1)
    
    image_size=b1_all_slices_2Dplus1_pca.shape[1:]

    b1_bart=np.moveaxis(b1_all_slices_2Dplus1_pca,0,-1)
    b1_bart=np.moveaxis(b1_bart,0,-2)

    all_weights = np.load(filename_weights)
    nbins=all_weights.shape[0]

    if gating_only:
        print("Using weights for gating only")
        all_weights=(all_weights>0)*1

    if filename_dcomp is not None:
        print("Weighing by density compensation file")
        dcomp=np.load(filename_dcomp)
        print(dcomp.shape)
        dcomp[:,:,:,:,-1]=dcomp[:,:,:,:,-2]
        dcomp[:,:,:,:,0]=dcomp[:,:,:,:,1]
        
        all_weights=all_weights*dcomp


    if filename_phi not in os.listdir():
        u = build_basis_bart(dictfile)
        print(u.shape)
        np.save(filename_u,u)
    else:
        u = np.load(filename_u)

    basis=u[:L0,:].T
    basis=basis[None,None,None,None,None,:,:]

    cfl.writecfl("basis",basis)

    print("Kdata shape {}".format(kdata_all_channels_all_slices.shape))
    print("Weights shape {}".format(all_weights.shape))
    print("basis shape {}".format(basis.shape))
    print("B1 shape {}".format(b1_all_slices_2Dplus1_pca.shape))

    print("nb_part {}".format(nb_part))
    print("image_size {}".format(image_size))

    radial_traj=Radial3D(total_nspokes=nb_segments,undersampling_factor=1,npoint=npoint,nb_slices=nb_part,incoherent=incoherent,mode=mode)
    
        
    traj_python = radial_traj.get_traj()
    window=8
    ntimesteps=int(nb_segments/window)
    traj_python = traj_python.reshape(ntimesteps,window,nb_part, -1, 3)
    traj_python = traj_python.T
    traj_python = np.moveaxis(traj_python, -2, -3)

    traj_python[0] = traj_python[0] / np.max(np.abs(traj_python[0])) * int(image_size[0]/2)
    traj_python[1] = traj_python[1] / np.max(np.abs(traj_python[1])) * int(
        image_size[1]/2)
    traj_python[2] = traj_python[2] / np.max(np.abs(traj_python[2])) * int(
        image_size[2] / 2)


    traj_python_bart = traj_python.reshape(3, npoint, window*nb_part,1,1,ntimesteps)
    print(traj_python_bart.shape)
    
    
    

    # kdata_singular = np.zeros((nb_channels,ntimesteps, nb_part*npoint*window) + (L0,), dtype="complex64")
    # kdata_all_channels_all_slices=kdata_all_channels_all_slices.reshape(nb_channels,nb_segments,nb_partnpoint)

    
    volumes_singular_allbins=[]
    for gr in tqdm(range(nbins)):
        print("Building singular volumes for bin {}".format(gr))
        curr_weights_bart=all_weights[gr]
        # traj_python_bart_gr=traj_python_bart[:,:,curr_weights_bart>0]
        
        print(curr_weights_bart.shape)
        print(kdata_all_channels_all_slices.shape)
        kdata_curr_gr=kdata_all_channels_all_slices*((curr_weights_bart>0)*1)
        kdata_curr_gr=kdata_curr_gr.reshape(nb_channels,ntimesteps,window,nb_part,npoint)
        kdata_curr_gr=kdata_curr_gr.T
        kdata_curr_gr=np.moveaxis(kdata_curr_gr,1,2)
        kdata_curr_gr=kdata_curr_gr.reshape(1,npoint,window*nb_part,nb_channels,1,ntimesteps)

        print(kdata_curr_gr.shape)

        volume_singular_gr=bart(1,bart_command+" -B basis -t",traj_python_bart,kdata_curr_gr,b1_bart)
        volumes_singular_allbins.append(volume_singular_gr.squeeze())


    volumes_singular_allbins=np.array(volumes_singular_allbins)
    volumes_singular_allbins=np.moveaxis(volumes_singular_allbins,-1,1)
    np.save(filename_volumes,volumes_singular_allbins)

    return

@ma.machine()
@ma.parameter("filename_kdata", str, default=None, description="MRF raw data")
@ma.parameter("filename_seqParams", str, default=None, description="Seq params")
@ma.parameter("filename_b1", str, default=None, description="B1")
@ma.parameter("filename_pca", str, default=None, description="filename for storing coil compression components")
@ma.parameter("filename_weights", str, default=None, description="Weights file to simulate undersampling from binning")
@ma.parameter("dictfile", str, default=None, description="MRF dictionary file for temporal basis (.pkl)")
@ma.parameter("L0", int, default=6, description="Number of retained temporal basis functions")
@ma.parameter("n_comp", int, default=None, description="Virtual coils components to load b1 and pca file")
@ma.parameter("nb_rep_center_part", int, default=1, description="Number of center partition repetition")
@ma.parameter("useGPU", bool, default=True, description="Use GPU")
@ma.parameter("full_volume", bool, default=False, description="Build full volume")
@ma.parameter("in_phase", bool, default=False, description="MRF T1-FF : Select in phase spokes from original MRF sequence")
@ma.parameter("out_phase", bool, default=False, description="MRF T1-FF : Select out of phase spokes from original MRF sequence")

def build_volumes_singular(filename_kdata, filename_b1, filename_pca,dictfile,L0,n_comp,nb_rep_center_part,useGPU,filename_weights,full_volume,in_phase,out_phase,filename_seqParams):
    '''
    Build singular volumes for MRF (no binning)
    Output shape nb_motion_bins x L0 x nz x nx x ny
    '''

    print("Using GPU: {}".format(useGPU))

    if filename_seqParams is None:
        filename_seqParams =filename_kdata.split("_kdata.npy")[0] + "_seqParams.pkl"

    file = open(filename_seqParams, "rb")
    dico_seqParams = pickle.load(file)
    file.close()

    use_navigator_dll = dico_seqParams["use_navigator_dll"]

    print(dico_seqParams)

    nb_gating_spokes=dico_seqParams["alFree"][6]
    print(nb_gating_spokes)

    if "use_kushball_dll" in dico_seqParams.keys(): 
        use_kushball_dll=dico_seqParams["use_kushball_dll"]
    else:
        use_kushball_dll=False


    if use_kushball_dll:
        meas_sampling_mode = dico_seqParams["alFree"][16]
    elif (use_navigator_dll):
        meas_sampling_mode = dico_seqParams["alFree"][15]
    else:
        meas_sampling_mode = dico_seqParams["alFree"][12]


    print(meas_sampling_mode)



    undersampling_factor = dico_seqParams["alFree"][9]

    nb_segments = dico_seqParams["alFree"][4]
    nb_slices = int(dico_seqParams["nb_part"])

    if meas_sampling_mode == 1:
        incoherent = False
        mode = None
    elif meas_sampling_mode == 2:
        incoherent = True
        mode = "old"
    elif meas_sampling_mode == 3:
        incoherent = True
        mode = "new"

    elif meas_sampling_mode == 4:
        incoherent = True
        mode = "Kushball"

    print(incoherent)

    filename_volumes = filename_kdata.split("_kdata.npy")[0] + "_volumes_singular.npy"

    if full_volume:
        filename_volumes = filename_kdata.split("_kdata.npy")[0] + "_volume_full.npy"
        if in_phase:
            filename_volumes = filename_volumes.split("_volume_full.npy")[0] + "_volume_full_ip.npy"
        elif out_phase:
            filename_volumes = filename_volumes.split("_volume_full.npy")[0] + "_volume_full_oop.npy"
    print("Loading Kdata")
    kdata_all_channels_all_slices = np.load(filename_kdata)

    nb_channels,nb_segments,nb_part,npoint=kdata_all_channels_all_slices.shape


    if ((filename_b1 is None)or(filename_pca is None))and(n_comp is None)and(not(incoherent)):
        raise ValueError('n_comp should be provided when B1 or PCA files are missing for stack of stars reco')


    if filename_b1 is None:
        if (not(incoherent)):
            filename_b1=str.split(filename_kdata, "_kdata.npy")[0] + "_b12Dplus1_{}.npy".format(n_comp)
        else:
            filename_b1 = str.split(filename_kdata, "_kdata.npy")[0] + "_b1.npy".format(n_comp)

    if filename_pca is None:
        filename_pca = str.split(filename_kdata, "_kdata.npy")[0] + "_virtualcoils_{}.pkl".format(n_comp)


    b1_all_slices_2Dplus1_pca = np.load(filename_b1)


    if full_volume:
        L0=1
        window=8
        phi=np.ones((1,1))
    else:
        # if filename_phi not in os.listdir():
        #     print("Build singular basis from dictionary {}".format(dictfile))
        #     phi=build_phi(dictfile,L0)
        # else:
        #     phi = np.load(filename_phi)

        mrf_dict = load_pickle(dictfile)
        if "phi" not in mrf_dict.keys():
            mrf_dict=add_temporal_basis(mrf_dict,L0)
            save_pickle(mrf_dict,mrf_dict)
        phi=mrf_dict["phi"]


    if filename_weights is not None:
        print("Applying weights mask to k-space")
        all_weights=np.load(filename_weights)
        all_weights=(np.sum(all_weights,axis=0)>0)*1
        kdata_all_channels_all_slices=kdata_all_channels_all_slices*all_weights


    print("Kdata shape {}".format(kdata_all_channels_all_slices.shape))
    #
    print("phi shape {}".format(phi.shape))

    print("Building Singular Volumes")
    if not(incoherent):

        if in_phase:
            # selected_spokes=np.r_[300:800,1200:1400]
            selected_spokes=np.r_[300:554,820:1023]
            #selected_spokes=np.r_[280:580]
        elif out_phase:
            selected_spokes = np.r_[554:820]
        else:
            selected_spokes=None

        print("Stack of stars - using 2D+1 reconstruction")
        file = open(filename_pca, "rb")
        pca_dict = pickle.load(file)
        file.close()
        print("virtual coils components shape {}".format(pca_dict[0].components_.shape))

        volumes_allbins=build_volume_singular_2Dplus1_cc(kdata_all_channels_all_slices, b1_all_slices_2Dplus1_pca, pca_dict,
                                                1, phi, L0,nb_rep_center_part=nb_rep_center_part,useGPU=useGPU,selected_spokes=selected_spokes)
    
    elif meas_sampling_mode==4:
        print("Kushball reconstruction")
        image_size=(nb_slices,int(npoint/2),int(npoint/2))
        print(image_size)
        print(nb_part)
        radial_traj=Radial3D(total_nspokes=nb_segments,undersampling_factor=1,npoint=npoint,nb_slices=nb_part,mode="Kushball")
        volumes_allbins=build_volume_singular_3D(kdata_all_channels_all_slices, b1_all_slices_2Dplus1_pca, radial_traj,1,phi,L0,useGPU,nb_rep_center_part,image_size=image_size)

    else:
        print("Non stack of stars - using 3D reconstruction")
        cond_us = np.zeros((nb_slices, nb_segments))

        cond_us = cond_us.reshape((nb_slices, -1, 8))

        curr_start = 0
        for sl in range(nb_slices):
            cond_us[sl, curr_start::undersampling_factor, :] = 1
            curr_start = curr_start + 1
            curr_start = curr_start % undersampling_factor

        cond_us = cond_us.flatten()
        included_spokes = cond_us
        included_spokes = (included_spokes > 0)

        radial_traj_allspokes = Radial3D(total_nspokes=nb_segments, undersampling_factor=1, npoint=npoint,
                                         nb_slices=nb_slices, incoherent=incoherent, mode=mode)

        from utils_reco import correct_mvt_kdata_zero_filled
        weights, retained_timesteps = correct_mvt_kdata_zero_filled(radial_traj_allspokes, included_spokes, 1)

        weights = weights.reshape(1, -1, 8, nb_slices)
        import math
        nb_rep = math.ceil(nb_slices / undersampling_factor)
        weights_us = np.zeros(shape=(1, 175, 8, nb_rep), dtype=weights.dtype)

        shift = 0

        for sl in range(nb_slices):
            if int(sl / undersampling_factor) < nb_rep:
                weights_us[:, shift::undersampling_factor, :, int(sl / undersampling_factor)] = weights[:,
                                                                                                shift::undersampling_factor,
                                                                                                :, sl]
                shift += 1
                shift = shift % (undersampling_factor)
            else:
                continue

        weights_us = weights_us.reshape(1, -1, nb_rep)
        weights_us = weights_us[..., None]

        
        radial_traj=Radial3D(total_nspokes=nb_segments,npoint=npoint,nb_slices=nb_slices,undersampling_factor=undersampling_factor,incoherent=incoherent,mode=mode,nb_rep_center_part=nb_rep_center_part)
        
        volumes_allbins=build_volume_singular_3D(kdata_all_channels_all_slices, b1_all_slices_2Dplus1_pca, radial_traj,weights_us,phi,L0,useGPU,nb_rep_center_part)
    np.save(filename_volumes, volumes_allbins)

    return



@ma.machine()
@ma.parameter("filename_kdata", str, default=None, description="MRF raw data")
@ma.parameter("nbins", int, default=5, description="Number of bins")
@ma.parameter("nkept", int, default=4, description="Number of bins kept")
@ma.parameter("nb_gating_spokes", int, default=50, description="Gating spokes count")
@ma.parameter("equal_spoke_per_bin", bool, default=False, description="Equal number of spokes per bin")
def generate_random_weights(filename_kdata, nbins,nkept,nb_gating_spokes,equal_spoke_per_bin):
    '''
    Build singular volumes for MRF (no binning)
    Output shape nb_motion_bins x L0 x nz x nx x ny
    '''
    filename_weights = filename_kdata.split("_kdata.npy")[0] + "_weights.npy"
    print("Loading Kdata")
    kdata_all_channels_all_slices = np.load(filename_kdata)

    nb_allspokes=kdata_all_channels_all_slices.shape[1]
    nb_slices=kdata_all_channels_all_slices.shape[2]

    all_weights=[]

    displacement=np.zeros((nb_slices,nb_gating_spokes))
    for sl in range(nb_slices):
        phase=np.random.uniform()*np.pi-np.pi/2
        amplitude=np.random.uniform()*0.2+0.9
        frequency=(np.random.uniform()*0.5+1)/nb_gating_spokes
        displacement[sl]=amplitude*np.sin(2*np.pi*np.arange(nb_gating_spokes)*frequency+phase)

    displacement=displacement.flatten()

    displacement_for_binning = displacement
    if not(equal_spoke_per_bin):
        max_bin = np.max(displacement_for_binning)
        min_bin = np.min(displacement_for_binning)
        bin_width = (max_bin-min_bin)/(nbins)
        min_bin = np.min(displacement_for_binning) 
        bins = np.arange(min_bin, max_bin + bin_width, bin_width)
        
        retained_categories=list(range(nbins-nkept+1,nbins+1))
    
    else:
        disp_sorted_index = np.argsort(displacement_for_binning)
        count_disp = len(disp_sorted_index)
        disp_width = int(count_disp / nbins)
        bins = []
        for j in range(1, nbins):
            bins.append(np.sort(displacement_for_binning)[j * disp_width])
        retained_categories=list(range(nbins-nkept,nbins))

    categories = np.digitize(displacement_for_binning, bins)
    df_cat = pd.DataFrame(data=np.array([displacement_for_binning, categories]).T, columns=["displacement", "cat"])
    df_groups = df_cat.groupby("cat").count()
    
    print(df_groups)
    print(retained_categories)
    groups=[]
    for cat in retained_categories:
        groups.append(categories==cat)

    spoke_groups = np.argmin(np.abs(
            np.arange(0, nb_allspokes * nb_slices, 1).reshape(-1, 1) - np.arange(0, nb_allspokes * nb_slices,
                                                                              nb_allspokes / nb_gating_spokes).reshape(1,
                                                                                                                      -1)),
            axis=-1)

    spoke_groups = spoke_groups.reshape(nb_slices, nb_allspokes)
    spoke_groups[:-1, -int(nb_allspokes / nb_gating_spokes / 2) + 1:] = spoke_groups[:-1, -int(
            nb_allspokes / nb_gating_spokes / 2) + 1:] - 1  # adjustment for change of partition
    spoke_groups = spoke_groups.flatten()
    dico_traj_retained={}
    radial_traj=Radial3D(total_nspokes=nb_allspokes,undersampling_factor=1,npoint=800,nb_slices=nb_slices*1,incoherent=False,mode="old",nspoke_per_z_encoding=8)

    for j,g in enumerate(groups):
        retained_nav_spokes_index = np.argwhere(g).flatten()
        included_spokes = np.array([s in retained_nav_spokes_index for s in spoke_groups])
        included_spokes[::int(nb_allspokes/nb_gating_spokes)]=False
        # included_spokes=included_spokes.reshape(nb_slices,nb_allspokes)
        # included_spokes=np.moveaxis(included_spokes,0,1)[None]
        
        weights, retained_timesteps = correct_mvt_kdata_zero_filled(radial_traj, included_spokes, 1)
        dico_traj_retained[j]=weights
        #all_weights.append(included_spokes)
    
    
    for gr in dico_traj_retained.keys():
        all_weights.append(np.expand_dims(dico_traj_retained[gr],axis=-1))
    all_weights=np.array(all_weights)
    

    print(all_weights.shape)

    np.save(filename_weights, all_weights)

    return


@ma.machine()
@ma.parameter("filename_volume", str, default=None, description="MRF raw data")
@ma.parameter("filename_b1", str, default=None, description="B1")
@ma.parameter("filename_weights", str, default=None, description="Motion bin weights")
@ma.parameter("filename_seqParams", str, default=None, description="Undersampling weights")
@ma.parameter("mu", float, default=1, description="Gradient step size")
@ma.parameter("mu_TV", float, default=1, description="Spatial Regularization")
@ma.parameter("weights_TV", [float,float,float], default=[1.0,0.2,0.2], description="Spatial Regularization Weights")
@ma.parameter("lambda_wav", float, default=0.5e-5, description="Lambda wavelet")
@ma.parameter("lambda_LLR", float, default=0.0005, description="Lambda LLR")
@ma.parameter("mu_bins", float, default=None, description="Interbin regularization")
@ma.parameter("niter", int, default=None, description="Number of iterations")
@ma.parameter("suffix", str, default="", description="Suffix")
@ma.parameter("gamma", float, default=None, description="Gamma Correction")
@ma.parameter("isgamma3D", bool, default=False, description="Do gamma intensity correction on the whole volume")
@ma.parameter("gating_only", bool, default=False, description="Use weights only for gating")
@ma.parameter("dens_adj", bool, default=True, description="Use Radial density adjustment")
@ma.parameter("nb_rep_center_part", int, default=1, description="Number of center partition repetition")
@ma.parameter("select_first_rep", bool, default=False, description="Select firt repetition of central partition only")
@ma.parameter("use_wavelet", bool, default=False, description="Wavelet regularization instead of TV")
@ma.parameter("use_proximal_TV", bool, default=False, description="Proximal gradient (FISTA) instead of gradient descent")
@ma.parameter("us", int, default=1, description="Undersampling")
@ma.parameter("use_LLR", bool, default=False, description="LLR regularization instead of TV")
@ma.parameter("block", str, default="4,10,10", description="Block size for LLR regularization")
@ma.parameter("axis", int, default=None, description="Gamma correction axis - default 0")
def build_volumes_iterative_allbins(filename_volume,filename_b1,filename_weights,filename_seqParams,mu,mu_TV,mu_bins,niter,gamma,suffix,gating_only,dens_adj,nb_rep_center_part,select_first_rep,use_proximal_TV,use_wavelet,lambda_wav,us,use_LLR,lambda_LLR,block,axis,isgamma3D,weights_TV):
    filename_target=filename_volume.split("_volumes_allbins.npy")[0] + "_volumes_allbins_denoised{}.npy".format(suffix)

    if gamma is not None:
        filename_target=filename_target.split(".npy")[0]+"_gamma_{}.npy".format(str(gamma).replace(".","_"))
    
    weights_TV/=np.sum(weights_TV)
    print("Loading Volumes")
    volumes=np.load(filename_volume)
    #To fit the input expected by the undersampling function with L0=1 in our case
    if volumes.ndim==4:
        volumes=np.expand_dims(volumes,axis=1)

    volumes=volumes.astype("complex64")

    if filename_weights is None:
        filename_weights = (filename_volume.split("_volumes_allbins.npy")[0] + "_weights.npy").replace("_no_densadj","")
    if gating_only:
        filename_weights=str.replace(filename_weights,"_no_dcomp","")

    

    b1_all_slices_2Dplus1_pca=np.load(filename_b1)
    all_weights=np.load(filename_weights)

    if gating_only:
        all_weights=(all_weights>0)*1

    if nb_rep_center_part>1:
        all_weights=weights_aggregate_center_part(all_weights,nb_rep_center_part,select_first_rep)


    nb_part = all_weights.shape[3]

    if filename_seqParams is not None:
        file = open(filename_seqParams, "rb")
        dico_seqParams = pickle.load(file)
        file.close()

        use_navigator_dll = dico_seqParams["use_navigator_dll"]
        nb_gating_spokes=dico_seqParams["alFree"][6]


        if "use_kushball_dll" in dico_seqParams.keys():
            use_kushball_dll=dico_seqParams["use_kushball_dll"]
        else:
            use_kushball_dll=False


        if use_kushball_dll:
            meas_sampling_mode = dico_seqParams["alFree"][16]
        elif (use_navigator_dll):
            meas_sampling_mode = dico_seqParams["alFree"][15]
        else:
            meas_sampling_mode = dico_seqParams["alFree"][12]
        
        if meas_sampling_mode == 1:
            incoherent = False
            mode = None
        elif meas_sampling_mode == 2:
            incoherent = True
            mode = "old"
        elif meas_sampling_mode == 3:
            incoherent = True
            mode = "new"
        
        elif meas_sampling_mode == 4:
            incoherent = True
            mode = "Kushball"

        nb_slices=int(dico_seqParams["nb_part"])
    
    else:
        incoherent=False
        mode="old"
        nb_slices=nb_part

    

    if us >1:
        weights_us = np.zeros_like(all_weights)
        
        nspoke_per_part = 8
        weights_us = weights_us.reshape((weights_us.shape[0], 1, -1, nspoke_per_part, nb_slices, 1))


        curr_start = 0

        for sl in range(nb_slices):
            weights_us[:, :, curr_start::us, :, sl] = 1
            curr_start = curr_start + 1
            curr_start = curr_start % us

        weights_us=weights_us.reshape(all_weights.shape)
        all_weights *= weights_us

    print("Volumes shape {}".format(volumes.shape))
    print("Weights shape {}".format(all_weights.shape))

    nb_allspokes=all_weights.shape[2]
    npoint=2*volumes.shape[-1]
    nbins=volumes.shape[0]

    
    if not(incoherent):
        radial_traj = Radial(total_nspokes=nb_allspokes, npoint=npoint)

    # elif mode=="Kushball":
    #     radial_traj = Radial3D(total_nspokes=nb_allspokes,npoint=npoint,nb_slices=nb_slices,undersampling_factor=undersampling_factor,incoherent=incoherent,mode=mode)
    else:
        radial_traj = Radial3D(total_nspokes=nb_allspokes,npoint=npoint,nb_slices=nb_part,undersampling_factor=1,incoherent=incoherent,mode=mode)


    ntimesteps=all_weights.shape[1]


    volumes0=copy(volumes)
    volumes=mu*volumes0

    if use_wavelet:
        print("Wavelet Denoising")
        wav_level = 4
        wav_type="db4"

        lambd = lambda_wav

        alpha_denoised = []

        

        for gr in tqdm(range(nbins)):
            print("#################   Denoising Bin {}   #######################".format(gr))
            weights = all_weights[gr]
            vol_denoised_log=[volumes[gr]]
            coefs = pywt.wavedecn(volumes[gr], wav_type, level=wav_level, mode="periodization",axes=(1,2,3))
            u, slices = pywt.coeffs_to_array(coefs,axes=(1,2,3))
            u0 = u
            y = u
            t = 1
            u = pywt.threshold(y, lambd * mu)

            print("Non zero percentage: {} ".format(np.count_nonzero(u)/np.prod(u.shape)))

            vol_denoised_log.append(pywt.waverecn(pywt.array_to_coeffs(u, slices), wav_type, mode="periodization",axes=(1,2,3)))

            t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
            y = u
            t = t_next

            for i in range(niter):
                u_prev = u
                x = pywt.array_to_coeffs(y, slices)
                x = pywt.waverecn(x, wav_type, mode="periodization",axes=(1,2,3))

                if not(incoherent):
                    volumesi = undersampling_operator_singular_new(x, radial_traj,
                                                               b1_all_slices_2Dplus1_pca, weights=weights,
                                                               density_adj=dens_adj)
                    
                else:
                    volumesi = undersampling_operator_singular(x, radial_traj,
                                                            b1_all_slices_2Dplus1_pca, weights=weights,
                                                               density_adj=dens_adj)
                #volumesi = volumesi.squeeze()
                coefs = pywt.wavedecn(volumesi, wav_type, level=wav_level, mode="periodization",axes=(1,2,3))
                grad_y, slices = pywt.coeffs_to_array(coefs,axes=(1,2,3))
                grad = grad_y - u0
                y = y - mu * grad

                u = pywt.threshold(y, lambd * mu)
                print("Non zero percentage: {} ".format(np.count_nonzero(u)/np.prod(u.shape)))

                t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
                y = u + (t - 1) / t_next * (u - u_prev)
                t = t_next

                if (i%1==0):
                    vol_denoised_log.append(pywt.waverecn(pywt.array_to_coeffs(u, slices), wav_type, mode="periodization",axes=(1,2,3)))

            vol_denoised_log=np.array(vol_denoised_log)
            filename_target_intermediate=filename_volume.split("_volumes_allbins.npy")[0] + "_volume_denoised_gr{}{}.npy".format(gr,suffix)
            np.save(filename_target_intermediate,vol_denoised_log)

            alpha_denoised.append(u)

        alpha_denoised = np.array(alpha_denoised)
        volumes = [pywt.waverecn(pywt.array_to_coeffs(alpha, slices), wav_type, mode="periodization",axes=(1,2,3)) for alpha in
                            alpha_denoised]
        volumes = np.array(volumes)


    elif use_LLR:
        print("LLR denoising")
        blck = np.array(block.split(",")).astype(int)
        strd = blck
        lambd = lambda_LLR
        threshold = lambd * mu

        volumes_denoised = []

        for gr in tqdm(range(nbins)):
            print("#################   Denoising Bin {}   #######################".format(gr))
            weights = all_weights[gr]
            u = mu * volumes[gr]
            u0 = volumes[gr]
            y = u
            t = 1

            u = proj_LLR(u.squeeze(), strd, blck, threshold)

            t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
            y = u
            t = t_next
            for i in range(niter):
                u_prev = u
                if u.ndim == 3:
                    u = np.expand_dims(u, axis=0)

                if not(incoherent):
                    volumesi = undersampling_operator_singular_new(x, radial_traj,
                                                               b1_all_slices_2Dplus1_pca, weights=weights,
                                                               density_adj=dens_adj)
                    
                else:
                    volumesi = undersampling_operator_singular(x, radial_traj,
                                                            b1_all_slices_2Dplus1_pca, weights=weights,
                                                               density_adj=dens_adj)


                grad = volumesi - u0
                y = y - mu * grad

                u = proj_LLR(y.squeeze(), strd, blck, threshold)

                t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
                y = u + (t - 1) / t_next * (u - u_prev)
                t = t_next

            volumes_denoised.append(u)

        volumes = np.array(volumes_denoised)


    else:


        for i in tqdm(range(niter)):
            print("Correcting volumes for iteration {}".format(i))
            all_grad_norm=0
            for gr in tqdm(range(nbins)):

                if not(incoherent):
                    volumesi = undersampling_operator_singular_new(volumes[gr], radial_traj,
                                                               b1_all_slices_2Dplus1_pca, weights=all_weights[gr],
                                                               density_adj=dens_adj)
                    
                else:
                    volumesi = undersampling_operator_singular(volumes[gr], radial_traj,
                                                            b1_all_slices_2Dplus1_pca, weights=all_weights[gr],
                                                               density_adj=dens_adj)


                grad = volumesi - volumes0[gr]
                volumes[gr] = volumes[gr] - mu * grad



                if (mu_TV is not None)and(not(mu_TV==0)):
                    print("Applying TV regularization")

                    grad_norm=np.linalg.norm(grad)
                    all_grad_norm+=grad_norm**2
                    print("grad norm {}".format(grad_norm))
                    del grad
                    grad_TV=np.zeros_like(volumes[gr])
                    for ts in tqdm(range(ntimesteps)):
                        for ind_w, w in (enumerate(weights_TV)):
                            if w > 0:
                                grad_TV[ts] += (w * grad_J_TV(volumes[gr,ts], ind_w,is_weighted=False,shift=0))

                            #grad_TV_norm = np.linalg.norm(grad_TV, axis=0)
                    grad_TV_norm = np.linalg.norm(grad_TV)
                                # signals = matched_signals + mu * grad

                    print("grad_TV_norm {}".format(grad_TV_norm))

                    volumes[gr] -= mu * mu_TV * grad_norm / grad_TV_norm * grad_TV
                    del grad_TV
                    del grad_TV_norm

            all_grad_norm=np.sqrt(all_grad_norm)
            if (mu_bins is not None)and (not(mu_bins==0)):
                grad_TV_bins=grad_J_TV(volumes,0,is_weighted=False,shift=0)
                grad_TV_bins_norm = np.linalg.norm(grad_TV_bins)

                volumes -= mu * mu_bins * grad_TV_bins/grad_TV_bins_norm*all_grad_norm
                print("grad_TV_bins norm {}".format(grad_TV_bins_norm))
                del grad_TV_bins

            if (i%1==0):
                filename_target_intermediate=filename_volume.split("_volumes_allbins.npy")[0] + "_volumes_allbins_denoised_it{}{}.npy".format(i,suffix)
                np.save(filename_target_intermediate,volumes)

    volumes=np.squeeze(volumes)
    
    if gamma is not None:

        if isgamma3D:
            for gr in range(volumes.shape[0]):
                volumes[gr]=gamma_transform_3D(volumes[gr],gamma)
        
        else:
            for gr in range(volumes.shape[0]):
                volumes[gr]=gamma_transform(volumes[gr],gamma,axis=axis)



    np.save(filename_target,volumes)
    #volumes_allbins=build_volume_2Dplus1_cc_allbins(kdata_all_channels_all_slices,b1_all_slices_2Dplus1_pca,pca_dict,all_weights)
    #np.save(filename_volumes,volumes_allbins)

    return


@ma.machine()
@ma.parameter("filename_volume", str, default=None, description="MRF raw data")
@ma.parameter("filename_b1", str, default=None, description="B1")
@ma.parameter("filename_weights", str, default=None, description="Undersampling weights")
@ma.parameter("filename_seqParams", str, default=None, description="Undersampling weights")
@ma.parameter("mu", float, default=1, description="Gradient step size")
@ma.parameter("mu_TV", float, default=1, description="Spatial Regularization")
@ma.parameter("lambda_wav", float, default=0.5e-5, description="Lambda wavelet")
@ma.parameter("lambda_LLR", float, default=0.0005, description="Lambda LLR")
@ma.parameter("niter", int, default=None, description="Number of iterations")
@ma.parameter("suffix", str, default="", description="Suffix")
@ma.parameter("gamma", float, default=None, description="Gamma Correction")
@ma.parameter("dens_adj", bool, default=True, description="Use Radial density adjustment")
@ma.parameter("use_wavelet", bool, default=False, description="Wavelet regularization instead of TV")
@ma.parameter("use_LLR", bool, default=False, description="LLR regularization instead of TV")
@ma.parameter("block", str, default="2,10,10", description="Block size for LLR regularization")

def build_volumes_iterative(filename_volume, filename_b1,filename_weights,filename_seqParams, mu, mu_TV, niter, gamma,
                                    suffix, dens_adj, use_wavelet, lambda_wav, use_LLR, lambda_LLR, block):
    filename_target = filename_volume.split(".npy")[0] + "_denoised{}.npy".format(
        suffix)

    if gamma is not None:
        filename_target = filename_target.split(".npy")[0] + "_gamma_{}.npy".format(str(gamma).replace(".", "_"))

    print(filename_target)

    weights_TV = np.array([1.0, 0.2, 0.2])
    weights_TV /= np.sum(weights_TV)
    print("Loading Volumes")
    volumes = np.load(filename_volume)
    # To fit the input expected by the undersampling function with L0=1 in our case

    volumes = volumes.astype("complex64")


    b1_all_slices_2Dplus1_pca = np.load(filename_b1)
    incoherent=False #stack of stars by default

    if filename_seqParams is None:
        filename_seqParams =filename_kdata.split("_kdata.npy")[0] + "_seqParams.pkl"

    file = open(filename_seqParams, "rb")
    dico_seqParams = pickle.load(file)
    file.close()
    print(dico_seqParams)

    use_navigator_dll = dico_seqParams["use_navigator_dll"]

    if "use_kushball_dll" in dico_seqParams.keys():
        use_kushball_dll=dico_seqParams["use_kushball_dll"]
    else:
        use_kushball_dll=False


    if use_kushball_dll:
        meas_sampling_mode = dico_seqParams["alFree"][16]
    elif (use_navigator_dll):
        meas_sampling_mode = dico_seqParams["alFree"][15]
    else:
        meas_sampling_mode = dico_seqParams["alFree"][12]

    nb_gating_spokes=dico_seqParams["alFree"][6]
    # if (use_navigator_dll)or(nb_gating_spokes>0):
    #     meas_sampling_mode = dico_seqParams["alFree"][15]
    # else:
    #     meas_sampling_mode = dico_seqParams["alFree"][12]

    undersampling_factor = dico_seqParams["alFree"][9]

    nb_segments = dico_seqParams["alFree"][4]
    nb_slices = int(dico_seqParams["nb_part"])

    if meas_sampling_mode == 1:
        incoherent = False
        mode = None
    elif meas_sampling_mode == 2:
        incoherent = True
        mode = "old"
    elif meas_sampling_mode == 3:
        incoherent = True
        mode = "new"
        
    elif meas_sampling_mode == 4:
        incoherent = True
        mode = "Kushball"
        

    if filename_weights is not None:
        print("Applying weights mask to k-space")
        weights=np.load(filename_weights)
    else:
        weights=1



    print("Volumes shape {}".format(volumes.shape))

    npoint = 2 * volumes.shape[-1]
    nbins = volumes.shape[0]

    if not(incoherent):
        radial_traj = Radial(total_nspokes=nb_segments, npoint=npoint)

    elif mode=="Kushball":
        nb_part=int(dico_seqParams["alFree"][12])
        radial_traj = Radial3D(total_nspokes=nb_segments,npoint=npoint,nb_slices=nb_part,undersampling_factor=undersampling_factor,incoherent=incoherent,mode=mode)
    else:
        radial_traj = Radial3D(total_nspokes=nb_segments,npoint=npoint,nb_slices=nb_slices,undersampling_factor=undersampling_factor,incoherent=incoherent,mode=mode)

        cond_us = np.zeros((nb_slices, nb_segments))

        cond_us = cond_us.reshape((nb_slices, -1, 8))

        curr_start = 0
        for sl in range(nb_slices):
            cond_us[sl, curr_start::undersampling_factor, :] = 1
            curr_start = curr_start + 1
            curr_start = curr_start % undersampling_factor

        cond_us = cond_us.flatten()
        included_spokes = cond_us
        included_spokes = (included_spokes > 0)

        radial_traj_allspokes = Radial3D(total_nspokes=nb_segments, undersampling_factor=1, npoint=npoint,
                                         nb_slices=nb_slices, incoherent=incoherent, mode=mode)

        from utils_reco import correct_mvt_kdata_zero_filled
        weights, retained_timesteps = correct_mvt_kdata_zero_filled(radial_traj_allspokes, included_spokes, 1)

        weights = weights.reshape(1, -1, 8, nb_slices)
        import math
        nb_rep = math.ceil(nb_slices / undersampling_factor)
        weights_us = np.zeros(shape=(1, 175, 8, nb_rep), dtype=weights.dtype)

        shift = 0

        for sl in range(nb_slices):
            if int(sl / undersampling_factor) < nb_rep:
                weights_us[:, shift::undersampling_factor, :, int(sl / undersampling_factor)] = weights[:,
                                                                                                shift::undersampling_factor,
                                                                                                :, sl]
                shift += 1
                shift = shift % (undersampling_factor)
            else:
                continue

        weights_us = weights_us.reshape(1, -1, nb_rep)
        weights_us = weights_us[..., None]
        weights=weights_us

    volumes0 = copy(volumes)
    volumes = mu * volumes0

    if use_wavelet:

        wav_level = None
        wav_type = "db4"
        axes=(2,3)

        lambd = lambda_wav

        alpha_denoised = []

        vol_denoised_log = [volumes]
        coefs = pywt.wavedecn(volumes, wav_type, level=wav_level, mode="periodization", axes=axes)
        u, slices = pywt.coeffs_to_array(coefs, axes=axes)
        u0 = u
        y = u
        t = 1
        u = pywt.threshold(y, lambd * mu)
        #u=np.maximum(np.abs(y)-lambd * mu,0)/np.abs(y)*y
        print("Non zero percentage: {} ".format(np.count_nonzero(u)/np.prod(u.shape)))

        vol_denoised_log.append(
            pywt.waverecn(pywt.array_to_coeffs(u, slices), wav_type, mode="periodization", axes=axes))

        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
        y = u
        t = t_next

        for i in range(niter):
            u_prev = u
            x = pywt.array_to_coeffs(y, slices)
            x = pywt.waverecn(x, wav_type, mode="periodization", axes=axes)

            print("x.shape : {}".format(x.shape))
            if not(incoherent):
                volumesi = undersampling_operator_singular_new(x, radial_traj,
                                                               b1_all_slices_2Dplus1_pca, weights=weights,
                                                               density_adj=dens_adj)
            else:


                volumesi = undersampling_operator_singular(x, radial_traj,
                                                         b1_all_slices_2Dplus1_pca,
                                                               density_adj=dens_adj,weights=weights)
                #volumesi = x

            print("volumesi.shape : {}".format(volumesi.shape))
            # volumesi = volumesi.squeeze()
            coefs = pywt.wavedecn(volumesi, wav_type, level=wav_level, mode="periodization", axes=axes)
            grad_y, slices = pywt.coeffs_to_array(coefs, axes=axes)
            grad = grad_y - u0/mu
            y = y - mu * grad

            u = pywt.threshold(y, lambd * mu)
            #u=grad_y

            print("Non zero percentage: {} ".format(np.count_nonzero(u)/np.prod(u.shape)))

            t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
            y = u + (t - 1) / t_next * (u - u_prev)
            t = t_next

            if (i % 1 == 0):
                vol_denoised_log.append(
                    pywt.waverecn(pywt.array_to_coeffs(u, slices), wav_type, mode="periodization", axes=axes))

        vol_denoised_log = np.array(vol_denoised_log)
        filename_target_intermediate = filename_volume.split("_volumes_allbins.npy")[
                                           0] + "_volume_denoised_{}.npy".format(suffix)
        np.save(filename_target_intermediate, vol_denoised_log)

        alpha_denoised.append(u)

        alpha_denoised = np.array(alpha_denoised)
        volumes = [pywt.waverecn(pywt.array_to_coeffs(alpha, slices), wav_type, mode="periodization", axes=axes)
                   for alpha in
                   alpha_denoised]
        volumes = np.array(volumes).squeeze()
        #volumes=volumesi


    elif use_LLR:
        blck = np.array(block.split(",")).astype(int)
        

        volumes_denoised = []

        u = mu * volumes
        u0 = volumes
        y = u
        t = 1

        if (u.ndim==(len(blck)+1)):
            blck=np.array([u.shape[0]]+list(blck))

        strd = blck
        lambd = lambda_LLR
        threshold = lambd * mu

        print(blck)
        
        

        print(u.shape)
        u = proj_LLR(u.squeeze(), strd, blck, threshold)
        print(u.shape)
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
        y = u
        t = t_next
        for i in range(niter):
            u_prev = u
            if u.ndim == 3:
                u = np.expand_dims(u, axis=0)
            
            if not(incoherent):
                volumesi = undersampling_operator_singular_new(u, radial_traj,
                                                               b1_all_slices_2Dplus1_pca, weights=weights,
                                                               density_adj=dens_adj)
            else:
                volumesi = undersampling_operator_singular(u, radial_traj,
                                                               b1_all_slices_2Dplus1_pca,
                                                               density_adj=dens_adj)

            grad = volumesi - u0/mu
            y = y - mu * grad

            u = proj_LLR(y.squeeze(), strd, blck, threshold)

            t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
            y = u + (t - 1) / t_next * (u - u_prev)
            t = t_next

        volumes_denoised.append(u)

        volumes = np.array(volumes_denoised).squeeze()


    else:

        for i in tqdm(range(niter)):
            print("Correcting volumes for iteration {}".format(i))
            all_grad_norm = 0
            for gr in tqdm(range(nbins)):
                volumesi = undersampling_operator_singular_new(volumes[gr], radial_traj, b1_all_slices_2Dplus1_pca,
                                                               weights=all_weights[gr], density_adj=dens_adj)

                grad = volumesi - volumes0[gr]
                volumes[gr] = volumes[gr] - mu * grad

                if (mu_TV is not None) and (not (mu_TV == 0)):
                    print("Applying TV regularization")

                    grad_norm = np.linalg.norm(grad)
                    all_grad_norm += grad_norm ** 2
                    print("grad norm {}".format(grad_norm))
                    del grad
                    grad_TV = np.zeros_like(volumes[gr])
                    for ts in tqdm(range(ntimesteps)):
                        for ind_w, w in (enumerate(weights_TV)):
                            if w > 0:
                                grad_TV[ts] += (w * grad_J_TV(volumes[gr, ts], ind_w, is_weighted=False, shift=0))

                            # grad_TV_norm = np.linalg.norm(grad_TV, axis=0)
                    grad_TV_norm = np.linalg.norm(grad_TV)
                    # signals = matched_signals + mu * grad

                    print("grad_TV_norm {}".format(grad_TV_norm))

                    volumes[gr] -= mu * mu_TV * grad_norm / grad_TV_norm * grad_TV
                    del grad_TV
                    del grad_TV_norm

            all_grad_norm = np.sqrt(all_grad_norm)
            if (mu_bins is not None) and (not (mu_bins == 0)):
                grad_TV_bins = grad_J_TV(volumes, 0, is_weighted=False, shift=0)
                grad_TV_bins_norm = np.linalg.norm(grad_TV_bins)

                volumes -= mu * mu_bins * grad_TV_bins / grad_TV_bins_norm * all_grad_norm
                print("grad_TV_bins norm {}".format(grad_TV_bins_norm))
                del grad_TV_bins

            if (i % 5 == 0):
                filename_target_intermediate = filename_volume.split("_volumes_allbins.npy")[
                                                   0] + "_volumes_allbins_denoised_it{}{}.npy".format(i, suffix)
                np.save(filename_target_intermediate, volumes)

    volumes = np.squeeze(volumes)
    if gamma is not None:
        for gr in range(volumes.shape[0]):
            volumes[gr] = gamma_transform(volumes[gr], gamma)

    np.save(filename_target, volumes)
    # volumes_allbins=build_volume_2Dplus1_cc_allbins(kdata_all_channels_all_slices,b1_all_slices_2Dplus1_pca,pca_dict,all_weights)
    # np.save(filename_volumes,volumes_allbins)

    return

@ma.machine()
@ma.parameter("filename_volume", str, default=None, description="MRF raw data")
@ma.parameter("filename_b1", str, default=None, description="B1")
@ma.parameter("filename_weights", str, default=None, description="Motion bin weights")
@ma.parameter("file_deformation", str, default=None, description="Deformation")
@ma.parameter("filename_seqParams", str, default=None, description="Undersampling weights")
@ma.parameter("index_ref", int, default=0, description="Registration reference")
@ma.parameter("mu", float, default=2, description="Gradient step size")
@ma.parameter("mu_TV", float, default=None, description="Spatial Regularization")
@ma.parameter("weights_TV", [float,float,float], default=[1.0,0.2,0.2], description="Spatial Regularization Weights")
@ma.parameter("niter", int, default=None, description="Number of iterations")
@ma.parameter("suffix", str, default="", description="Suffix")
@ma.parameter("gamma", float, default=None, description="Gamma Correction")
@ma.parameter("gating_only", bool, default=False, description="Use weights only for gating")
@ma.parameter("dens_adj", bool, default=True, description="Use Radial density adjustment")
@ma.parameter("nb_rep_center_part", int, default=1, description="Number of center partition repetition")
@ma.parameter("beta", float, default=None, description="Relative importance of registered volumes vs volume of reference")
@ma.parameter("interp", str, default=None, description="Registration interpolation")
@ma.parameter("select_first_rep", bool, default=False, description="Select firt repetition of central partition only")
@ma.parameter("lambda_wav", float, default=0.5e-5, description="Lambda wavelet")
@ma.parameter("use_wavelet", bool, default=False, description="Wavelet regularization instead of TV")
@ma.parameter("us", int, default=1, description="Undersampling")
@ma.parameter("kept_bins",str,default=None,description="Bins to keep")
@ma.parameter("axis", int, default=None, description="Registration axis")
def build_volumes_iterative_allbins_registered(filename_volume,filename_b1,filename_weights,filename_seqParams,mu,mu_TV,niter,gamma,suffix,gating_only,dens_adj,nb_rep_center_part,file_deformation,index_ref,beta,interp,select_first_rep,lambda_wav,use_wavelet,us,kept_bins,axis,weights_TV):
    filename_target=filename_volume.split("_volumes_allbins.npy")[0] + "_volumes_allbins_registered_ref{}{}.npy".format(index_ref,suffix)

    if gamma is not None:
        filename_target=filename_target.split(".npy")[0]+"_gamma_{}.npy".format(str(gamma).replace(".","_"))
    
    if interp is None:
        interp=cv2.INTER_LINEAR

    elif interp=="nearest":
        interp=cv2.INTER_NEAREST
    
    elif interp=="cubic":
        interp=cv2.INTER_CUBIC


    weights_TV/=np.sum(weights_TV)
    print("Loading Volumes")
    volumes=np.load(filename_volume)
    #To fit the input expected by the undersampling function with L0=1 in our case
    #volumes=np.expand_dims(volumes,axis=1)

    volumes=volumes.astype("complex64")

    if volumes.ndim==4:
        #To fit the input expected by the undersampling function with L0=1 in our case
        volumes=np.expand_dims(volumes,axis=1)
        shift=0
    else:
        shift=1

    if filename_weights is None:
        filename_weights = (filename_volume.split("_volumes_allbins.npy")[0] + "_weights.npy").replace("_no_densadj","")
    if gating_only:
        filename_weights=str.replace(filename_weights,"_no_dcomp","")

    

    b1_all_slices_2Dplus1_pca=np.load(filename_b1)
    all_weights=np.load(filename_weights)



    if gating_only:
        print("Using weights only for gating")
        all_weights=(all_weights>0)*1

    if nb_rep_center_part>1:
        all_weights=weights_aggregate_center_part(all_weights,nb_rep_center_part,select_first_rep)


    if us >1:
        weights_us = np.zeros_like(all_weights)
        nb_slices = all_weights.shape[3]
        nspoke_per_part = 8
        weights_us = weights_us.reshape((weights_us.shape[0], 1, -1, nspoke_per_part, nb_slices, 1))


        curr_start = 0

        for sl in range(nb_slices):
            weights_us[:, :, curr_start::us, :, sl] = 1
            curr_start = curr_start + 1
            curr_start = curr_start % us

        weights_us=weights_us.reshape(all_weights.shape)
        all_weights *= weights_us


    if kept_bins is not None:
        kept_bins_list=np.array(str.split(kept_bins,",")).astype(int)
        print(kept_bins_list)
        volumes=volumes[kept_bins_list]
        all_weights=all_weights[kept_bins_list]


    print("Volumes shape {}".format(volumes.shape))
    print("Weights shape {}".format(all_weights.shape))

    nb_allspokes=all_weights.shape[2]
    npoint_image=volumes.shape[-1]
    npoint=2*npoint_image
    nbins=volumes.shape[0]
    nb_slices=volumes.shape[2]
    nb_part = all_weights.shape[3]

    if filename_seqParams is not None:
        file = open(filename_seqParams, "rb")
        dico_seqParams = pickle.load(file)
        file.close()

        use_navigator_dll = dico_seqParams["use_navigator_dll"]
        nb_gating_spokes=dico_seqParams["alFree"][6]
        if "use_kushball_dll" in dico_seqParams.keys():
            use_kushball_dll=dico_seqParams["use_kushball_dll"]
        else:
            use_kushball_dll=False


        if use_kushball_dll:
            meas_sampling_mode = dico_seqParams["alFree"][16]
        elif (use_navigator_dll):
            meas_sampling_mode = dico_seqParams["alFree"][15]
        else:
            meas_sampling_mode = dico_seqParams["alFree"][12]

        print(meas_sampling_mode)

        
        if meas_sampling_mode == 1:
            incoherent = False
            mode = None
        elif meas_sampling_mode == 2:
            incoherent = True
            mode = "old"
        elif meas_sampling_mode == 3:
            incoherent = True
            mode = "new"
        
        elif meas_sampling_mode == 4:
            incoherent = True
            mode = "Kushball"
    
    else:
        incoherent=False
        mode="old"


    if not(incoherent):
        radial_traj = Radial(total_nspokes=nb_allspokes, npoint=npoint)


    else:
        radial_traj = Radial3D(total_nspokes=nb_allspokes,npoint=npoint,nb_slices=nb_part,undersampling_factor=1,incoherent=incoherent,mode=mode)


    if file_deformation is None:#identity by default
        X,Y=np.meshgrid(np.arange(npoint_image),np.arange(npoint_image))
        def_identity_x=np.tile(np.expand_dims(X,axis=(0,1)),(nbins,nb_slices,1,1))
        def_identity_y=np.tile(np.expand_dims(Y,axis=(0,1)),(nbins,nb_slices,1,1))
        deformation_map=np.stack([def_identity_x,def_identity_y],axis=0)
        

    else:
        deformation_map=np.load(file_deformation)

    nb_slices_def=deformation_map.shape[2]
    npoint_def=deformation_map.shape[-1]


    if (nb_slices>nb_slices_def)or(npoint_image>npoint_def):
        print("Regridding deformation map")
        new_shape=(nb_slices,npoint_image,npoint_image)
        deformation_map=interp_deformation_resize(deformation_map,new_shape)

    nb_slices_b1=b1_all_slices_2Dplus1_pca.shape[1]
    npoint_b1=b1_all_slices_2Dplus1_pca.shape[-1]
    nb_channels=b1_all_slices_2Dplus1_pca.shape[0]

    if (nb_slices>nb_slices_b1)or(npoint_image>npoint_b1):
        print("Regridding B1")
        new_shape=(nb_slices,npoint_image,npoint_image)
        b1_all_slices_2Dplus1_pca=interp_b1_resize(b1_all_slices_2Dplus1_pca,new_shape)
        print("Warning: pca_dict can only be interpolated when no coil compression for the moment")
        pca_dict={}
        for sl in range(nb_slices):
            pca=PCAComplex(n_components_=nb_channels)
            pca.explained_variance_ratio_=[1]
            pca.components_=np.eye(nb_channels)
            pca_dict[sl]=deepcopy(pca)

    deformation_map=change_deformation_map_ref(deformation_map,index_ref,axis)

    print("Calculating inverse deformation map")

    if file_deformation is None:#identity by default
        inv_deformation_map=deformation_map
    else:
        inv_deformation_map = np.zeros_like(deformation_map)
        for gr in tqdm(range(nbins)):
            inv_deformation_map[:, gr] = calculate_inverse_deformation_map(deformation_map[:, gr],axis)
        
    print(volumes.shape)
    volumes_registered=np.zeros(volumes.shape[1:],dtype=volumes.dtype)
    print(volumes_registered.shape)


    if file_deformation is None:
        print("No deformation - summing volumes for all bins")
        for gr in range(nbins):
            volumes_registered+=volumes[gr].squeeze()
    else:

        print("Registering initial volumes")
        for gr in range(nbins):
            if beta is None:
                volumes_registered+=apply_deformation_to_complex_volume(volumes[gr].squeeze(),deformation_map[:,gr],interp=interp,axis=axis)
            else:
                if gr==index_ref:
                    volumes_registered+=beta*apply_deformation_to_complex_volume(volumes[gr].squeeze(),deformation_map[:,gr],interp=interp,axis=axis)
                else:
                    volumes_registered += (1-beta) * apply_deformation_to_complex_volume(volumes[gr].squeeze(),
                                                                                    deformation_map[:, gr],interp=interp,axis=axis)


    volumes0=copy(volumes_registered)

    volumes_registered=mu*volumes0

    print("volumes_registered.shape {}".format(volumes_registered.shape))

    print(use_wavelet)
    if use_wavelet:

        wav_level = None
        wav_type="db4"

        lambd = lambda_wav

        print("Wavelet regularization penalty {}".format(lambd))

        if volumes_registered.ndim==3:
            volumes_registered=np.expand_dims(volumes_registered,axis=0)

        coefs = pywt.wavedecn(volumes_registered, wav_type, level=wav_level, mode="periodization",axes=(1,2,3))
        u, slices = pywt.coeffs_to_array(coefs,axes=(1,2,3))
        u0 = u
        y = u
        t = 1

        u = pywt.threshold(y, lambd * mu)

        print("Non zero percentage: {} ".format(np.count_nonzero(u)/np.prod(u.shape)))

        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
        y = u 
        t = t_next

        for i in range(niter):
            u_prev = u
            x = pywt.array_to_coeffs(y, slices)
            x = pywt.waverecn(x, wav_type, mode="periodization",axes=(1,2,3))

        
            for gr in tqdm(range(nbins)):
                
                volumesi=apply_deformation_to_complex_volume(x,inv_deformation_map[:,gr],interp=interp,axis=axis)
                if volumesi.ndim==3:
                    volumesi=np.expand_dims(volumesi,axis=0)
                print("volumesi.shape {}".format(volumesi.shape))

                if not(incoherent):
                    volumesi = undersampling_operator_singular_new(volumesi, radial_traj,
                                                               b1_all_slices_2Dplus1_pca, weights=all_weights[gr],
                                                               density_adj=dens_adj)
                else:


                    volumesi = undersampling_operator_singular(volumesi, radial_traj,
                                                            b1_all_slices_2Dplus1_pca,
                                                                density_adj=dens_adj,weights=all_weights[gr])
                    
                volumesi=apply_deformation_to_complex_volume(volumesi.squeeze(),deformation_map[:,gr],interp=interp,axis=axis)

                if volumesi.ndim==3:
                    volumesi=np.expand_dims(volumesi,axis=0)

                if beta is not None:
                    if gr == index_ref:
                        volumesi *= beta
                    else:
                        volumesi *= (1 - beta)

                if gr==0:
                    final_volumesi=volumesi
                else:
                    final_volumesi+=volumesi
            #volumesi = volumesi.squeeze()
            coefs = pywt.wavedecn(final_volumesi, wav_type, level=wav_level, mode="periodization",axes=(1,2,3))
            grad_y, slices = pywt.coeffs_to_array(coefs,axes=(1,2,3))
            grad = grad_y - u0
            y = y - mu * grad

            u = pywt.threshold(y, lambd * mu)

            print("Non zero percentage: {} ".format(np.count_nonzero(u)/np.prod(u.shape)))

            t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
            y = u + (t - 1) / t_next * (u - u_prev)
            t = t_next

            if (i%1==0):    
                filename_target_intermediate=filename_volume.split("_volumes_allbins.npy")[0] + "_volumes_allbins_registered_ref{}_it{}{}.npy".format(index_ref,i,suffix)
                volumes_registered = pywt.waverecn(pywt.array_to_coeffs(u, slices), wav_type, mode="periodization",axes=(1,2,3)).squeeze()
                np.save(filename_target_intermediate,volumes_registered)

        volumes_registered = pywt.waverecn(pywt.array_to_coeffs(u, slices), wav_type, mode="periodization",axes=(1,2,3)).squeeze()



    else:
        for i in tqdm(range(niter)):
            print("Correcting volumes for iteration {}".format(i))
            all_grad_norm=0
            for gr in tqdm(range(nbins)):
                
                volumesi=apply_deformation_to_complex_volume(volumes_registered,inv_deformation_map[:,gr],interp=interp,axis=axis)
                if volumesi.ndim==3:
                    volumesi=np.expand_dims(volumesi,axis=0)
                print("volumesi.shape {}".format(volumesi.shape))


                if not(incoherent):
                    volumesi = undersampling_operator_singular_new(volumesi, radial_traj,
                                                               b1_all_slices_2Dplus1_pca, weights=all_weights[gr],
                                                               density_adj=dens_adj)
                else:


                    volumesi = undersampling_operator_singular(volumesi, radial_traj,
                                                            b1_all_slices_2Dplus1_pca,
                                                                density_adj=dens_adj,weights=all_weights[gr])
                volumesi=apply_deformation_to_complex_volume(volumesi.squeeze(),deformation_map[:,gr],interp=interp,axis=axis)

                if beta is not None:
                    if gr == index_ref:
                        volumesi *= beta
                    else:
                        volumesi *= (1 - beta)

                if gr==0:
                    final_volumesi=volumesi
                else:
                    final_volumesi+=volumesi

            grad = final_volumesi - volumes0
            volumes_registered = volumes_registered - mu * grad
            


            if (mu_TV is not None)and(not(mu_TV==0)):
                print("Applying TV regularization")
                                
                grad_norm=np.linalg.norm(grad)
                all_grad_norm+=grad_norm**2
                print("grad norm {}".format(grad_norm))
                del grad
                grad_TV=np.zeros_like(volumes_registered)
                
                for ind_w, w in (enumerate(weights_TV)):
                    if w > 0:
                        grad_TV += (w * grad_J_TV(volumes_registered, ind_w,is_weighted=False,shift=shift))

                            #grad_TV_norm = np.linalg.norm(grad_TV, axis=0)
                grad_TV_norm = np.linalg.norm(grad_TV)
                                # signals = matched_signals + mu * grad

                print("grad_TV_norm {}".format(grad_TV_norm))

                volumes_registered -= mu * mu_TV * grad_norm / grad_TV_norm * grad_TV
                del grad_TV
                del grad_TV_norm

            

            if (i%1==0):    
                filename_target_intermediate=filename_volume.split("_volumes_allbins.npy")[0] + "_volumes_allbins_registered_ref{}_it{}{}.npy".format(index_ref,i,suffix)
                np.save(filename_target_intermediate,volumes_registered)

    volumes_registered=np.squeeze(volumes_registered)
    if gamma is not None:
        volumes_registered=gamma_transform(volumes_registered)



    np.save(filename_target,volumes_registered)
    #volumes_allbins=build_volume_2Dplus1_cc_allbins(kdata_all_channels_all_slices,b1_all_slices_2Dplus1_pca,pca_dict,all_weights)
    #np.save(filename_volumes,volumes_allbins)

    return

@ma.machine()
@ma.parameter("filename_volume", str, default=None, description="MRF raw data")
@ma.parameter("filename_b1", str, default=None, description="B1")
@ma.parameter("filename_weights", str, default=None, description="Motion bin weights")
@ma.parameter("filename_seqParams", str, default=None, description="Undersampling weights")
@ma.parameter("file_deformation", str, default=None, description="Deformation")
@ma.parameter("mu", float, default=1, description="Gradient step size")
@ma.parameter("mu_TV", float, default=None, description="Spatial Regularization")
@ma.parameter("niter", int, default=None, description="Number of iterations")
@ma.parameter("suffix", str, default="", description="Suffix")
@ma.parameter("gamma", float, default=None, description="Gamma Correction")
@ma.parameter("gating_only", bool, default=False, description="Use weights only for gating")
@ma.parameter("dens_adj", bool, default=True, description="Use Radial density adjustment")
@ma.parameter("nb_rep_center_part", int, default=1, description="Number of center partition repetition")
@ma.parameter("beta", float, default=None, description="Relative importance of registered volumes vs volume of reference")
@ma.parameter("interp", str, default=None, description="Registration interpolation")
@ma.parameter("select_first_rep", bool, default=False, description="Select firt repetition of central partition only")
@ma.parameter("us", int, default=1, description="Undersampling")
@ma.parameter("axis", int, default=None, description="Registration axis")
def build_volumes_iterative_allbins_registered_allindex(filename_volume,filename_b1,filename_weights,filename_seqParams,mu,mu_TV,niter,gamma,suffix,gating_only,dens_adj,nb_rep_center_part,file_deformation,beta,interp,select_first_rep,us,axis):
    filename_target=filename_volume.split("_volumes_allbins.npy")[0] + "_volumes_allbins_registered_allindex{}.npy".format(suffix)

    if gamma is not None:
        filename_target=filename_target.split(".npy")[0]+"_gamma_{}.npy".format(str(gamma).replace(".","_"))
    
    weights_TV=np.array([1.0,0.2,0.2])
    weights_TV/=np.sum(weights_TV)
    print("Loading Volumes")
    volumes=np.load(filename_volume)
    
    if volumes.ndim==4:
        #To fit the input expected by the undersampling function with L0=1 in our case
        volumes=np.expand_dims(volumes,axis=1)
        shift=0
    else:
        shift=1
    
    if interp is None:
        interp=cv2.INTER_LINEAR

    elif interp=="nearest":
        interp=cv2.INTER_NEAREST
    
    elif interp=="cubic":
        interp=cv2.INTER_CUBIC

    volumes=volumes.astype("complex64")

    if filename_weights is None:
        filename_weights = (filename_volume.split("_volumes_allbins.npy")[0] + "_weights.npy").replace("_no_densadj","")
    if gating_only:
        filename_weights=str.replace(filename_weights,"_no_dcomp","")

    

    b1_all_slices_2Dplus1_pca=np.load(filename_b1)
    all_weights=np.load(filename_weights)

    if gating_only:
        all_weights=(all_weights>0)*1

    if nb_rep_center_part>1:
        all_weights=weights_aggregate_center_part(all_weights,nb_rep_center_part,select_first_rep)

    if us >1:
        weights_us = np.zeros_like(all_weights)
        nb_slices = all_weights.shape[3]
        nspoke_per_part = 8
        weights_us = weights_us.reshape((weights_us.shape[0], 1, -1, nspoke_per_part, nb_slices, 1))


        curr_start = 0

        for sl in range(nb_slices):
            weights_us[:, :, curr_start::us, :, sl] = 1
            curr_start = curr_start + 1
            curr_start = curr_start % us

        weights_us=weights_us.reshape(all_weights.shape)
        all_weights *= weights_us

    print("Volumes shape {}".format(volumes.shape))
    print("Weights shape {}".format(all_weights.shape))

    nb_allspokes=all_weights.shape[2]
    npoint=2*volumes.shape[-1]
    nbins=volumes.shape[0]
    nb_slices=volumes.shape[2]
    nb_part = all_weights.shape[3]

    if filename_seqParams is not None:
        file = open(filename_seqParams, "rb")
        dico_seqParams = pickle.load(file)
        file.close()

        use_navigator_dll = dico_seqParams["use_navigator_dll"]
        nb_gating_spokes=dico_seqParams["alFree"][6]
        if "use_kushball_dll" in dico_seqParams.keys():
            use_kushball_dll=dico_seqParams["use_kushball_dll"]
        else:
            use_kushball_dll=False


        if use_kushball_dll:
            meas_sampling_mode = dico_seqParams["alFree"][16]
        elif (use_navigator_dll):
            meas_sampling_mode = dico_seqParams["alFree"][15]
        else:
            meas_sampling_mode = dico_seqParams["alFree"][12]

        
        if meas_sampling_mode == 1:
            incoherent = False
            mode = None
        elif meas_sampling_mode == 2:
            incoherent = True
            mode = "old"
        elif meas_sampling_mode == 3:
            incoherent = True
            mode = "new"
        
        elif meas_sampling_mode == 4:
            incoherent = True
            mode = "Kushball"
    
    else:
        incoherent=False
        mode="old"

    # radial_traj_2D=Radial(total_nspokes=nb_allspokes,npoint=npoint)

    if not(incoherent):
        radial_traj = Radial(total_nspokes=nb_allspokes, npoint=npoint)

    # elif mode=="Kushball":
    #     radial_traj = Radial3D(total_nspokes=nb_allspokes,npoint=npoint,nb_slices=nb_slices,undersampling_factor=undersampling_factor,incoherent=incoherent,mode=mode)
    else:
        radial_traj = Radial3D(total_nspokes=nb_allspokes,npoint=npoint,nb_slices=nb_part,undersampling_factor=1,incoherent=incoherent,mode=mode)


    deformation_map=np.load(file_deformation)


    deformation_map_allindex=[]
    inv_deformation_map_allindex=[]
    volumes_registered_allindex=[]

    print("Extraction deformation map for all bin reference")
    for index_ref in range(nbins):
        deformation_map_allindex.append(change_deformation_map_ref(deformation_map,index_ref,axis))

    print("Building inverse deformation map for all bin reference")
    for index_ref in range(nbins):
        deformation_map=deformation_map_allindex[index_ref]
        print("Calculating inverse deformation map")
        inv_deformation_map = np.zeros_like(deformation_map)
        for gr in tqdm(range(nbins)):
            inv_deformation_map[:, gr] = calculate_inverse_deformation_map(deformation_map[:, gr],axis)
        inv_deformation_map_allindex.append(inv_deformation_map)

        volumes_registered=np.zeros(volumes.shape[1:],dtype=volumes.dtype)
        

        print("Registering initial volumes")
        for gr in range(nbins):

            if beta is None:
                volumes_registered+=apply_deformation_to_complex_volume(volumes[gr].squeeze(),deformation_map[:,gr],interp,axis)

            else:
                if gr==index_ref:
                    volumes_registered+=beta*apply_deformation_to_complex_volume(volumes[gr].squeeze(),deformation_map[:,gr],interp,axis)
                else:
                    volumes_registered += (1-beta) * apply_deformation_to_complex_volume(volumes[gr].squeeze(),
                                                                                     deformation_map[:, gr],interp,axis)
        volumes_registered_allindex.append(volumes_registered)
    
    deformation_map_allindex=np.array(deformation_map_allindex)
    inv_deformation_map_allindex=np.array(inv_deformation_map_allindex)

    volumes_registered_allindex=np.array(volumes_registered_allindex)
    volumes0_allindex=copy(volumes_registered_allindex)
    volumes_registered_allindex=mu*volumes0_allindex


    for i in tqdm(range(niter)):
        print("Correcting volumes for iteration {}".format(i))
        for index_ref in range(nbins):
            all_grad_norm=0
            for gr in tqdm(range(nbins)):
                print(volumes_registered_allindex.shape)
                print(inv_deformation_map_allindex[index_ref,:,gr].shape)

                volumesi=apply_deformation_to_complex_volume(volumes_registered_allindex[index_ref],inv_deformation_map_allindex[index_ref,:,gr],interp,axis)
                if volumesi.ndim==3:
                    volumesi=np.expand_dims(volumesi,axis=0)
                if not(incoherent):
                    volumesi = undersampling_operator_singular_new(volumesi, radial_traj,
                                                               b1_all_slices_2Dplus1_pca, weights=all_weights[gr],
                                                               density_adj=dens_adj)
                else:


                    volumesi = undersampling_operator_singular(volumesi, radial_traj,
                                                            b1_all_slices_2Dplus1_pca,
                                                                density_adj=dens_adj,weights=all_weights[gr])
                                                                
                volumesi=apply_deformation_to_complex_volume(volumesi.squeeze(),deformation_map_allindex[index_ref,:,gr],interp,axis)

                if beta is not None:
                    if gr == index_ref:
                        volumesi *= beta
                    else:
                        volumesi *= (1 - beta)

                if gr==0:
                    final_volumesi=volumesi
                else:
                    final_volumesi+=volumesi

            grad = final_volumesi - volumes0_allindex[index_ref]
            volumes_registered_allindex[index_ref] = volumes_registered_allindex[index_ref] - mu * grad
            


            if (mu_TV is not None)and(not(mu_TV==0)):
                print("Applying TV regularization")
                                
                grad_norm=np.linalg.norm(grad)
                all_grad_norm+=grad_norm**2
                print("grad norm {}".format(grad_norm))
                del grad
                grad_TV=np.zeros_like(volumes_registered_allindex[index_ref])
                
                for ind_w, w in (enumerate(weights_TV)):
                    if w > 0:
                        grad_TV += (w * grad_J_TV(volumes_registered_allindex[index_ref], ind_w,is_weighted=False,shift=shift))

                            #grad_TV_norm = np.linalg.norm(grad_TV, axis=0)
                grad_TV_norm = np.linalg.norm(grad_TV)
                                # signals = matched_signals + mu * grad

                print("grad_TV_norm {}".format(grad_TV_norm))

                volumes_registered_allindex[index_ref] -= mu * mu_TV * grad_norm / grad_TV_norm * grad_TV
                del grad_TV
                del grad_TV_norm

        

        if (i%5==0):    
            filename_target_intermediate=filename_volume.split("_volumes_allbins.npy")[0] + "_volumes_allbins_registered_allindex_it{}{}.npy".format(i,suffix)
            np.save(filename_target_intermediate,volumes_registered_allindex)

    volumes_registered_allindex=np.squeeze(volumes_registered_allindex)
    if gamma is not None:
        volumes_registered_allindex=gamma_transform(volumes_registered_allindex)



    np.save(filename_target,volumes_registered_allindex)
    #volumes_allbins=build_volume_2Dplus1_cc_allbins(kdata_all_channels_all_slices,b1_all_slices_2Dplus1_pca,pca_dict,all_weights)
    #np.save(filename_volumes,volumes_allbins)

    return




# @ma.machine()
# @ma.parameter("sequence_file", str, default="./dico/mrf_sequence_adjusted.json", description="Sequence File")
# @ma.parameter("reco", float, default=4, description="Recovery (s)")
# @ma.parameter("min_TR_delay", float, default=1.14, description="TR delay (ms)")
# @ma.parameter("dictconf", str, default="./dico/mrf_dictconf_Dico2_Invivo_overshoot.json", description="Dictionary grid")
# @ma.parameter("dictconf_light", str, default="./dico/mrf_dictconf_Dico2_Invivo_light_for_matching_overshoot.json", description="Coarse dictionary grid (clustering)")
# @ma.parameter("diconame", str, default="dico", description="Dictionary prefix name")
# @ma.parameter("inversion", bool, default=True, description="Use initial inversion")
# @ma.parameter("TI", float, default=8.32, description="Inversion time (ms)")
# @ma.parameter("is_build_phi", bool, default=True, description="Whether to build temporal basis phi")
# @ma.parameter("L0", int, default=6, description="Number of temporal components")
# def generate_dictionaries_T1FF(sequence_file,reco,min_TR_delay,dictconf,dictconf_light,inversion,TI, is_build_phi,L0,diconame):
#     generate_dictionaries(sequence_file,reco,min_TR_delay,dictconf,dictconf_light,TI=TI, dest=None,diconame=diconame,is_build_phi=is_build_phi,L0=L0)
#     return



@ma.machine()
@ma.parameter("dest",ma.Path(),default="mrf_dict",help="Destination directory for MRF dictionary")
@ma.parameter("folder",ma.Path(exists=True),default=None,help="Directory of sequence and dictionary configuration files.")
@ma.parameter("seqfile",ma.Path(),default="./dico/mrf_sequence_adjusted.json",help="Sequence parameter file (.json)")
@ma.parameter("dictconf",ma.Path(),default="./dico/mrf_dictconf_Dico2_Invivo_overshoot.json",help="Full dictionary grid (.json)")
@ma.parameter("dictconf_light",ma.Path(),default="./dico/mrf_dictconf_Dico2_Invivo_light_for_matching_overshoot.json",help="Light dictionary grid (.json)")
@ma.parameter("datafile",ma.Path(),default=None,help="MRF .dat file to get echo spacing")
@ma.parameter("index",int,default=-1,help="Index of the .dat file for multiple acquisitions (default -1 for single acquisition).")
@ma.parameter("wait_time",float,default=5.0,help="Waiting time (s) at the end of each MRF repetition.")
@ma.parameter("echo_spacing",float,default=None,help="Waiting time (ms) from echo time to next RF pulse.")
@ma.parameter("is_build_phi",bool,default=True,help="Whether to build temporal basis from dictionary")
@ma.parameter("pca",int,default=6,help="Number of components for the dictionary projection on the temporal basis")
@ma.parameter("force",bool,default=False,help="Force generation even if folder already exists")
@ma.parameter("inversion_time", float, default=8.32, help="Inversion time (ms).")
def mrf_gendict(dest,folder,seqfile,dictconf,dictconf_light,datafile,wait_time,echo_spacing,inversion_time,is_build_phi,pca,force,index):
    

    # if seqfile is None:
    #     print("No sequence file provided. Using default SEQ_CONFIG")
    #     seqfile = cfg_mrf.SEQ_CONFIG
    # if dictconf is None:
    #     print("No dictionary config file provided. Using default DICT_CONFIG")
    #     dictconf = cfg_mrf.DICT_CONFIG
    # if dictconf_light is None:
    #     print(
    #         "No dictionary light config file provided. Using default DICT_LIGHT_CONFIG"
    #     )
    #     dictconf_light = cfg_mrf.DICT_LIGHT_CONFIG

    if folder is not None:
        folder = Path(folder)
        if type(seqfile) == str:
            seqfile = folder / seqfile
        if type(dictconf) == str:
            dictconf = folder / dictconf
        if type(dictconf_light) == str:
            dictconf_light = folder / dictconf_light

    # dest

    if datafile is None:
        if echo_spacing is None:
            raise ma.ExpectedError("Echo spacing should be provided when no datafile is given")
    else:
        import numpy as np
        seq_params = build_dico_seqParams(str(datafile),index)
        echo_spacing=np.round(seq_params["dTR"],2)
        inversion_time=np.round(seq_params["TI"],2)
        print("Echo spacing of {} ms read in the .dat file".format(echo_spacing))
        print("Inversion time of {} ms read in the .dat file".format(inversion_time))

    dest = Path(dest)
    if not(force) and (dest.exists() and list(dest.glob("*"))):
        raise ma.ExpectedError(f"Output directory ({dest}) is not empty. Aborting.")
    dest.mkdir(exist_ok=True, parents=True)

    generate_dictionaries(
        seqfile,
        wait_time,
        echo_spacing,
        dictconf,
        dictconf_light,
        TI=inversion_time,
        dest=dest,
        is_build_phi=is_build_phi,
        L0=pca
    )



@ma.machine()
@ma.parameter("filemap", str, default=None, description="MRF maps (.pkl)")
@ma.parameter("fileseq", str, default=None, description="Sequence File")
@ma.parameter("spacing", [float,float,float], default=[1,1,5], description="Voxel size")
@ma.parameter("reorient", bool, default=True, description="Reorient to match usual orientation")
@ma.parameter("filename", str, default=None, description=".dat file for adding geometry if necessary")
@ma.parameter("t1_weighted", bool, default=False, description="Whether to ensure T1 weighting - short TR")
def generate_dixon_volumes_for_segmentation(filemap,fileseq,spacing,reorient,filename,t1_weighted):
    gen_mode="other"

    if filename is not None :
        reorient=False

    if fileseq is not None:
        with open(fileseq) as f:
            sequence_config = json.load(f)

    else:
        sequence_config={}
        sequence_config["TI"]=8.32
        sequence_config["FA"]=5

    sequence_config["TE"]=[2.39,3.45]
    if t1_weighted:
        sequence_config["TR"]=list(np.array(sequence_config["TE"])+2)
    else:
        sequence_config["TR"]=list(np.array(sequence_config["TE"])+10000)
    
    sequence_config["B1"]=[3.0,3.0]

    nrep=1
    rep=nrep-1
    TR_total = np.sum(sequence_config["TR"])

    Treco = TR_total-np.sum(sequence_config["TR"])
    Treco=10000
    ##other options
    sequence_config["T_recovery"]=Treco
    sequence_config["nrep"]=nrep
    sequence_config["rep"]=rep

    if t1_weighted:
        seq=T1MRFSS(**sequence_config)
    else:
        seq=T1MRFSS_NoInv(**sequence_config)

    #seq=T1MRF(**sequence_config)

    gen_mode="other"

    import pickle
    with open(filemap,"rb") as file:
        all_maps=pickle.load(file)

    #print(all_maps)
    mask=all_maps[0][1]
    map_rebuilt=all_maps[0][0]
    print(map_rebuilt.keys())
    map_rebuilt["att"]=np.ones_like(map_rebuilt["att"])
    #map_rebuilt["att"]=1/map_rebuilt["att"]
    norm=all_maps[0][4]
    phase=all_maps[0][3]


    values_simu = [makevol(map_rebuilt[k], mask > 0) for k in map_rebuilt.keys()]
    map_for_sim = dict(zip(list(map_rebuilt.keys()), values_simu))

                # predict spokes
    m = MapFromDict("RebuiltMapFromParams", paramMap=map_for_sim, rounding=True,gen_mode=gen_mode)

    m.buildParamMap()
    m.build_ref_images(seq,norm=norm,phase=phase)

    if reorient:
        volume_ip=np.flip(np.moveaxis(np.abs(m.images_series[0]),0,-1),axis=(1,2))
        volume_oop=np.flip(np.moveaxis(np.abs(m.images_series[1]),0,-1),axis=(1,2))
    else:
        volume_ip=np.abs(m.images_series[0])
        volume_oop=np.abs(m.images_series[1])
    
    
    split=str.split(filemap,"CF_iterative_2Dplus1_MRF_map.pkl")

    if len(split)==1:
        split=str.split(filemap,"MRF_map.pkl")
    file_ip=split[0]+"ip.mha"
    file_oop=split[0]+"oop.mha"

    if filename is None:
        io.write(file_ip, volume_ip, tags={"spacing": spacing})
        io.write(file_oop, volume_oop, tags={"spacing": spacing})

    else:
        print("Getting geometry from {}".format(filename))
        geom,is3D,orientation,offset=get_volume_geometry(filename)
        if is3D:
            volume_ip=np.flip(np.moveaxis(volume_ip,0,-1),axis=(1,2))
            volume_oop=np.flip(np.moveaxis(volume_oop,0,-1),axis=(1,2))
        else:
            volume_ip=np.moveaxis(volume_ip,0,2)[:,::-1]
            volume_oop=np.moveaxis(volume_oop,0,2)[:,::-1]
        
        
        volume_ip = io.Volume(volume_ip, **geom)
        volume_oop = io.Volume(volume_oop, **geom)
        io.write(file_ip,volume_ip)
        io.write(file_oop,volume_oop)
    


    return




@ma.machine()
@ma.parameter("fileseg", str, default=None, description="Segmentation (.nii or .nii.gz)")
def generate_mask_roi_from_segmentation(fileseg):

    file_maskROI=str.split(fileseg,".nii")[0]+"_maskROI.npy"
    img = nib.load(fileseg)
    data = np.array(img.dataobj)
    print("Reorienting in the reconstructed maps original orientation")
    maskROI=np.moveaxis(np.flip(data,axis=(1,2)),-1,0)

    np.save(file_maskROI,maskROI)

    return


@ma.machine()
@ma.parameter("filemap", str, default=None, description="Maps (.pkl)")
@ma.parameter("fileroi", str, default=None, description="ROIs (.npy)")
@ma.parameter("filelabels", str, default=None, description="Labels name mapping (.txt)")
@ma.parameter("adj_wT1", bool, default=True, description="Filter water T1 value for FF")
@ma.parameter("fat_threshold", float, default=0.7, description="FF threshold for water T1 filtering")
@ma.parameter("excluded", int, default=5, description="number of excluded border slices on each extremity")
@ma.parameter("kernel", int, default=5, description="Kernel size for erosion")
@ma.parameter("roi", int, default=15, description="Min ROI number of pixel for inclusion")
def getROIresults(filemap,fileroi,filelabels,adj_wT1,fat_threshold,excluded,kernel,roi):

    file_results_ROI=str.split(filemap,".pkl")[0]+"_ROI_results.csv"

    import pickle
    with open(filemap,"rb") as file:
        all_maps=pickle.load(file)
    mask=all_maps[0][1]
    map_=all_maps[0][0]
    maskROI=np.load(fileroi)

    results=get_ROI_values(map_,mask,maskROI,adj_wT1=adj_wT1, fat_threshold=fat_threshold,kept_keys=None,min_ROI_count=roi,return_std=True,excluded_border_slices=excluded,kernel_size=kernel)

    if filelabels is not None:
        labels=pd.read_csv(filelabels,skiprows=14,header=None,delim_whitespace=True).iloc[:,-1]
        labels.name="ROI"
        results=pd.merge(results,labels,left_index=True,right_index=True).set_index("ROI")


    print(results)
    results.to_csv(file_results_ROI)

    return



@ma.machine()
@ma.parameter("filemap", str, default=None, description="Maps (.mha)")
@ma.parameter("fileroi", str, default=None, description="ROIs (.mha)")
@ma.parameter("filelabels", str, default=None, description="Labels name mapping (.txt)")
@ma.parameter("excluded", int, default=5, description="number of excluded border slices on each extremity")
@ma.parameter("kernel", int, default=5, description="Kernel size for erosion")
@ma.parameter("roi", int, default=15, description="Min ROI number of pixel for inclusion")
@ma.parameter("threshold", int, default=None, description="maximum value threshold for excluding values")
def getROIresults_mha(filemap,fileroi,filelabels,excluded,kernel,roi,threshold):

    extension=str.split(filemap,'.')[-1]
    file_results_ROI=str.split(filemap,".{}".format(extension))[0]+"_ROI_results.csv"

    map_=np.array(io.read(filemap))
    maskROI=np.array(io.read(fileroi))

    if threshold is None:
        if "wT1" in filemap:
            threshold=1700
        else:
            threshold=None
    
    results=get_ROI_values_image(map_,maskROI,min_ROI_count=roi,return_std=True,excluded_border_slices=excluded,kernel_size=kernel,wT1_threshold=threshold)

    print(results)

    return



def build_basis_bart(dictfile):

    #mrfdict = dictsearch.Dictionary()
    keys,values=read_mrf_dict(dictfile,np.arange(0.,1.01,0.1))

    u,s,vh=bart(3,"svd -e",values.T)
    
    return u

toolbox = Toolbox("script_recoInVivo_3D_machines", description="Reading Siemens 3D MRF data and performing image series reconstruction")
toolbox.add_program("build_kdata", build_kdata)
toolbox.add_program("build_coil_sensi", build_coil_sensi)
toolbox.add_program("build_volumes", build_volumes)
toolbox.add_program("build_mask", build_mask)
toolbox.add_program("build_maps", build_maps)
toolbox.add_program("build_additional_maps", build_additional_maps)
toolbox.add_program("generate_image_maps", generate_image_maps)
# toolbox.add_program("output_test", output_test)
toolbox.add_program("generate_movement_gif", generate_movement_gif)
toolbox.add_program("generate_matchedvolumes_allgroups", generate_matchedvolumes_allgroups)
toolbox.add_program("build_data_nacq", build_data_nacq)
toolbox.add_program("calculate_displacement_weights", calculate_displacement_weights)
toolbox.add_program("coil_compression", coil_compression)
toolbox.add_program("coil_compression_bart", coil_compression_bart)
toolbox.add_program("build_volumes_allbins", build_volumes_allbins)
toolbox.add_program("build_volumes_singular_allbins_registered", build_volumes_singular_allbins_registered)
toolbox.add_program("build_volumes_singular_allbins", build_volumes_singular_allbins)
toolbox.add_program("build_volumes_singular", build_volumes_singular)
toolbox.add_program("build_volumes_iterative_allbins", build_volumes_iterative_allbins)
toolbox.add_program("build_mask_from_singular_volume", build_mask_from_singular_volume)
toolbox.add_program("build_mask_full_from_mask", build_mask_full_from_mask)
toolbox.add_program("select_slices_volume", select_slices_volume)
toolbox.add_program("mrf_gendict", mrf_gendict)
toolbox.add_program("extract_singular_volume_allbins", extract_singular_volume_allbins)
toolbox.add_program("extract_allsingular_volumes_bin", extract_allsingular_volumes_bin)
toolbox.add_program("build_volumes_iterative_allbins_registered", build_volumes_iterative_allbins_registered)
toolbox.add_program("build_volumes_iterative_allbins_registered_allindex", build_volumes_iterative_allbins_registered_allindex)
toolbox.add_program("getTR", getTR)
toolbox.add_program("getGeometry", getGeometry)
toolbox.add_program("convertArrayToImage", convertArrayToImage)
toolbox.add_program("concatenateVolumes", concatenateVolumes)
toolbox.add_program("plot_deformation", plot_deformation)
toolbox.add_program("build_navigator_images", build_navigator_images)
toolbox.add_program("generate_random_weights", generate_random_weights)
toolbox.add_program("build_volumes_singular_allbins_3D", build_volumes_singular_allbins_3D)
toolbox.add_program("build_volumes_singular_allbins_3D_BART", build_volumes_singular_allbins_3D_BART)
toolbox.add_program("build_volumes_singular_allbins_3D_BART_v2", build_volumes_singular_allbins_3D_BART_v2)
toolbox.add_program("build_volumes_singular_allbins_3D_BART_inv", build_volumes_singular_allbins_3D_BART_inv)
toolbox.add_program("build_volumes_iterative", build_volumes_iterative)
toolbox.add_program("generate_dixon_volumes_for_segmentation", generate_dixon_volumes_for_segmentation)
toolbox.add_program("generate_mask_roi_from_segmentation", generate_mask_roi_from_segmentation)
toolbox.add_program("getROIresults", getROIresults)
toolbox.add_program("getROIresults_mha", getROIresults_mha)
toolbox.add_program("build_coil_images", build_coil_images)
toolbox.add_program("calculate_dcomp_voronoi_3D", calculate_dcomp_voronoi_3D)
toolbox.add_program("calculate_dcomp_pysap_3D", calculate_dcomp_pysap_3D)
toolbox.add_program("plot_image_grid_bart", plot_image_grid_bart)
toolbox.add_program("showPickle", showPickle)

if __name__ == "__main__":
    toolbox.cli()





# python script_recoInVivo_3D_machines.py build_maps --filename-volume data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_volumes_singular_allbins_registered.npy --filename-mask data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_mask.npy --filename-b1 data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_b12Dplus1_16.npy --filename-weights data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_weights.npy --dictfile mrf_dictconf_Dico2_Invivo_adjusted_2.22_reco4_w8_simmean.dict --dictfile-light mrf_dictconf_Dico2_Invivo_light_for_matching_adjusted_2.22_reco4_w8_simmean.dict --optimizer-config opt_config_iterative_singular.json --file-deformation data/InVivo/3D/patient.002.v8/meas_MID00021_FID57919_raFin_3D_tra_1x1x5mm_FULL_new_motion_volumes_allbins_denoised_gamma_0_4_deformation_map.npy --filename data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF.dat
# python script_VoxelMorph_machines.py register_allbins_to_baseline --filename-volumes data/InVivo/3D/patient.002.v8/meas_MID00021_FID57919_raFin_3D_tra_1x1x5mm_FULL_new_motion_volumes_allbins_denoised_gamma_0_4.npy --file-model data/InVivo/3D/patient.002.v8/meas_MID00021_FID57919_raFin_3D_tra_1x1x5mm_FULL_new_motion_volumes_allbins_denoised_gamma_0_4_vxm_model_weights.h5
# python script_recoInVivo_3D_machines.py build_volumes_singular_allbins_registered --filename-kdata data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_kdata.npy --filename-b1 data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_b12Dplus1_16.npy --filename-pca data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_virtualcoils_16.pkl --filename-weights data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_weights.npy --file-deformation-map data/InVivo/3D/patient.002.v8/meas_MID00021_FID57919_raFin_3D_tra_1x1x5mm_FULL_new_motion_volumes_allbins_denoised_gamma_0_4_deformation_map.npy --filename-phi mrf_dictconf_Dico2_Invivo_adjusted_2.22_reco4_w8_simmean_phi_L0_10.npy --L0 10
# python script_recoInVivo_3D_machines.py build_volumes_singular_allbins --filename-kdata data/InVivo/3D/patient.002.v8/meas_MID00024_FID57922_raFin_3D_tra_1x1x5mm_FULL_new_MRF_kdata.npy --filename-phi mrf_dictconf_Dico2_Invivo_adjusted_2.22_reco4_w8_simmean_phi_L0_10.npy --L0 10 --n-comp 16