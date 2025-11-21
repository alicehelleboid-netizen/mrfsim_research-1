set -e

#Step 2 : Uses motion field to build the MRF singular volumes from MRF scan and perform fingerprinting to build the maps 
# $1 : MRF scan data file e.g. data/InVivo/3D/patient.003.v17/meas_MID00021_FID67001_raFin_3D_tra_1x1x5mm_FULL_new_mrf.dat
# $2 : Full MRF dictionary e.g. mrf_dict_optim/dico_TR2.49_reco2.99.pkl
# $3 : Nb segments (optional - default 1024)
# $4 : Example slice 1 (optional - default 180)
# $5 : Example slice 2 (optional - default 80)
# $6 : Index for parsing header file - default -1


INDEX_def=-1
NSEGMENTS_def=1024
SLICE1_def=180
SLICE2_def=80

GAMMA=0.7
GAMMASTR=$(echo "$GAMMA" | sed "s/\./"_"/")

NCOMP=12
NBINS=5
NSING=6
NITERTV=0
NSEGMENTS=${3-${NSEGMENTS_def}}
SLICE1=${4-${SLICE1_def}}
SLICE2=${5-${SLICE2_def}}
INDEX=${6-${INDEX_def}}

SIMUS=1
US=1


# #Extracting k-space and navigator data
# echo "######################################################"
# echo "Extracting k-space and navigator data"
# python scripts/script_recoInVivo_3D_machines.py build_kdata --filename $1.dat --index ${INDEX} #--nb-rep 40 #--dens-adj False

# # rm $1.npy
# rm $1.dat

# echo "Building navigator images to help with channel choice"
# python scripts/script_recoInVivo_3D_machines.py build_navigator_images --filename-nav-save $1_nav.npy
# cp $1_image_nav.jpg /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo
# cp $1_image_nav_diff.jpg /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo


# echo "######################################################"
# echo "Based on the navigator images, what is the channel with best contrast for motion estimation ?"
# read CHANNEL
# echo "Channel $CHANNEL will be used for motion estimation"

# python scripts/script_recoInVivo_3D_machines.py calculate_displacement_weights --filename-nav-save $1_nav.npy --bottom -20 --top 45 --incoherent False --nb-segments ${NSEGMENTS} --ntimesteps 1 --lambda-tv 0 --equal-spoke-per-bin True --ch $CHANNEL --nbins $NBINS --retained-categories "0,1,2,3,4" --sim-us $SIMUS --us $US --interp-bad-correl True --seasonal-adj True --randomize True --pad 0 --nspoke-per-z 1 #--soft-weight True #--stddisp 1.5
# # python scripts/script_recoInVivo_3D_machines.py calculate_displacement_weights --filename-nav-save $1_nav.npy --bottom -5 --top 15 --incoherent False --nb-segments ${NSEGMENTS} --ntimesteps 1 --lambda-tv 0 --equal-spoke-per-bin True --ch $CHANNEL --nbins $NBINS --retained-categories "0,1,2,3,4" --sim-us $SIMUS --us $US --interp-bad-correl True --seasonal-adj True --randomize True --pad 0 --nspoke-per-z 1 #--soft-weight True #--stddisp 1.5

# cp $1_displacement.jpg /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo

# # Coil compression
# echo "######################################################"
# echo "Coil Compression $NCOMP virtual coils"
# python scripts/script_recoInVivo_3D_machines.py coil_compression_bart --filename-kdata $1_kdata.npy --n-comp $NCOMP --spoke-start 400 --filename-seqParams $1_seqParams.pkl --lowmem True

# # exit

# rm $1_kdata.npy

# # #Rebuild singular volumes for all bins
# echo "######################################################"
# echo "Rebuilding singular volumes for all bins"
# python scripts/script_recoInVivo_3D_machines.py build_volumes_singular_allbins_3D --filename-kdata $1_bart${NCOMP}_kdata.npy --filename-seqParams $1_seqParams.pkl --L0 $NSING --dictfile $2 --useGPU False --filename-b1 $1_bart${NCOMP}_b12Dplus1_${NCOMP}.npy --filename-weights $1_weights.npy --gating-only True

# rm $1_bart${NCOMP}_kdata.npy

# python scripts/script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volumes_singular_allbins.npy --sl ${SLICE1} --l 0
# cp $1_bart${NCOMP}_volumes_singular_allbins.npy_sl${SLICE1}_moving_singular_l0.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo
# python scripts/script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volumes_singular_allbins.npy --sl ${SLICE2} --l 0
# cp $1_bart${NCOMP}_volumes_singular_allbins.npy_sl${SLICE2}_moving_singular_l0.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo

# # exit

# echo "######################################################"
# echo "Extracting one singular volume for all bins"
# python scripts/script_recoInVivo_3D_machines.py extract_singular_volume_allbins --file-volume $1_bart${NCOMP}_volumes_singular_allbins.npy --l 0

# # exit

# echo "######################################################"
# echo "Denoising singular volume for all bins"
# python scripts/script_recoInVivo_3D_machines.py build_volumes_iterative_allbins --filename-volume $1_bart${NCOMP}_volume_singular_l0_allbins.npy --filename-b1 $1_bart${NCOMP}_b12Dplus1_${NCOMP}.npy --mu-TV 0.01 --mu-bins 0. --gamma $GAMMA --niter ${NITERTV} --filename-weights $1_weights.npy --filename-seqParams $1_seqParams.pkl --weights-TV 1.0,1.0,1.0 --isgamma3D True

# python scripts/script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volume_singular_l0_allbins_gamma_$GAMMASTR.npy --sl ${SLICE1}
# cp $1_bart${NCOMP}_volume_singular_l0_allbins_gamma_$GAMMASTR.npy_sl${SLICE1}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo
# python scripts/script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volume_singular_l0_allbins_gamma_$GAMMASTR.npy --sl ${SLICE2}
# cp $1_bart${NCOMP}_volume_singular_l0_allbins_gamma_$GAMMASTR.npy_sl${SLICE2}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo

# exit
#Initialization of deformation fields
echo "######################################################"
echo "Initialization of deformation field"
python scripts/script_VoxelMorph_machines.py train_voxelmorph --filename-volumes $1_bart${NCOMP}_volume_singular_l0_allbins_gamma_${GAMMASTR}.npy --file-config-train config/config_train_voxelmorph_highres_shell.json --nepochs 1500 --kept-bins "0,1,2,3,4" --excluded 10 --us 4 #--lr 0.0002 --init-weights $1_volumes_allbins_denoised_gamma_0_8_vxm_model_weights.h5
python scripts/script_VoxelMorph_machines.py register_allbins_to_baseline --filename-volumes $1_bart${NCOMP}_volume_singular_l0_allbins_gamma_${GAMMASTR}.npy --file-model $1_bart${NCOMP}_volume_singular_l0_allbins_gamma_${GAMMASTR}_vxm_model_weights.h5 --file-config-train config/config_train_voxelmorph_highres_shell.json --axis 0

python scripts/script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volume_singular_l0_allbins_gamma_${GAMMASTR}_registered.npy --sl ${SLICE1}
cp $1_bart${NCOMP}_volume_singular_l0_allbins_gamma_${GAMMASTR}_registered.npy_sl${SLICE1}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo
python scripts/script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volume_singular_l0_allbins_gamma_${GAMMASTR}_registered.npy --sl ${SLICE2}
cp $1_bart${NCOMP}_volume_singular_l0_allbins_gamma_${GAMMASTR}_registered.npy_sl${SLICE2}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo

NITER=0

python scripts/script_recoInVivo_3D_machines.py build_volumes_iterative_allbins_registered_allindex --filename-volume $1_bart${NCOMP}_volumes_singular_allbins.npy --filename-b1 $1_bart${NCOMP}_b12Dplus1_${NCOMP}.npy --niter 0 --file-deformation $1_bart${NCOMP}_volume_singular_l0_allbins_gamma_${GAMMASTR}_deformation_map.npy --filename-weights $1_weights.npy --axis 0 --filename-seqParams $1_seqParams.pkl #--us 2

python scripts/script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volumes_singular_allbins.npy_volumes_allbins_registered_allindex.npy --sl ${SLICE1}
cp $1_bart${NCOMP}_volumes_singular_allbins.npy_volumes_allbins_registered_allindex.npy_sl${SLICE1}_moving_singular_l0.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo
python scripts/script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volumes_singular_allbins.npy_volumes_allbins_registered_allindex.npy --sl ${SLICE2}
cp $1_bart${NCOMP}_volumes_singular_allbins.npy_volumes_allbins_registered_allindex.npy_sl${SLICE2}_moving_singular_l0.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo




REF=0
NITER=0

if [ $((NITER)) -eq 0 ]
then
  VOLUMESSUFFIX=ref${REF}
else
  VOLUMESSUFFIX=ref${REF}_it$(($NITER-1))
fi
echo $VOLUMESSUFFIX

python scripts/script_recoInVivo_3D_machines.py build_volumes_iterative_allbins_registered --index-ref ${REF} --filename-volume $1_bart${NCOMP}_volumes_singular_allbins.npy  --niter $NITER --file-deformation $1_bart${NCOMP}_volume_singular_l0_allbins_gamma_${GAMMASTR}_deformation_map.npy --filename-weights $1_weights.npy --use-wavelet True --lambda-wav 5e-13 --mu 1 --kept-bins "0,1,2,3,4" --filename-seqParams $1_seqParams.pkl --filename-b1 $1_bart${NCOMP}_b12Dplus1_${NCOMP}.npy --gating-only True --axis 0 #--filename-b1 data/InVivo/3D/patient.003.v19/meas_MID00088_FID70345_raFin_3D_tra_0_8x0_8x3mm_FULL_new_mrf_us2_b12Dplus1_12.npy


python scripts/script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volumes_singular_allbins.npy_volumes_allbins_registered_${VOLUMESSUFFIX}.npy --sl ${SLICE1}
cp $1_bart${NCOMP}_volumes_singular_allbins.npy_volumes_allbins_registered_${VOLUMESSUFFIX}.npy_sl${SLICE1}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo
python scripts/script_recoInVivo_3D_machines.py generate_movement_gif --file-volume $1_bart${NCOMP}_volumes_singular_allbins.npy_volumes_allbins_registered_${VOLUMESSUFFIX}.npy --sl ${SLICE2}
cp $1_bart${NCOMP}_volumes_singular_allbins.npy_volumes_allbins_registered_${VOLUMESSUFFIX}.npy_sl${SLICE2}_moving_singular.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo


# python scripts/script_recoInVivo_3D_machines.py build_mask_from_singular_volume --filename-volume $1_bart${NCOMP}_volumes_singular_allbins.npy_volumes_allbins_registered_${VOLUMESSUFFIX}.npy --l 0 --threshold 0.015 --it 2
python scripts/script_recoInVivo_3D_machines.py build_mask_from_singular_volume --filename-volume $1_bart${NCOMP}_volumes_singular_allbins.npy_volumes_allbins_registered_${VOLUMESSUFFIX}.npy --l 0 --threshold 0.03 --it 2

cp $1_bart${NCOMP}_volumes_singular_allbins_volumes_allbins_registered_${VOLUMESSUFFIX}_l0_mask.gif /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#9_2021_MR_MyoMap/3_Data_Processed/log_MRF_MoCo

# #Build maps for all bins
echo "######################################################"
echo "Building MRF maps for all iter"
#VOLUMESSUFFIX=ref0_it1
python scripts/script_recoInVivo_3D_machines.py build_maps --filename-volume "$1_bart${NCOMP}_volumes_singular_allbins.npy_volumes_allbins_registered_${VOLUMESSUFFIX}.npy" --filename-mask "$1_bart${NCOMP}_volumes_singular_allbins_volumes_allbins_registered_${VOLUMESSUFFIX}_l0_mask.npy" --dico-full-file $2 

python scripts/script_recoInVivo_3D_machines.py generate_image_maps --filename-map $1_bart${NCOMP}_volumes_singular_allbins_volumes_allbins_registered_${VOLUMESSUFFIX}_CF_iterative_2Dplus1_MRF_map.pkl --filename-seqParams $1_seqParams.pkl


# #Build maps for all bins
echo "######################################################"
echo "Reorienting maps and correcting for distortion"
filepath=$(dirname "$1.dat")
bash -i shellscripts/run_correct_maps.sh ${filepath}

# cp $1*.nii /mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/#6_2024_RespiMRF/2_Data_Raw/Spherical/dia.volunteer.011.v1