#set -e

#Processing whole body hybrid scan data
# $1 : Folder containing all the .dat
# $2 : Nb segments respi (optional - default 1400)
# $3 : Nb segments mrf (optional - default 1024)
# $4 : Example slice 1 (optional - default 46)
# $5 : Example slice 2 (optional - default 20)


# New dictionaries
# DICTFILE="./mrf_dict_optim/dico_TR2.1_reco2.99.pkl"

DICTFILE_LOWBODY="./mrf_dict/dico_TR2.14_reco5.0.pkl"

# Old dictionaries

# DICTFILE="mrf_dictconf_Dico2_Invivo_overshoot_B1df_random_FA_varsp_varparam_allparams_2.22_2.17_reco2.99_w8_simmean.dict"
# DICTFILE_LIGHT="mrf_dictconf_Dico2_Invivo_light_for_matching_overshoot_B1df_random_FA_varsp_varparam_allparams_2.22_2.17_reco2.99_w8_simmean.dict"

# DICTFILE_LOWBODY="mrf_dictconf_Dico2_Invivo_overshoot_B1df_random_FA_varsp_varparam_allparams_2.22_2.12_reco2.99_w8_simmean.dict"
# DICTFILE_LOWBODY_LIGHT="mrf_dictconf_Dico2_Invivo_light_for_matching_overshoot_B1df_random_FA_varsp_varparam_allparams_2.22_2.12_reco2.99_w8_simmean.dict"



targetfolder="data/InVivo/3D/processing"
targetfolder="/mnt/ssd2/slioussarenko/mrf-sim/data/processing"

NSEGMENTS_def=1400
NSEGMENTS_def_mrf=1024
SLICE1_def=46
SLICE2_def=20

NSEGMENTS=${2-${NSEGMENTS_def}}
NSEGMENTS_MRF=${3-${NSEGMENTS_def_mrf}}

SLICE1=${4-${SLICE1_def}}
SLICE2=${5-${SLICE2_def}}


echo "I AM NEW ######################################################"

EXAM=-3
find $1 -type f -name "*mrf*.dat" | sort | while read -r line ; do

    echo $EXAM

    filename=$(basename "$line")
    filepath=$(dirname "$line")


    file_mrf="${filename%.*}"


    if [ $EXAM -eq -4 ]
    then
        
        EXAM=$((EXAM+1))
        continue
    fi

    if [ $EXAM -eq -3 ]
    then
        EXAM=$((EXAM+1))
        continue
    fi

    if [ $EXAM -eq -2 ]
    then
        EXAM=$((EXAM+1))
        continue
    fi

    if [ $EXAM -eq -1 ]
    then
        EXAM=$((EXAM+1))
        continue
    fi

    echo ${filepath}/${file_mrf}

    echo "Copying ${filepath}/${filename} in ${targetfolder}"
    # cp ${filepath}/${filename} ${targetfolder}
    echo "Copied ${filepath}/${filename} in ${targetfolder}"

    file_mrf_full=${targetfolder}/${file_mrf}
    echo ${file_mrf_full}

    echo "Processing lower body mrf"  
        
    # sh shellscripts/run_MRF_3D_denoised_bart_new_dict.sh ${file_mrf_full} ${DICTFILE_LOWBODY} ${EXAM}
    echo "##############################################"

    if [ $EXAM -eq -2 ]
    then
        echo "Thighs: Automatic segmentation and ROI values"
        yes | bash -i shellscripts/run_MRF_mask_ROI_wholebody.sh ${file_mrf_full} museg-thighs:model3
        yes | bash -i shellscripts/run_MRF_offseted_segmentations.sh ${file_mrf_full} museg-thighs:model3 roi_thighs
    fi

    if [ $EXAM -eq -1 ]
    then
        echo "Legs: Automatic segmentation and ROI values"
        yes | bash -i shellscripts/run_MRF_mask_ROI_wholebody.sh ${file_mrf_full} museg-legs:model1
        yes | bash -i shellscripts/run_MRF_offseted_segmentations.sh ${file_mrf_full} museg-legs:model1 roi_legs
    fi

    

    EXAM=$((EXAM+1))
done




echo "######################################################"
echo "Generating whole-body maps"
echo "######################################################"
echo "Correcting maps for distortion and applying offset"

sh shellscripts/run_correct_maps.sh ${targetfolder}

echo "######################################################"
echo "Creating whole-body FF and T1H2O maps"
python scripts/script_recoInVivo_3D_machines.py concatenateVolumes --folder ${targetfolder} --key "wT1" --overlap 0.0,0.0,50.0
python scripts/script_recoInVivo_3D_machines.py concatenateVolumes --folder ${targetfolder} --key "ff" --overlap 0.0,0.0,50.0

sh shellscripts/generate_MRF_HalfBody_synth_volumes.sh ${1} ${targetfolder}

echo "######################################################"

copyfolder=$(echo ${1} | sed 's/2_Data_Raw/3_Data_Processed/')
echo "Moving results to ${copyfolder}"
mv ${targetfolder}/* ${copyfolder}

# test