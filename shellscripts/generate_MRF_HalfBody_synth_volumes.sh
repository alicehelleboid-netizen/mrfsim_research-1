#set -e

#Processing whole body hybrid scan data
# $1 : Folder containing all the .dat
# $2 : Processing folder

targetfolder_def="data/InVivo/3D/processing"
targetfolder=${2-${targetfolder_def}}


processedfolder=$(echo ${1} | sed 's/2_Data_Raw/3_Data_Processed/')


# EXAM=-3
# echo "Generating synthetic volumes"
# find $1 -type f -name "*mrf*.dat" | sort | while read -r line ; do

#     echo $EXAM

#     filename=$(basename "$line")
#     filepath=$(dirname "$line")


#     file_mrf="${filename%.*}"
#     file_mrf_full=${targetfolder}/${file_mrf}

    
#     bash -i shellscripts/run_MRF_synthetic_volumes.sh ${file_mrf_full}

    

#     if [ $EXAM -eq -3 ]
#     then
#         find ${targetfolder} -type f -name "*${file_mrf}*ip_corrected_offset.mha" | while read -r line2 ; do
#             echo "Renaming ${line2} to ${targetfolder}/vol_001.mha"*
#             cp ${line2} ${targetfolder}/vol_001.mha

#         done
#         find ${targetfolder} -type f -name "*${file_mrf}*oop_corrected_offset.mha" | while read -r line2 ; do
#             echo "Renaming ${line2} to ${targetfolder}/vol_002.mha"
#             cp ${line2} ${targetfolder}/vol_002.mha
#         done           
#     fi

#     EXAM=$((EXAM+1))


# done


echo "######################################################"
echo "Creating whole-body synthetic volumes"
python scripts/script_recoInVivo_3D_machines.py concatenateVolumes --folder ${targetfolder} --key "ip" --suffix "_corrected_offset.mha" --overlap 0.0,0.0,50.0 --spacing 1,1,5
python scripts/script_recoInVivo_3D_machines.py concatenateVolumes --folder ${targetfolder} --key "oop" --suffix "_corrected_offset.mha" --overlap 0.0,0.0,50.0 --spacing 1,1,5

# python scripts/script_recoInVivo_3D_machines.py concatenateVolumes --folder ${targetfolder} --key "ip" --suffix "_corrected_offset.mha" --overlap 0.0,0.0,50.0
# python scripts/script_recoInVivo_3D_machines.py concatenateVolumes --folder ${targetfolder} --key "oop" --suffix "_corrected_offset.mha" --overlap 0.0,0.0,50.0


# echo "######################################################"

# copyfolder=$(echo ${1} | sed 's/2_Data_Raw/3_Data_Processed/')
# echo "Moving results to ${copyfolder}"
# mv ${targetfolder}/* ${copyfolder}

