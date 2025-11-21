eval "$(conda shell.bash hook)"
conda deactivate
eval "$(conda shell.bash hook)"
conda activate mrfsim_research

find $1 -type f -name "*_seqParams.pkl" | while read -r line ; do
    echo "Processing $line"

    filedico=$(basename "$line")
    echo $filedico
    filename=$(echo $filedico | sed 's/_seqParams.pkl/.dat/')
    echo $filename
    
    filepath=$(dirname "$line")


    file_no_ext="${filename%.*}"
    # file_no_ext="${filename##*/}"
    # file_no_ext=$(echo "$filename" | cut -f 1 -d '.')

    echo ${filepath}/${file_no_ext}

    echo "Removing ${filepath}/${file_no_ext}*_corrected_offset.nii"
    rm ${filepath}/${file_no_ext}*_corrected_offset.nii

    find $1 -type f -name "${file_no_ext}*wT1.mha" | while read -r line2 ; do
        echo "Processing $line2"
        # python scripts/script_recoInVivo_3D_machines.py getGeometry --filemha ${line2} --filename ${line} --suffix "_adjusted"
        python scripts/script_recoInVivo_3D_machines.py convertArrayToImage --filevolume ${line2} --filedico ${filepath}/${filedico} --nifti True

        filemha=$(basename "$line2")
        filemha_no_ext="${filemha%.*}"
        echo $filemha
        filenii=${filemha_no_ext}.nii
        echo $filenii
        filecorrected=${filemha_no_ext}_corrected.nii
        

        eval "$(conda shell.bash hook)"
        conda deactivate
        conda activate distorsion
        gradient_unwarp.py ${filepath}/${filenii} ${filepath}/${filecorrected} siemens -g ../gradunwarp/coeff.grad --fovmin -0.4 --fovmax 0.4 -n
        conda deactivate

        eval "$(conda shell.bash hook)"
        conda activate mrfsim_research

        python scripts/script_recoInVivo_3D_machines.py convertArrayToImage --filevolume ${filepath}/${filecorrected} --filedico ${filepath}/${filedico} --suffix "_offset" --apply-offset True --nifti True --reorient False

        python scripts/script_recoInVivo_3D_machines.py convertArrayToImage --filevolume ${filepath}/${filenii} --filedico ${filepath}/${filedico} --suffix "_offset" --apply-offset True --nifti True --reorient False
            
    done

    find $1 -type f -name "${file_no_ext}*ff.mha" | while read -r line2 ; do
        echo "Processing $line2"
        # python scripts/script_recoInVivo_3D_machines.py getGeometry --filemha ${line2} --filename ${line} --suffix "_adjusted"
        python scripts/script_recoInVivo_3D_machines.py convertArrayToImage --filevolume ${line2} --filedico ${filepath}/${filedico} --nifti True

        filemha=$(basename "$line2")
        filemha_no_ext="${filemha%.*}"
        echo $filemha
        filenii=${filemha_no_ext}.nii
        echo $filenii
        filecorrected=${filemha_no_ext}_corrected.nii

        eval "$(conda shell.bash hook)"
        conda deactivate
        conda activate distorsion
        gradient_unwarp.py ${filepath}/${filenii} ${filepath}/${filecorrected} siemens -g ../gradunwarp/coeff.grad --fovmin -0.4 --fovmax 0.4 -n
        conda deactivate

        eval "$(conda shell.bash hook)"
        conda activate mrfsim_research

        python scripts/script_recoInVivo_3D_machines.py convertArrayToImage --filevolume ${filepath}/${filecorrected} --filedico ${filepath}/${filedico} --suffix "_offset" --apply-offset True --nifti True --reorient False

        python scripts/script_recoInVivo_3D_machines.py convertArrayToImage --filevolume ${filepath}/${filenii} --filedico ${filepath}/${filedico} --suffix "_offset" --apply-offset True --nifti True --reorient False
            
    done

done