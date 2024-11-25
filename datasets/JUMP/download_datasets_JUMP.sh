#!/bin/bash

# set remote root
remote_root="s3://cellpainting-gallery"

# Set the batch data to be downloaded
project_name="cpg0000-jump-pilot"
source_name="source_4"
batch_name="2020_11_04_CPJUMP1"

# set the plate to be downloaded
# folder size of each plate is about 60 GB.

# U2OS 48 h
plate_names=(
"BR00117010__2020-11-08T18_18_00-Measurement1"
"BR00117011__2020-11-08T19_57_47-Measurement1"
"BR00117012__2020-11-08T14_58_34-Measurement1"
"BR00117013__2020-11-08T16_38_19-Measurement1"
)

# U2OS 24 h
plate_names=(
"BR00116995__2020-11-06T02_41_05-Measurement1"
"BR00117024__2020-11-06T04_20_37-Measurement1"
"BR00117025__2020-11-06T06_00_19-Measurement1"
"BR00117026__2020-11-06T07_39_45-Measurement1"
)

# A549 48 h
plate_names=(
"BR00117015__2020-11-10T23_51_39-Measurement1"
"BR00117016__2020-11-11T02_32_26-Measurement1"
"BR00117017__2020-11-10T18_25_46-Measurement1"
"BR00117019__2020-11-10T21_10_40-Measurement1"
)

# A549 24 h
plate_names=(
"BR00116991__2020-11-05T19_51_35-Measurement1"
"BR00116992__2020-11-05T21_31_31-Measurement1"
"BR00116993__2020-11-05T23_11_39-Measurement1"
"BR00116994__2020-11-06T00_59_44-Measurement1"
)

# set save root dir (local)
save_root=$1

if [ -z "$save_root" ]
then
    echo "\$save_root is empty. Please set \$save_root (See README.md for details.)"
else
    # run download
    for plate_name in "${plate_names[@]}" ; do

      # parse id from plate_name
      id=${plate_name:0:10}

      # set remote url
      illum_remote_url="${remote_root}/${project_name}/${source_name}/images/${batch_name}/illum/${id}/"
      image_remote_url="${remote_root}/${project_name}/${source_name}/images/${batch_name}/images/${plate_name}/"
      loadcsv_remote_url="${remote_root}/${project_name}/${source_name}/workspace/load_data_csv/${batch_name}/${id}/"

      # set save dir
      illum_save_dir=${illum_remote_url/$remote_root/$save_root}
      image_save_dir=${image_remote_url/$remote_root/$save_root}
      loadcsv_save_dir=${loadcsv_remote_url/$remote_root/$save_root}

      # mkdir save dir
      mkdir -p ${illum_save_dir}
      mkdir -p ${image_save_dir}
      mkdir -p ${loadcsv_save_dir}

      # cp files from remote to local
      aws s3 cp --recursive ${illum_remote_url} ${illum_save_dir} --no-sign-request
      aws s3 cp --recursive ${image_remote_url} ${image_save_dir} --no-sign-request
      aws s3 cp --recursive ${loadcsv_remote_url} ${loadcsv_save_dir} --no-sign-request
    done

    # download metadata
    metadata_remote_url="${remote_root}/${project_name}/${source_name}/workspace/metadata/external_metadata/"
    platemaps_remote_url="${remote_root}/${project_name}/${source_name}/workspace/metadata/platemaps/${batch_name}/"

    metadata_save_dir=${metadata_remote_url/$remote_root/$save_root}
    platemaps_save_dir=${platemaps_remote_url/$remote_root/$save_root}

    mkdir -p ${metadata_save_dir}
    mkdir -p ${platemaps_save_dir}

    aws s3 cp --recursive ${metadata_remote_url} ${metadata_save_dir} --no-sign-request
    aws s3 cp --recursive ${platemaps_remote_url} ${platemaps_save_dir} --no-sign-request
fi