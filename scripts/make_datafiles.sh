python create_image_file.py
./create_imageset.sh
./compute_image_mean.sh
python convert_proto_mean_to_npy.py ../data/plankton_mean.binaryproto ../data/plankton_mean.npy
