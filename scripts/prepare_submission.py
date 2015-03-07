from scipy.misc import imread, imresize
import numpy as np
import os
import sys
import math

BATCH_SIZE = 500

# hack sys.path so we can import caffe
caffe_python_path = '~/caffe/python'
sys.path.insert(0, caffe_python_path)

# Set the default GPU that caffe will use.
import caffe
#caffe.set_device(0)
caffe.set_mode_gpu()

header = 'image,acantharia_protist_big_center,acantharia_protist_halo,acantharia_protist,amphipods,appendicularian_fritillaridae,appendicularian_s_shape,appendicularian_slight_curve,appendicularian_straight,artifacts_edge,artifacts,chaetognath_non_sagitta,chaetognath_other,chaetognath_sagitta,chordate_type1,copepod_calanoid_eggs,copepod_calanoid_eucalanus,copepod_calanoid_flatheads,copepod_calanoid_frillyAntennae,copepod_calanoid_large_side_antennatucked,copepod_calanoid_large,copepod_calanoid_octomoms,copepod_calanoid_small_longantennae,copepod_calanoid,copepod_cyclopoid_copilia,copepod_cyclopoid_oithona_eggs,copepod_cyclopoid_oithona,copepod_other,crustacean_other,ctenophore_cestid,ctenophore_cydippid_no_tentacles,ctenophore_cydippid_tentacles,ctenophore_lobate,decapods,detritus_blob,detritus_filamentous,detritus_other,diatom_chain_string,diatom_chain_tube,echinoderm_larva_pluteus_brittlestar,echinoderm_larva_pluteus_early,echinoderm_larva_pluteus_typeC,echinoderm_larva_pluteus_urchin,echinoderm_larva_seastar_bipinnaria,echinoderm_larva_seastar_brachiolaria,echinoderm_seacucumber_auricularia_larva,echinopluteus,ephyra,euphausiids_young,euphausiids,fecal_pellet,fish_larvae_deep_body,fish_larvae_leptocephali,fish_larvae_medium_body,fish_larvae_myctophids,fish_larvae_thin_body,fish_larvae_very_thin_body,heteropod,hydromedusae_aglaura,hydromedusae_bell_and_tentacles,hydromedusae_h15,hydromedusae_haliscera_small_sideview,hydromedusae_haliscera,hydromedusae_liriope,hydromedusae_narco_dark,hydromedusae_narco_young,hydromedusae_narcomedusae,hydromedusae_other,hydromedusae_partial_dark,hydromedusae_shapeA_sideview_small,hydromedusae_shapeA,hydromedusae_shapeB,hydromedusae_sideview_big,hydromedusae_solmaris,hydromedusae_solmundella,hydromedusae_typeD_bell_and_tentacles,hydromedusae_typeD,hydromedusae_typeE,hydromedusae_typeF,invertebrate_larvae_other_A,invertebrate_larvae_other_B,jellies_tentacles,polychaete,protist_dark_center,protist_fuzzy_olive,protist_noctiluca,protist_other,protist_star,pteropod_butterfly,pteropod_theco_dev_seq,pteropod_triangle,radiolarian_chain,radiolarian_colony,shrimp_caridean,shrimp_sergestidae,shrimp_zoea,shrimp-like_other,siphonophore_calycophoran_abylidae,siphonophore_calycophoran_rocketship_adult,siphonophore_calycophoran_rocketship_young,siphonophore_calycophoran_sphaeronectes_stem,siphonophore_calycophoran_sphaeronectes_young,siphonophore_calycophoran_sphaeronectes,siphonophore_other_parts,siphonophore_partial,siphonophore_physonect_young,siphonophore_physonect,stomatopod,tornaria_acorn_worm_larvae,trichodesmium_bowtie,trichodesmium_multiple,trichodesmium_puff,trichodesmium_tuft,trochophore_larvae,tunicate_doliolid_nurse,tunicate_doliolid,tunicate_partial,tunicate_salp_chains,tunicate_salp,unknown_blobs_and_smudges,unknown_sticks,unknown_unclassified'


model_file = '../plankton_caffenet/plankton_deploy.prototxt'
weights_file = '../plankton_caffenet/snapshot_iter_10000.caffemodel'
net = caffe.Net(model_file, weights_file, caffe.TEST)

directory = '../data/test'
outputFile = open('../data/submission.csv', 'w')
outputFile.write(header)

images = [item for item in os.listdir(directory)]
num_images = len(images)
num_batches = int(math.ceil(num_images / float(BATCH_SIZE)))

mean_file =  '../data/plankton_mean.npy'
mean = np.load(mean_file)
low, high = np.min(mean), np.max(mean)
mean = 255.0 * (mean - low) / (high - low)

input_data = np.zeros((len(images), 1, 227, 227))
for batch_num in range(num_batches):
  print "Batch:", batch_num, "/", num_batches
  for i in range(BATCH_SIZE):
    img_index = i + BATCH_SIZE * batch_num
    if (img_index >= num_images): break
    image = images[img_index]
    
    #if image == '.DS_Store': continue
    img = imread('../data/test/' + image)
  
    # Resize image to same size as mean
    H_mean, W_mean = mean.shape[1:]
    img = imresize(img, (H_mean, W_mean))
    
    # Reshape from (H, W, K) to (K, H, W)
    #img = img.transpose(2, 0, 1)
    
    # Subtract mean
    img = img - mean
    
    # Crop to input size of network
    H_in, W_in = net.blobs['data'].data.shape[2:]
    H0 = (H_mean - H_in) / 2
    H1 = H0 + H_in
    W0 = (W_mean - W_in) / 2
    W1 = W0 + W_in
    img = img[:, H0:H1, W0:W1]
  
    # Copy input data to input blob of the network
    net.blobs['data'].data[i] = img

  # Call net forward
  net.forward()
  
  # Pull out the probabilities from the network and print them
  probs = net.blobs['prob'].data
  for i in range(BATCH_SIZE):
   img_index = i + BATCH_SIZE * batch_num
   if (img_index >= num_images): break
   image = images[img_index]
   outputFile.write(image +',' + ','.join(['%.16f' % num for num in probs[i]]) + '\n')

outputFile.close()
