#!/usr/bin/env bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
# Apache 2.0.
#
# See ../README.txt for more info on data required.
# Results (mostly equal error-rates) are inline in comments below.


train_cmd=
stage=10
train_stage=-1
iter=

. ./cmd.sh
. ./path.sh
set -e
. ./utils/parse_options.sh
#expdir=/home/xucan/data/20200525/voxcelebExp/
expdir=`pwd`
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

# The trials file is downloaded by local/make_voxceleb1_v2.pl.
voxceleb1_trials=${expdir}/data/voxceleb1_test/trials
voxceleb1_root=/home/xucan/data/ccdata/voxceleb/1/
voxceleb2_root=/home/xucan/data/ccdata/voxceleb/2/


# path to model

#nnet_dir=${expdir}/exp/baseline_atten_RVegs/
#nnet_dir=${expdir}/exp/model96nodeCluster/
#nnet_dir=${expdir}/exp/model80nodeCluster/
#nnet_dir=${expdir}/exp/model96nodeCluster2layer/
nnet_dir=${expdir}/exp/modelMultitask2layer/
nnet_dir=${expdir}/exp/modelMultitask3layer/
nnet_dir=${expdir}/exp/modelMultitask4layer/
nnet_dir=${expdir}/exp/modelPvector/
#nnet_dir=${expdir}/exp/model96nodePequal/
#nnet_dir=${expdir}/exp/model80nodePequal/
#nnet_dir=${expdir}/exp/model96nodePatten/
nnet_dir=${expdir}/exp/modelMultitaskPequal/



if [ $stage -le 0 ]; then
  
  # This script creates data/voxceleb1_test and data/voxceleb1_train for latest version of VoxCeleb1.
  # Our evaluation set is the test portion of VoxCeleb1.
  local/make_voxceleb1_v2.pl $voxceleb1_root test ${expdir}/data/voxceleb1_test
  

  local/make_voxceleb2.pl $voxceleb2_root dev ${expdir}/data/train
  # This should give 5994 speakers and 1092009 utterances.

fi


if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in voxceleb1_test train; do
    local/phoneme/make_mfcc_and_phoneme_posterior.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 10 --cmd "$train_cmd" \
      ${expdir}/data/${name} ${expdir}/exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh ${expdir}/data/${name}
    sid/compute_vad_decision.sh --nj 15 --cmd "$train_cmd" \
      ${expdir}/data/${name} ${expdir}/exp/make_vad $vaddir
    utils/fix_data_dir.sh ${expdir}/data/${name}
  done
fi

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 3 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs_23dimCMVN.sh --nj 10 --cmd "$train_cmd" \
    ${expdir}/data/train ${expdir}/data/train_no_sil ${expdir}/exp/train_no_sil
  utils/fix_data_dir.sh ${expdir}/data/train_no_sil

  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  min_len=500
  mv ${expdir}/data/train_no_sil/utt2num_frames ${expdir}/data/train_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' ${expdir}/data/train_no_sil/utt2num_frames.bak > ${expdir}/data/train_no_sil/utt2num_frames
  utils/filter_scp.pl ${expdir}/data/train_no_sil/utt2num_frames data/train_no_sil/utt2spk > ${expdir}/data/train_no_sil/utt2spk.new
  mv ${expdir}/data/train_no_sil/utt2spk.new ${expdir}/data/train_no_sil/utt2spk
  utils/fix_data_dir.sh ${expdir}/data/train_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' ${expdir}/data/train_no_sil/spk2utt > ${expdir}/data/train_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/train_no_sil/spk2num | utils/filter_scp.pl - data/train_no_sil/spk2utt > data/train_no_sil/spk2utt.new
  mv ${expdir}/data/train_no_sil/spk2utt.new ${expdir}/data/train_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl ${expdir}/data/train_no_sil/spk2utt > ${expdir}/data/train_no_sil/utt2spk

  utils/filter_scp.pl data/train_no_sil/utt2spk data/train_no_sil/utt2num_frames > data/train_no_sil/utt2num_frames.new
  mv ${expdir}/data/train_no_sil/utt2num_frames.new ${expdir}/data/train_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh ${expdir}/data/train_no_sil
fi


local/tf/run_xvector.sh --stage ${stage} --train-stage ${train_stage} \
  --data ${expdir}/data/train_no_sil --nnet-dir ${nnet_dir} \
  --egs-dir ${nnet_dir}/egs



if [ ${stage} -le 7 ]; then

  gpu_bool=true
  gpu_init=1
  gpu_nj=1

  # Extract x-vectors used in the evaluation.
  local/tf/extract_xvectors_23dimCMVN.sh --cmd "${train_cmd} --mem 6G"  --use-gpu ${gpu_bool} --gpu_index ${gpu_init} --nj ${gpu_nj}\
    ${nnet_dir} ${expdir}/data/voxceleb1_test \
    ${nnet_dir}/xvectors_voxceleb1_test  
  
  # Extract x-vectors for centering, LDA, and PLDA training.
  local/tf/extract_xvectors_23dimCMVN.sh --cmd "${train_cmd} --mem 6G"  --use-gpu ${gpu_bool} --gpu_index ${gpu_init} --nj ${gpu_nj}\
    ${nnet_dir} ${expdir}/data/train \
    ${nnet_dir}/xvectors_train
  
fi


if [ ${stage} -le 8 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  $train_cmd $nnet_dir/xvectors_train/log/compute_mean.log \
    ivector-mean scp:$nnet_dir/xvectors_train/xvector.scp \
    $nnet_dir/xvectors_train/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=100
  $train_cmd $nnet_dir/xvectors_train/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_train/xvector.scp ark:- |" \
    ark:${expdir}/data/train/utt2spk $nnet_dir/xvectors_train/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd $nnet_dir/xvectors_train/log/plda.log \
    ivector-compute-plda ark:${expdir}/data/train/spk2utt \
    "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_train/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $nnet_dir/xvectors_train/plda || exit 1;
	
fi

if [ $stage -le 9 ]; then
  $train_cmd ${nnet_dir}/scores/log/voxceleb1_test_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $nnet_dir/xvectors_train/plda - |" \
    "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train/mean.vec scp:$nnet_dir/xvectors_voxceleb1_test/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train/mean.vec scp:$nnet_dir/xvectors_voxceleb1_test/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" ${nnet_dir}/scores/scores_voxceleb1_test || exit 1;


    python local/phoneme/computer_eer_dcf.py $voxceleb1_trials ${nnet_dir}/scores/scores_voxceleb1_test
fi


if [ $stage -le 10 ]; then
  eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials ${nnet_dir}/scores/scores_voxceleb1_test) 2> /dev/null`
  mindcf1=`sid/compute_min_dcf.py --p-target 0.001 ${nnet_dir}/scores/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.05 ${nnet_dir}/scores/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.001): $mindcf1"
  echo "minDCF(p-target=0.05): $mindcf2"
  # EER: 3.128%
  # minDCF(p-target=0.01): 0.3258
  # minDCF(p-target=0.001): 0.5003
  #
  # For reference, here's the ivector system from ../v1:
  # EER: 5.329%
  # minDCF(p-target=0.01): 0.4933
  # minDCF(p-target=0.001): 0.6168
fi



