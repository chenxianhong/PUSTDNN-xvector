#!/bin/bash
# Copyright      2018   Hossein Zeinali (Brno University of Technology)
#                2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (mostly EERs) are inline in comments below.
#
# This example demonstrates a "bare bones" NIST SRE 2016 recipe using xvectors.
# In the future, we will add score-normalization and a more effective form of
# PLDA domain adaptation.
#
# Pretrained models are available for this recipe.
# See http://kaldi-asr.org/models.html and
# https://david-ryan-snyder.github.io/2017/10/04/model_sre16_v2.html
# for details.

train_cmd=
stage=0
train_stage=-1
iter=

. ./cmd.sh
. ./path.sh
set -e
. ./utils/parse_options.sh

mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

sre10_trials_female=data/sre10_test_female/trials
sre10_trials_male=data/sre10_test_male/trials
sre10_trials=data/sre10_test/trials


# SRE10 trials
nnet_dir=exp/xvector_tf_but_test/modelPhonemeCluster/



# Path to some, but not all of the training corpora
data_root=/mnt/workspace/project/SRE18



if [ $stage -le 0 ]; then
  # Prepare telephone and microphone speech from Mixer6.
  local/make_mx6.sh ${data_root}/ data/

  # Prepare SRE08 test and enroll. Includes some microphone speech.
  local/make_sre08.pl $data_root/SRE08/sp08-11/test/ $data_root/SRE08/sp08-11/train/ data/

  # This prepares the older NIST SREs from 2004-2006.
  local/make_sre.sh ${data_root} data/

  # Combine all SREs prior to 2010 into one dataset
  utils/combine_data.sh data/sre \
    data/sre2004 data/sre2005_train \
    data/sre2005_test data/sre2006_train \
    data/sre2006_test data/sre08 
  utils/validate_data_dir.sh --no-text --no-feats data/sre
  utils/fix_data_dir.sh data/sre

  # Prepare SWBD corpora.
  local/make_swbd_cellular1.pl ${data_root}/LDC2018E48_Comprehensive_Switchboard/LDC2001S13 \
    data/swbd_cellular1_train
  local/make_swbd_cellular2.pl ${data_root}/LDC2018E48_Comprehensive_Switchboard/LDC2004S07 \
    data/swbd_cellular2_train
  local/make_swbd2_phase1.pl ${data_root}/LDC2018E48_Comprehensive_Switchboard/LDC98S75 \
    data/swbd2_phase1_train
  local/make_swbd2_phase2.pl ${data_root}/LDC2018E48_Comprehensive_Switchboard/LDC99S79 \
    data/swbd2_phase2_train
  local/make_swbd2_phase3.pl ${data_root}/LDC2018E48_Comprehensive_Switchboard/LDC2002S06 \
    data/swbd2_phase3_train

  # Combine all SWB corpora into one dataset.
  utils/combine_data.sh data/swbd \
    data/swbd_cellular1_train data/swbd_cellular2_train \
    data/swbd2_phase1_train data/swbd2_phase2_train data/swbd2_phase3_train

  # Prepare NIST SRE 2010 coreext-coreext test data.
  local/make_sre_2010_test.pl /mnt/database/sre/NIST/SRE10/eval/ data/
  
  # Prepare NIST SRE 2010 coreext-coreext train data.
  local/make_sre_2010_train.pl /mnt/database/sre/NIST/SRE10/eval/ data/
fi


if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in sre swbd sre10_train sre10_test; do
    local/phoneme/make_mfcc_and_phoneme_posterior.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
  utils/combine_data.sh --extra-files "utt2num_frames" data/swbd_sre data/swbd data/sre
  utils/fix_data_dir.sh data/swbd_sre
fi


# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 3 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs_23dimCMVN.sh --nj 40 --cmd "$train_cmd" \
    data/swbd_sre data/swbd_sre_no_sil exp/swbd_sre_no_sil
  utils/fix_data_dir.sh data/swbd_sre_no_sil

  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  min_len=500
  mv data/swbd_sre_no_sil/utt2num_frames data/swbd_sre_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/swbd_sre_no_sil/utt2num_frames.bak > data/swbd_sre_no_sil/utt2num_frames
  utils/filter_scp.pl data/swbd_sre_no_sil/utt2num_frames data/swbd_sre_no_sil/utt2spk > data/swbd_sre_no_sil/utt2spk.new
  mv data/swbd_sre_no_sil/utt2spk.new data/swbd_sre_no_sil/utt2spk
  utils/fix_data_dir.sh data/swbd_sre_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' data/swbd_sre_no_sil/spk2utt > data/swbd_sre_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/swbd_sre_no_sil/spk2num | utils/filter_scp.pl - data/swbd_sre_no_sil/spk2utt > data/swbd_sre_no_sil/spk2utt.new
  mv data/swbd_sre_no_sil/spk2utt.new data/swbd_sre_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl data/swbd_sre_no_sil/spk2utt > data/swbd_sre_no_sil/utt2spk

  utils/filter_scp.pl data/swbd_sre_no_sil/utt2spk data/swbd_sre_no_sil/utt2num_frames > data/swbd_sre_no_sil/utt2num_frames.new
  mv data/swbd_sre_no_sil/utt2num_frames.new data/swbd_sre_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh data/swbd_sre_no_sil
fi

local/tf/run_xvector.sh --stage ${stage} --train-stage ${train_stage} \
  --data data/swbd_sre_no_sil --nnet-dir ${nnet_dir} \
  --egs-dir ${nnet_dir}/egs

  
if [ ${stage} -le 7 ]; then
  # The SRE10 test data
  local/tf/extract_xvectors_23dimCMVN.sh --cmd "${train_cmd} --long 1 --mem 12G" --nj 20 \
    ${nnet_dir} data/sre \
    ${nnet_dir}/xvectors_sre
  
  local/tf/extract_xvectors_23dimCMVN.sh --cmd "${train_cmd} --mem 6G" --nj 20 \
    ${nnet_dir} data/sre10_test \
    ${nnet_dir}/xvectors_sre10_test

  # The SRE10 enroll data
  local/tf/extract_xvectors_23dimCMVN.sh --cmd "${train_cmd} --mem 6G" --nj 20 \
    ${nnet_dir} data/sre10_train \
    ${nnet_dir}/xvectors_sre10_train
  
    
fi

if [ ${stage} -le 8 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  mkdir ${nnet_dir}/xvectors_sre10_major/
  cp ${nnet_dir}/xvectors_sre/xvector.scp ${nnet_dir}/xvectors_sre10_major/
 
  ${train_cmd} ${nnet_dir}/xvectors_sre10_major/log/compute_mean.log \
    ivector-mean scp:${nnet_dir}/xvectors_sre10_major/xvector.scp \
    ${nnet_dir}/xvectors_sre10_major/mean.vec || exit 1;

  lda_dim=100
  # This script uses LDA to decrease the dimensionality prior to PLDA.
  ${train_cmd} ${nnet_dir}/xvectors_sre/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:${nnet_dir}/xvectors_sre/xvector.scp ark:- |" \
    ark:data/sre_no10/utt2spk ${nnet_dir}/xvectors_sre/transform.mat || exit 1;

  # Train an out-of-domain PLDA model.
  ${train_cmd} ${nnet_dir}/xvectors_sre/log/plda.log \
    ivector-compute-plda ark:data/sre_no10/spk2utt \
    "ark:ivector-subtract-global-mean scp:${nnet_dir}/xvectors_sre/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    ${nnet_dir}/xvectors_sre/plda || exit 1;

  # Here we adapt the out-of-domain PLDA model to SRE16 major, a pile
  # of unlabeled in-domain data.  In the future, we will include a clustering
  # based approach for domain adaptation, which tends to work better.
  ${train_cmd} ${nnet_dir}/xvectors_sre10_major/log/plda_adapt.log \
    ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
    ${nnet_dir}/xvectors_sre/plda \
    "ark:ivector-subtract-global-mean scp:${nnet_dir}/xvectors_sre10_major/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    ${nnet_dir}/xvectors_sre10_major/plda_adapt || exit 1;
fi

if [ ${stage} -le 9 ]; then
  # Get results using the out-of-domain PLDA model.
  ${train_cmd} ${nnet_dir}/scores/log/sre10_eval_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_sre10_train/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre/plda - |" \
    "ark:ivector-mean ark:data/sre10_train/spk2utt scp:${nnet_dir}/xvectors_sre10_train/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre10_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre10_major/mean.vec scp:${nnet_dir}/xvectors_sre10_test/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sre10_trials' | cut -d\  --fields=1,2 |" ${nnet_dir}/scores/sre10_eval_scores || exit 1;

  #pooled_eer=$(paste $sre10_trials ${nnet_dir}/scores/sre10_eval_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "Using Out-of-Domain PLDA"
  python local/phoneme/computer_eer_dcf.py $sre10_trials ${nnet_dir}/scores/sre10_eval_scores
fi
