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
stage=7
train_stage=-1
iter=

. ./cmd.sh
. ./path.sh
set -e
. ./utils/parse_options.sh

mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

sre10_trials_female=data/sre10_core_test_female/trials
sre10_trials_male=data/sre10_core_test_male/trials
sre10_trials=data/sre10_core_test/trials


# SRE10 trials
nnet_dir=exp/xvector_tf_but_test/modelPhonemeCluster/


# Path to some, but not all of the training corpora
data_root=/mnt/workspace/project/SRE18


if [ $stage -le 0 ]; then
  local/make_sre_2010_test_core.pl /mnt/database/sre/NIST/SRE10/eval/ data/
  local/make_sre_2010_train_core.pl /mnt/database/sre/NIST/SRE10/eval/ data/
fi

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in sre10_core_train sre10_core_test; do
    local/phoneme/make_mfcc_and_phoneme_posterior.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
fi


if [ ${stage} -le 7 ]; then
  # The SRE10 test data
  local/tf/extract_xvectors_23dimCMVN.sh --cmd "${train_cmd} --mem 6G" --nj 40 \
    ${nnet_dir} data/sre10_core_test \
    ${nnet_dir}/xvectors_sre10_core_test

  # The SRE10 enroll data
  local/tf/extract_xvectors_23dimCMVN.sh --cmd "${train_cmd} --mem 6G" --nj 40 \
    ${nnet_dir} data/sre10_core_train \
    ${nnet_dir}/xvectors_sre10_core_train
    
    
fi

if [ ${stage} -le 8 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  #mkdir ${nnet_dir}/xvectors_sre10_major/
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
    --num-utts=ark:${nnet_dir}/xvectors_sre10_core_train/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_sre/plda - |" \
    "ark:ivector-mean ark:data/sre10_core_train/spk2utt scp:${nnet_dir}/xvectors_sre10_core_train/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_sre10_major/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_sre10_major/mean.vec scp:${nnet_dir}/xvectors_sre10_core_test/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_sre/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sre10_trials' | cut -d\  --fields=1,2 |" ${nnet_dir}/scores/sre10_core_eval_scores || exit 1;

  #pooled_eer=$(paste $sre10_trials ${nnet_dir}/scores/sre10_core_eval_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "Using Out-of-Domain PLDA,"
  python local/phoneme/computer_eer_dcf.py $sre10_trials ${nnet_dir}/scores/sre10_core_eval_scores
fi
