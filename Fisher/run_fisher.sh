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
expdir=/home/xucan/data/20200525/FisherExp/PhonemeExp/
mfccdir=${expdir}/mfcc
vaddir=${expdir}/mfcc

fisher_trials=${expdir}/data/fisher_test/trials

# SRE10 trials
nnet_dir=${expdir}/exp/xvector_tf_but_test/baseline/


if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in fisher_train fisher_enroll fisher_test; do
    local/phoneme/make_mfcc_and_phoneme_posterior.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
      ${expdir}/data/${name} ${expdir}exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh ${expdir}data/${name}
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      ${expdir}/data/${name} ${expdir}/exp/make_vad $vaddir
    utils/fix_data_dir.sh ${expdir}/data/${name}
  done
fi



local/tf/run_xvector.sh --stage ${stage} --train-stage ${train_stage} \
  --data ${expdir}/data/fisher_train --nnet-dir ${nnet_dir} \
  --egs-dir ${nnet_dir}/egs

  
if [ ${stage} -le 7 ]; then
  # The fisher train data
  local/tf/extract_xvectors_23dimCMVN.sh --cmd "${train_cmd} --long 1 --mem 12G" --nj 20 \
    ${nnet_dir} ${expdir}/data/fisher_train \
    ${nnet_dir}/xvectors_fisher_train
  
  # The fisher test data
  local/tf/extract_xvectors_23dimCMVN.sh --cmd "${train_cmd} --mem 6G" --nj 20 \
    ${nnet_dir} ${expdir}/data/fisher_test \
    ${nnet_dir}/xvectors_fisher_test

  # The fisher enroll data
  local/tf/extract_xvectors_23dimCMVN.sh --cmd "${train_cmd} --mem 6G" --nj 20 \
    ${nnet_dir} ${expdir}/data/fisher_enroll \
    ${nnet_dir}/xvectors_fisher_enroll
    
fi

if [ ${stage} -le 8 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
 
  ${train_cmd} ${nnet_dir}/xvectors_fisher_train/log/compute_mean.log \
    ivector-mean scp:${nnet_dir}/xvectors_fisher_train/xvector.scp \
    ${nnet_dir}/xvectors_fisher_train/mean.vec || exit 1;

  lda_dim=100
  # This script uses LDA to decrease the dimensionality prior to PLDA.
  ${train_cmd} ${nnet_dir}/xvectors_fisher_train/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:${nnet_dir}/xvectors_fisher_train/xvector.scp ark:- |" \
    ark:${expdir}/data/fisher_train/utt2spk ${nnet_dir}/xvectors_fisher_train/transform.mat || exit 1;

  # Train an out-of-domain PLDA model.
  ${train_cmd} ${nnet_dir}/xvectors_fisher_train/log/plda.log \
    ivector-compute-plda ark:${expdir}/data/fisher_train/spk2utt \
    "ark:ivector-subtract-global-mean scp:${nnet_dir}/xvectors_fisher_train/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_fisher_train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    ${nnet_dir}/xvectors_fisher_train/plda || exit 1;

fi

if [ ${stage} -le 9 ]; then
  # Get results using the out-of-domain PLDA model.
  ${train_cmd} ${nnet_dir}/scores/log/fisher_eval_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:${nnet_dir}/xvectors_fisher_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 ${nnet_dir}/xvectors_fisher_train/plda - |" \
    "ark:ivector-mean ark:${expdir}/data/fisher_enroll/spk2utt scp:${nnet_dir}/xvectors_fisher_enroll/xvector.scp ark:- | ivector-subtract-global-mean ${nnet_dir}/xvectors_fisher_train/mean.vec ark:- ark:- | transform-vec ${nnet_dir}/xvectors_fisher_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${nnet_dir}/xvectors_fisher_train/mean.vec scp:${nnet_dir}/xvectors_fisher_test/xvector.scp ark:- | transform-vec ${nnet_dir}/xvectors_fisher_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sre10_trials' | cut -d\  --fields=1,2 |" ${nnet_dir}/scores/fisher_eval_scores || exit 1;

  #pooled_eer=$(paste $sre10_trials ${nnet_dir}/scores/sre10_eval_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "Using Out-of-Domain PLDA"
  python local/phoneme/computer_eer_dcf.py $fisher_trials ${nnet_dir}/scores/fisher_eval_scores
fi

