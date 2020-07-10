#!/bin/bash

# Copyright 2012-2016  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
phoneme_nn_type=FisherMono
# End configuration section.


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
   echo "Usage: $0 [options] <data-dir> <mfcc-ark> ";
   echo "e.g.: $0 data/sre16.scp mfcc/mfcc_sre16.ark"
   echo "Note: <log-dir> defaults to <data-dir>/log, and <mfccdir> defaults to <data-dir>/data"
   echo "Options: "
   echo "  --phoneme-nn-type (FisherMono, )                      # config passed to phoneme_nn_type "
   exit 1;
fi

#wav_scp=exp/make_mfcc/wav_sre16_major.1.scp
#mfcc_ark=mfcc/raw_mfcc_sre16_major.1.ark

wav_scp=$1
mfcc_ark=$2
mfcc_comb_ark=$3


wavdir=${wav_scp%.*}
wavdir=${wavdir//./_}
mkdir -p $wavdir || exit 1;


# turn sph to wav
while read line
do
    line=${line%|*} 
    utterance_name=`echo $line | cut -d ' ' -f 1`
    `echo $line | cut -d ' ' -f 2-` > ${wavdir}/${utterance_name}.wav
done < $wav_scp


# compute phoneme posterior and stack it with mfcc, then save in mfcc_comb_ark files
cat $wav_scp | cut -d ' ' -f 1 > ${wavdir}/utterance_names

MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python local/phoneme/audio2posterior_and_stack_mfcc.py $phoneme_nn_type \
    $wavdir $mfcc_ark $mfcc_comb_ark \
    || exit 1

rm $mfcc_ark

rm ${wavdir}/ -r

