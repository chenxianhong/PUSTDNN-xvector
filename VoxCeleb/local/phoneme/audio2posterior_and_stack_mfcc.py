#!/usr/bin/env python

########################################################################################
#  copyright (C) 2017 by Anna Silnova, Pavel Matejka, Oldrich Plchot, Frantisek Grezl  #
#                         Brno Universioty of Technology                               #
#                         Faculty of information technology                            #
#                         Department of Computer Graphics and Multimedia               #
#  email             : {isilnova,matejkap,iplchot,grezl}@vut.cz                        #
########################################################################################
#                                                                                      #
#  This software and provided models can be used freely for research                   #
#  and educational purposes. For any other use, please contact BUT                     #
#  and / or LDC representatives.                                                       #
#                                                                                      #
########################################################################################

import sys, os, logging
import numpy as np
import scipy.io.wavfile as wav
import h5py

sys.path.insert(0, 'local/phoneme/')
import utils
import nn_def

sys.path.insert(0, 'local/tf/')
import kaldi_io

logging.basicConfig( format= '%(message)s',level=logging.INFO)


def read_signal(file_name):
    if os.path.isfile(file_name):
        extension= file_name.split('.')[-1]
        if extension=='wav':
            fs,signal=wav.read(file_name)
            if not fs==8000:
                logging.info("Unsupported audio format, expected audio input should be 8kHz")
        elif extension=='raw':
            signal=np.fromfile(file_name,dtype='int16')
        else:
            logging.info('Unvalid file extension, cannot load signal %s',file_name )
            signal=[]
    else :
        logging.info('File %s is missing',file_name)
        signal=[]
    return signal

def signal2fbank(signal):
    signal=utils.add_dither(signal,0.1)
    fea=utils.fbank_htk(signal,window,noverlap,fbank_mx)
    return fea

    
    
if len(sys.argv)==1:
    logging.info( "The BUT/Phonexia Bottleneck feature extractor is a tool for extracting bottleneck features from audio signal.And stack the posterior with mfcc")
    logging.info('Usage: python audio2bottleneck_and_stack_mfcc.py nn_weights wavdir mfcc_ark_file mfcc_comb_ark_file \n ** nn_weights\t weights of the pre-trained Neural Network.\n    Options are:\n\t* FisherMono -  NN trained on Fisher English with 120 phone states as targets\n\t* FisherTri - NN trained on Fisher English wtih 2423 senones as targets\n\t* BabelMulti - multilingual NN trained on 17 BABEL languages, targets are language specific phone states (3096 in total)\n ** input.wav\tinput audio file (.raw or .wav) - only sampling frequency 8kHz and linear16bit coding accepted\n ** output.fea\toutput feature file in HTK format\n ** vad.lab.gz\toptional parameter - input label file in HTK label file format with label "sil" for silence - all labels are considered as speech.\n\t If this is not provided internal energy based VAD is performed. VAD is used for file based mean normalization of input features')
    sys.exit()
elif len(sys.argv)==5:
    nn_type,wav_scp,mfcc_ark_file,mfcc_comb_ark_file=sys.argv[1:5]
    vad_file=''
    logging.info("No VAD for file %s was provided", wav_scp)
else:
    print("Wrong number of input arguments. 4 are expected")
    sys.exit()
    
        
#load correct bottleneck network weights
if nn_type=="FisherMono":
    #logging.info('Using NN trained on Fisher English with 120 phone states as targets to extract BNs')
    nn_BN='FisherEnglish_FBANK_HL500_SBN80_PhnStates120'
elif nn_type=="FisherTri":
    #logging.info('Using NN trained on Fisher English with 2423 senones as targets to extract BNs')
    nn_BN='FisherEnglish_FBANK_HL500_SBN80_triphones2423'
elif nn_type=="BabelMulti":
    #logging.info('Using multilingual NN trained on 17 BABEL languages with language specific phone states as targets to extract BNs')
    nn_BN='Babel-ML17_FBANK_HL1500_SBN80_PhnStates3096'
else:
    #logging.info('Unknown option %s for NN weights, cannot extract BNs. Valid options are: FisherMono, FisherTri, BabelMulti',nn)
    sys.exit()

if os.path.dirname(sys.argv[0])=='':
    nn_weights_BN='nn_weights/'+nn_BN+'.npz'
else:
    nn_weights_BN=os.path.dirname(sys.argv[0])+'/nn_weights/'+nn_BN+'.npz'

#define parameters to extract fbanks
window = np.hamming(200)
noverlap=120
fbank_mx= utils.mel_fbank_mx(window.size, fs=8000, NUMCHANS=24, LOFREQ=64.0, HIFREQ=3800.0)
#load NN weights
nn_weights_BN=np.load(nn_weights_BN)
left_ctx=right_ctx=15 #global context
left_ctx_bn1 = right_ctx_bn1 = nn_weights_BN['context'] #context  to extract first BN
        

#load correct weights
if nn_type=="FisherMono":
    #logging.info('Using NN trained on Fisher English with 120 phone states as targets to calculate posteriors')
    nn_Phn='FisherEnglish_SBN80_PhnStates120'
elif nn_type=="FisherTri":
    #logging.info('Using NN trained on Fisher English with 2423 senones as targets to calculate posteriors')
    nn_Phn='FisherEnglish_SBN80_triphones2423'
elif nn_type=="BabelMulti":
    #logging.info('Using multilingual NN trained on 17 BABEL languages with language specific phone states as targets to calculate posteriors')
    nn_Phn='Babel-ML17_SBN80_PhnStates3096'
else:
    #logging.info('Unknown option %s for NN weights, cannot extract posteriors. Valid options are: FisherMono, FisherTri, BabelMulti',nn)
    sys.exit()

if os.path.dirname(sys.argv[0])=='':
    nn_weights_Phn='nn_weights/'+nn_Phn+'.npz'
else:
    nn_weights_Phn=os.path.dirname(sys.argv[0])+'/nn_weights/'+nn_Phn+'.npz'
nn_weights_Phn=np.load(nn_weights_Phn)
            
            

name_mfcc_dict={ key:mat for key,mat in kaldi_io.read_mat_ark(mfcc_ark_file) }

with open(mfcc_comb_ark_file,'wb') as f:
    for line in open(wav_scp):
        line=line.strip('\n')
        name=line.split()[0]
        audio_input=line.split()[1]
        
        #extract fbanks and BN features
        try:
            mfcc=name_mfcc_dict[name]
        except:
            logging.info("There is no mfcc feautre for %s", audio_input)
            continue
        try:
            signal=read_signal(audio_input)

        except:
            logging.info("Could not read file %s, wav is broken", audio_input)
            continue

        if len(signal)==0: 
            logging.info("Could not read file %s, length is 0", audio_input)
        else:
            fea=signal2fbank(signal)
            try:
                vad=utils.read_lab_to_bool_vec(vad_file,length=len(fea))
            except:
                #logging.info("Could not read VAD file, energy-based VAD will be computed and used")
                vad=utils.compute_vad(signal)
                #logging.info("%d frames of speech detected", sum(vad))
            if sum(vad)==0:
                logging.info("All the audio of %s is silence, No features will be created", audio_input)
                continue
            fea-=np.mean(fea[vad],axis=0)
            fea= np.r_[ np.repeat(fea[[0]], left_ctx, axis=0), fea, np.repeat(fea[[-1]], right_ctx, axis=0)]
            nn_input=nn_def.preprocess_nn_input(fea,left_ctx_bn1,right_ctx_bn1)
            if False: #change to True to save also BN features, it will be saved to file output.fea.bn
                st_BN_fea,BN_fea=nn_def.create_nn_extract_st_BN(nn_input,nn_weights_BN,2)
                BN_fea=np.vstack(BN_fea)[left_ctx-left_ctx_bn1:-(right_ctx-right_ctx_bn1)]
                #BN_fea=BN_fea[vad]# uncomment if you want BN features after vad
                utils.write_htk(fea_output+'.bn', BN_fea)
            else:
                st_BN_fea=nn_def.create_nn_extract_st_BN(nn_input,nn_weights_BN,2)[0]
            st_BN_fea=np.vstack(st_BN_fea)
            #st_BN_fea=st_BN_fea[vad]# uncomment if you want SBN features after vad
            #utils.write_htk(fea_output, st_BN_fea)
            #logging.info('Features are successfully generated for file %s', audio_input )
        
        
        
            # bottleneck to posterior
            if nn_Phn=="Babel-ML17_SBN80_PhnStates3096":
                post=np.vstack(nn_def.create_nn_extract_posterior_ml(st_BN_fea,nn_weights_Phn))
            else:
                post=np.vstack(nn_def.create_nn_extract_posterior(st_BN_fea,nn_weights_Phn))
            
            # only consider phoneme level, combined its 3 states together
            (frame_num, state_num)=post.shape
            post=post.reshape(frame_num, state_num//3, 3).sum(axis=2)
        
            post = post.astype(np.float32)
            
            # stack mfcc feature with posterior.
            #os.remove(audio_input)
            mfcc_posterior_mat = np.hstack((mfcc, post))
            kaldi_io.write_mat(f, mfcc_posterior_mat, key=name)
            logging.info('Posterior are successfully generated for file %s', audio_input )
