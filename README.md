# PUSTDNN-xvector
This is a phoneme-unit-specific TDNN (PUSTDNN) based x-vector system for speaker verification.

It is modified based on the Tensorflow implementation of x-vector topology proposed by David Snyder in Deep Neural Network Embeddings for Text-Independent Speaker Verfication: https://github.com/hsn-zeinali/x-vector-kaldi-tf.

We use phoneme recognizer from Brno University of Technology to get phoneme probability for the PUSTDNN construcation. https://speech.fit.vutbr.cz/software/but-phonexia-bottleneck-feature-extractor.




## Usage
There are two folders "SRE10" and "Fisher". SRE10 is for the experiments on SRE 2010 dataset. Fisher is for the experiments on Fisher dataset.

For using the codes, you first should install Kaldi and move apply-cmvn-sliding-23dim.cc to kaldi⁩/⁨src⁩/featbin⁩/ and make it.

Then clone the two folders to egs/sre16 (or somewhere else that you want, by changing the symlinks to proper positions). 

If you want to test different topology, you can add a subclass of Model to local/tf/models.py and overwrite the build_model function (in the file you can see several topologies). Then you just need to pass class name to local/tf/train_dnn.py by changing --tf-model-class in local/tf/run_xvector.sh.

    In local/tf/models.py, ModelL2LossWithoutDropoutLReluAttention_63dim is our baseline. 
    It is in fact the same as ModelL2LossWithoutDropoutLReluAttention except that its input feature are 63 dimensional. 
    Only the first 23 MFCCs are used. The following 40 dimensions are phoneme posterior.
    ModelL2LossWithoutDropoutLReluAttentionPhoneme is x-vector with PUSTDNN without phoneme classification.
    ModelL2LossWithoutDropoutLReluAttentionPhonemeCluster is x-vector with PUSTDNN and phoneme classification.
    ModelL2LossWithoutDropoutLReluAttentionPhonemeCluster2layer is x-vector with PUSTDNN and phoneme classification. The first two TDNN layers are replaced with PUSTDNN layers.
    ModelL2LossWithoutDropoutLReluAttentionPhonemeClusterPequal has the same structure as ModelL2LossWithoutDropoutLReluAttentionPhonemeCluster. 
    The only difference is in the phoneme unit probability. The probabilities of all frames belonging to all phoneme units are set to be the same. 
	ModelL2LossWithoutDropoutLReluAttentionPhonemeMultitask is the multitask model.
	ModelL2LossWithoutDropoutLReluAttentionPhonemeMultitaskPequal is the 'P_equal+multitask' model. 
	ModelL2LossWithoutDropoutLReluAttentionPhonemeVector is the PV model.
	


## Requirements

    Kaldi from https://github.com/kaldi-asr/kaldi
    Python
    NumPy
    TensorFlow

