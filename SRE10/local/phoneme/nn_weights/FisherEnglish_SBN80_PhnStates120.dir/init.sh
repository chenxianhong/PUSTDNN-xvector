#list of phonemes:
PHN_LIST=../FisherEnglish_SBN80_PhnStates120.dict

#working directory
CONFIGDIR=`pwd`

cat $PHN_LIST  |sed 's/_s.//;s/_.$//' |uniq > $CONFIGDIR/phonemes
cat $CONFIGDIR/phonemes | awk '{print $1,$1}' > $CONFIGDIR/dict
cat $CONFIGDIR/phonemes | awk '{printf $1"__1\n"$1"__2\n"$1"__3\n" }' > $CONFIGDIR/states
# create recognition net
HBuild $CONFIGDIR/phonemes $CONFIGDIR/monophones_lnet.hvite
# create HMM definition file
do.HMM.sh  $CONFIGDIR/states  $CONFIGDIR/hmmdefs.hvite

echo "TARGETKIND     = USER" > $CONFIGDIR/HVite.cfg




