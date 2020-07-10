#!/bin/bash 

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


# At first you need to modify the script bottleneck2posterior.py so that it save 
# posteriors to HTK feature file by uncomenting the line which starts - #utils.write_htk 

#set -e  # if any command fail - this script exits with non zero status

START=$(date +%s.%N)

################################################################################

CONFIGDIR=`pwd`

PYTHON_HTK_PATH=$CONFIGDIR/../.. #path for python where are the reading and writing functions for htk feature file

#PosteriorFile=../../example/example.post.fea #in HTK format
#OutputLattice=out

PosteriorFile=$1              # input feature files with phoneme posteriors
OutputFileNoExtension=$2      # output file name without extension - latt.gz will be added for lattices and count.gz for counts

LatticeCountORDER=3

HLIST=/usr/local/share/HTK/HList
HVITE=/usr/local/share/HTK/HVite
SRI_LATTICE_TOOL=/usr/local/share/Srilm/bin/lattice-tool 


# Absolute path to this script.
SCRIPT=$(readlink -f $0)
# Absolute path this script is in.
SCRIPTPATH=`dirname $SCRIPT`

###################################################################
#Functions

function WARNMSG {
  echo ""
  echo "WARNINING: $1"
  echo ""
}
function LOGMSG {
  echo ""
  echo "LOG: $1"
  echo ""
}
function ERRMSG {
  echo ""
  if [ $2 -gt 0 ];then linecount="line $2"; else linecount=""; fi
  echo "ERROR: $linecount in $SCRIPT"
  echo "ERROR: $1"
  echo ""
  exit 1
}

function EXIT_FUNCTION {
   if [ -e "$TMP_DIR" ];then
     echo "Deleting TMP_DIR: $TMP_DIR"
     rm -rf "$TMP_DIR"
   fi

   END=$(date +%s.%N)
   DIFF=$(echo "$END - $START" | bc)
   echo "LOG: Program finished in $DIFF sec."
   echo "LOG: Program stopped at " `date`
   echo "LOG: End of program $SCRIPT"
   echo ""
}
trap  EXIT_FUNCTION EXIT 

echo "Pragram started on " `hostname` " at " `date` 

###################################################################

#check if input exist
if [ ! -e "$PosteriorFile" ];then
  ERRMSG "Input posterior file does not exist: $PosteriorFile" $LINENO
fi

#check if output already exist
if [ -z "$OutputFileNoExtension" ]; then
  ERRMSG "Output file argument is unset or set to the empty string." $LINENO
fi
if [ -e "$OutputFileNoExtension.latt.gz" ];then
  size=`ls -all $OutputFileNoExtension.latt.gz |awk '{print $5}'`
  if [ $size -gt 0 ];then
    LOGMSG "Output file already exists: $OutputFileNoExtension.latt.gz"
    exit 0
  else
    WARNMSG "Output file already exists but has zero size - recomputing";
  fi
fi

###################################################################
# create tmp dir
mkdir -p /tmp/PHNREC_${USER}_$$
TMP_DIR=`TMPDIR=/tmp/PHNREC_${USER}_$$ mktemp -d -u`
rm -rf $TMP_DIR
mkdir -p $TMP_DIR/
###################################################################

FileName=`echo $PosteriorFile | awk -F "/" '{print $NF}' |  sed 's/\.[^\.]*$//'`

num=`$HLIST $PosteriorFile |grep nan |wc -l`
if [ $num -ne 0 ];then
  ERRMSG "Nan in the file: $PosteriorFile" $LINENO
fi

#FRMNUM=`$printHTKheader $PosteriorFile  |awk '{print $2}'`
#if [ $FRMNUM -le 0 ];then
#    ERRMSG "Feature file con not have 0 frames: $FRMNUM, $PosteriorFile" $LINENO
#fi

#convert Posterior file to a file which would fit under the HVite HMM model and generate correct posteriors inside HVite phoneme decoder.

echo "
import sys
sys.path.append('$PYTHON_HTK_PATH');
import utils
import numpy as np
inp=\"$PosteriorFile\"
out=\"$TMP_DIR/$FileName.fea\"
m = utils.read_htk(inp);
d1=np.sqrt(-2*np.log(m.clip(min=1e-10)))
utils.write_htk(out,d1)
sys.stdout.flush();
sys.stderr.flush();
" >  $TMP_DIR/convert.py

MKL_NUM_THREADS=1 python $TMP_DIR/convert.py

if [ ! -e $TMP_DIR/$FileName.fea ];then
  ERRMSG "File was not generated : $TMP_DIR/$FileName.fea" $LINENO
fi


#echo "Best string decoding .... "
#$HVITE \
#-T 1 -y 'rec' -l '*' \
#-C ${CONFIGDIR}/HVite.cfg   \
#-w ${CONFIGDIR}/monophones_lnet.hvite \
#-p -1 \
#-i $TMP_DIR/$FileName.mlf \
#-H ${CONFIGDIR}/hmmdefs.hvite \
#${CONFIGDIR}/dict \
#${CONFIGDIR}/phonemes \
#$TMP_DIR/$FileName.fea


#lattice decoding - word insertion penalty has to be tuned for your application

echo "Lattice decoding .... "
mkdir -p $TMP_DIR/lattice
$HVITE \
-T 1 -y 'lab' -z 'latt'   \
-C ${CONFIGDIR}/HVite.cfg   \
-w ${CONFIGDIR}/monophones_lnet.hvite \
-n 2 1  \
-p -1 \
-l $TMP_DIR   \
-H ${CONFIGDIR}/hmmdefs.hvite \
${CONFIGDIR}/dict \
${CONFIGDIR}/phonemes \
$TMP_DIR/$FileName.fea

if [ ! -e $TMP_DIR/$FileName.latt ];then
  ERRMSG "File was not generated : $TMP_DIR/$FileName.latt" $LINENO
fi

echo "Count generation .... "
$SRI_LATTICE_TOOL -in-lattice $TMP_DIR/$FileName.latt  -order $LatticeCountORDER -compute-posteriors -compact-expansion -read-htk -write-htk -write-ngrams $TMP_DIR/$FileName.count

if [ ! -e $TMP_DIR/$FileName.count ];then
  ERRMSG "File was not generated : $TMP_DIR/$FileName.count" $LINENO
fi


echo "Copy to destination .... "
OUTPATH=`dirname $OutputFileNoExtension`
mkdir -p $OUTPATH
gzip -c $TMP_DIR/$FileName.count > $OutputFileNoExtension.count.gz.part
mv $OutputFileNoExtension.count.gz.part $OutputFileNoExtension.count.gz
gzip -c $TMP_DIR/$FileName.latt > $OutputFileNoExtension.latt.gz.part
mv $OutputFileNoExtension.latt.gz.part $OutputFileNoExtension.latt.gz





