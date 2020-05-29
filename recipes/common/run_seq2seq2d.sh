#!/bin/bash

mkdir -p experiments/seq2seq2d
#conda activate seq2seq
subid=2
f_bs=400
for n_mels in 200; do
  for win_length in 800 ; do
    for keep in -1 ; do
      for preemp in -1 ; do 
        for subid in {1..10}; do
      #preemp=-1
#      if [ ! -f experiments/seq2seq2d/librosa_mfcc.keep$keep.subid$subid.out ]; then
#        CUDA_VISIBLE_DEVICES="" python3 src/neural_net/seq2seq2d.py --feature librosa_mfcc --taskname ComParE2020_Mask --keep $keep --predict --subid $subid > experiments/seq2seq2d/librosa_mfcc.keep$keep.subid$subid.out
#      fi
#      if [ ! -f experiments/seq2seq2d/librosa_mfcc.stand.keep$keep.subid$subid.out ]; then
#        CUDA_VISIBLE_DEVICES="" python3 src/neural_net/seq2seq2d.py --feature librosa_mfcc --taskname ComParE2020_Mask --keep $keep --predict --subid $subid --standardize > experiments/seq2seq2d/librosa_mfcc.stand.keep$keep.subid$subid.out
#      fi
#      if [ ! -f experiments/seq2seq2d/zff.mel$n_mels.wl$win_length.preemp$preemp.keep$keep.subid$subid.out ]; then
#        CUDA_VISIBLE_DEVICES="" python3 src/neural_net/seq2seq2d.py --feature mel$n_mels.wl$win_length.preemp$preemp.lmel20 --taskname ComParE_zff --keep $keep --predict --subid $subid > experiments/seq2seq2d/zff.mel$n_mels.wl$win_length.preemp$preemp.keep$keep.subid$subid.out
#      fi
      if [ ! -f experiments/seq2seq2d/mel$n_mels.wl$win_length.preemp$preemp.f$f_bs.keep$keep.subid$subid.out ]; then
        CUDA_VISIBLE_DEVICES="" python3 src/neural_net/seq2seq2d.py --feature mel$n_mels.wl$win_length.preemp$preemp.f$f_bs --taskname ComParE --keep $keep --devpredict --subid $subid > experiments/seq2seq2d/mel$n_mels.wl$win_length.preemp$preemp.f$f_bs.keep$keep.subid$subid.out
      fi

#      if [ ! -f experiments/seq2seq2d/mel$n_mels.wl$win_length.preemp$preemp.hmel20.lmel10.out ]; then
#        CUDA_VISIBLE_DEVICES="" python3 src/neural_net/seq2seq2d.py --feature mel$n_mels.wl$win_length.preemp$preemp.hmel20.lmel10 --taskname ComParE > experiments/seq2seq2d/mel$n_mels.wl$win_length.preemp$preemp.hmel20.lmel10.out
#      fi
        done
      done
    done
  done
done


