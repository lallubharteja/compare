#!/bin/bash

function header {
  feat_name=$1
  feats=$2
  #print the header
  printf "name"
  for i in `seq $feats`; do
    printf ";"
    printf "${feat_name}_$i"
  done
  echo
}

low_mel=10
high_mel=10
wav=wav
bs_fc=400

for n_mels in 200; do
  for win_length in 800; do
    for preemp in -1 ; do
       for lfn in 1 ; do
      num_feats=$(python3 recipes/common/feature_generator.py --preemp $preemp --audio-path data/$wav/train_00001.wav --n-mels $n_mels --win-length $win_length --low-freq-norm $lfn | tr ';' '\n' | wc -l)
      ((--num_feats))
      header "mel" $num_feats > generated/features/ComParE_Mask.mel$n_mels.wl$win_length.preemp$preemp.f$bs_fc.train.csv
      for f in $(ls data/$wav/train*wav); do
        python3 recipes/common/feature_generator.py --preemp $preemp --audio-path $f --n-mels $n_mels --win-length $win_length --filter $bs_fc >> generated/features/ComParE_Mask.mel$n_mels.wl$win_length.preemp$preemp.f$bs_fc.train.csv
        done
      done
    done
  done
done

for n_mels in 200; do #10 20 40 50 100 200; do
  for win_length in 800; do
    for preemp in -1 ; do
       for lfn in 1 ; do
      num_feats=$(python3 recipes/common/feature_generator.py --preemp $preemp --audio-path data/$wav/devel_00001.wav --n-mels $n_mels --win-length $win_length --low-freq-norm $lfn | tr ';' '\n' | wc -l)
      ((--num_feats))
      header "mel" $num_feats > generated/features/ComParE_Mask.mel$n_mels.wl$win_length.preemp$preemp.f$bs_fc.devel.csv
      for f in $(ls data/$wav/devel*wav); do
        python3 recipes/common/feature_generator.py --preemp $preemp --audio-path $f --n-mels $n_mels --win-length $win_length --filter $bs_fc >> generated/features/ComParE_Mask.mel$n_mels.wl$win_length.preemp$preemp.f$bs_fc.devel.csv
      done
      done
#      fi
    done
  done
done

for n_mels in 200; do #10 20 40 50 100 200; do
  for win_length in 800; do
    for preemp in -1 ; do
       for lfn in 1 ; do
      num_feats=$(python3 recipes/common/feature_generator.py --preemp $preemp --audio-path data/$wav/test_00001.wav --n-mels $n_mels --win-length $win_length --low-freq-norm $lfn | tr ';' '\n' | wc -l)
      ((--num_feats))
      header "mel" $num_feats > generated/features/ComParE_Mask.mel$n_mels.wl$win_length.preemp$preemp.f$bs_fc.test.csv
      for f in $(ls data/$wav/test*wav); do
        python3 recipes/common/feature_generator.py --preemp $preemp --audio-path $f --n-mels $n_mels --win-length $win_length --filter $bs_fc >> generated/features/ComParE_Mask.mel$n_mels.wl$win_length.preemp$preemp.f$bs_fc.test.csv
      done
      done
    done
  done
done

