#!/bin/bash -l

mkdir -p data generated experiments 
mkdir -p generated/features experiments/seq2seq2d

if [ -d /teamwork/t40511_asr/p/ComParE/ComParE_2020_Mask/wav ]; then 
  cp -r /teamwork/t40511_asr/p/ComParE/ComParE_2020_Mask/wav data/
  cp -r /teamwork/t40511_asr/p/ComParE/ComParE_2020_Mask/lab data/
else
  echo "ERROR: you don't have access to the above directory. You will need to copy the wav folder manually." 
  exit 1;
fi
