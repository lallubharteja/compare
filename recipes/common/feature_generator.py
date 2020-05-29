# Copyright 2019 RnD at Spoon Radio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import librosa
import os, sys
import numpy as np
from scipy import signal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature generator')
    parser.add_argument('--audio-path', default='../data/61-70968-0002.wav',
                        help='The audio file.')
    parser.add_argument('--feature', type=str, default='mel',
                        help='Select feature: [mel]')
    parser.add_argument('--n-mels', type=int, default=40,
                        help='number of mel filters to be used for mel feature')
    parser.add_argument('--low-mels-only', type=int, default=0,
                        help='extract lower mel features after mel production')
    parser.add_argument('--low-freq-ratio', type=int, default=0,
                        help='extract lower mel features after mel production')
    parser.add_argument('--low-freq-norm', type=int, default=0,
                        help='extract lower mel features after mel production')
    parser.add_argument('--high-mels-only', type=int, default=0,
                        help='extract higher mel features after mel production')
    parser.add_argument('--win-length', type=int, default=160,
                        help='window length milliseconds*fs')
    parser.add_argument('--masking-line-number', default=1,
                        help='masking line number')
    parser.add_argument('--preemp', type=float, default=0.0,
                        help="apply preemphasis using librosa")    
    parser.add_argument('--filter', type=float, default=0.0,
                        help="apply butterworth filtering to audio")

    args = parser.parse_args()
    audio_path = args.audio_path

    # load audio file, extract mel spectrogram
    audio, sampling_rate = librosa.load(audio_path)

    # apply preemphasis if requested
    if args.preemp != 0:
        audio = librosa.effects.preemphasis(audio, coef=args.preemp ) # coef = 0.97, 0.99
    
    if args.filter > 0 :
        w = args.filter / (sampling_rate / 2) # Normalize the frequency
        b, a = signal.butter(5, w, 'low')
        audio = np.asfortranarray(signal.filtfilt(b, a, audio))
   
    # apply feature
    if args.feature == 'mel':
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sampling_rate,
                                                         n_mels=args.n_mels,        # try 10 20 40 50 100 200
                                                         hop_length=128,
                                                         win_length=args.win_length,   # 160 480 800
                                                         fmax=8000)

        if args.low_freq_ratio > 0:
            lowf = np.sum(mel_spectrogram[:args.low_freq_ratio,:], axis=0)
            highf = np.sum(mel_spectrogram[args.low_freq_ratio:,:], axis=0)
            mel_spectrogram = lowf / highf
        if args.low_freq_norm > 0:
            lowf = np.sum(mel_spectrogram[:args.low_freq_norm,:], axis=0)
            normf = np.sum(mel_spectrogram, axis=0)
            mel_spectrogram = lowf / normf
        # flatten spectrogram shape to [time*frequency]
        if args.low_mels_only > 0 and args.high_mels_only > 0:
            mel_spectrogram = np.delete(mel_spectrogram,np.arange(args.low_mels_only,mel_spectrogram.shape[0]-args.high_mels_only),0)
        elif args.low_mels_only > 0:
            mel_spectrogram = mel_spectrogram[:args.low_mels_only,:]
        elif args.high_mels_only > 0:
            mel_spectrogram = mel_spectrogram[mel_spectrogram.shape[0]-args.high_mels_only:,:]
        orig_ms = mel_spectrogram.flatten()

        basename = os.path.basename(audio_path)
        print("'",basename,"';",";".join(list(map(str,orig_ms))),sep="")
