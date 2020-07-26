# MelGAN
Train melgan on `VCTK`

Put wav data in `wavs` dir
```
wavs
    p255
        p255-*.wav
    p256
        p256-*.wav
    p257
        p257-*.wav
```
### requirement
1. librosa
2. pytorch
3. numpy
4. pathlib
### train
```python
python train.py
```
### How to generate wave?
It's training now

### Mel Spec
1. fftlength = 1024
2. hoplength = 256
3. samplerate = 16Khz
4. Mel_dim = 80
