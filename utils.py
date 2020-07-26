from pathlib import Path
from librosa.core import load


################################################
#          get wavs list in wavs dir           #
################################################
def getfilepath(dir_path='./wavs'):
    p = Path(dir_path)
    filelist = []
    print(p)
    for pth in p.iterdir():
        for fpth in pth.iterdir():
            filelist.append(fpth)
    return filelist


################################################
#               load wav files                 #
################################################

def getwavfile(path, sr=160000):
    data, samplerate = load(path, sr)
    print(data.shape)
    print(type(data))
    print(samplerate)
    return data, samplerate