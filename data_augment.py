# -*- coding: utf-8 -*-
"""
This is a fast and dirty implem of WavAugment, the waveform equivalent of SpecAugment
Written using pysox.

The frequency and time masking are implemented with filters and amplitude, respectively.

The time warping is not exactly similar to that of SpecAugmentsince it only stretches the time axis uniformly;
it can therefore stretch is quite a bit more without causing too much distortion in comprehension;
it is therefore supplemented with a pitch warping which independantly warps the spectrum.


"""

import os
import torch
import sox
import tempfile
import numpy as np
import soundfile as sf
import torchaudio
import argparse
import logging
sox.logger.setLevel(logging.ERROR)

DEBUG=True
SOXVERBOSITY=0

# transforming Hz into mels
def freq2mel(f):
    return 2595.*np.log10(1+f/700)

# and back
def mel2freq(m):
    return ((10.**(m/2595.)-1)*700)

# the SpecAugment F parameter (nb of mel channels removed) assumes a 256 mel spectrogram. We emulate this here as a fraction over the maximum frequency (normalized by 256).
def frequency_masking(filein,fileout,F):
    samplerate=sox.file_info.sample_rate(filein)  # get file sampling rate
    melfmax=freq2mel(samplerate/2)                # the max frequency range expressed in mel scale
    meldf=np.random.uniform(0,melfmax*F/256.)     # select a width of frequency (in mel space)
    melf0=np.random.uniform(0,melfmax-meldf)      # select the lower frequency bound
    tfm = sox.Transformer()
    #tfm.set_globals(verbosity=SOXVERBOSITY)
    low=mel2freq(melf0)                           # back into frequency space
    high=mel2freq(melf0+meldf)
    tfm.sinc(filter_type='reject',cutoff_freq=[high,low]) # for some reason high should be before low
    tfm.build(filein,fileout)                     # do the transform
    if DEBUG: print("Filtering between","%.2f" % low,"and","%.2f" % high, "Hz")

# In specaugment, T is in frames (which are usually 10ms); we assume 10 ms here.
def time_masking(filein,fileout,T,p=1):
    file_duration=sox.file_info.duration(filein)
    T=T/100. # here we convert into seconds
    T=min(T,p*file_duration)
    deltat=np.random.uniform(0,T)
    t0=np.random.uniform(0,file_duration-deltat)
    temp1=tempfile.NamedTemporaryFile('w+b',suffix=".wav") # temporary files for the initial part
    temp2=tempfile.NamedTemporaryFile('w+b',suffix=".wav") # temporary files for the initial part
    temp3=tempfile.NamedTemporaryFile('w+b',suffix=".wav") # temporary files for the second part
    tfm1 = sox.Transformer()
    tfm2 = sox.Transformer()
    tfm3 = sox.Transformer()
    #tfm1.set_globals(verbosity=SOXVERBOSITY)
    #tfm2.set_globals(verbosity=SOXVERBOSITY)
    #tfm3.set_globals(verbosity=SOXVERBOSITY)
    tfm1.trim(0,t0)
    tfm2.trim(t0,t0+deltat)
    tfm2.gain(-90)
    tfm3.trim(t0+deltat)
    tfm1.build(filein,temp1.name) # buildind the file
    tfm2.build(filein,temp2.name) # building the file
    tfm3.build(filein,temp3.name) # building the file

    cbn = sox.combine.Combiner()
    #cbn.set_globals(verbosity=SOXVERBOSITY)
    cbn.build([temp1.name,temp2.name,temp3.name],fileout,'concatenate')        # do the transform
    if DEBUG: print("Masking between","%.3f" % t0,"to","%.3f" % (t0+deltat),"sec")

# Wmax is the maximum time warping (>1; eg 1.2 for 20%); Pmax is the max pitch warping (constant in semitones)
def time_warping(filein,fileout,Wmax,Pmax):
     sign=np.sign(np.random.uniform(0,1)-0.5)
     w1=np.exp(sign*np.random.uniform(0.05,np.log(Wmax))) # large warp factor (uniform in log space)
     w2=np.exp(-sign*np.random.uniform(0.05,np.log(Wmax))) # small warp factor (uniform in log space)
     pitch=np.random.uniform(-Pmax,Pmax)       # pitch warp factor
     cut=(1-w2)/(w1-w2)  # where to cut the initial file such that the total duration is the same
     dur=sox.file_info.duration(filein)  # file duration

     temp1=tempfile.NamedTemporaryFile('w+b',suffix=".wav") # temporary files for the initial part
     temp2=tempfile.NamedTemporaryFile('w+b',suffix=".wav") # temporary files for the second part
     tfm1 = sox.Transformer()
     tfm2 = sox.Transformer()
     #tfm1.set_globals(verbosity=SOXVERBOSITY)
     #tfm2.set_globals(verbosity=SOXVERBOSITY)
     tfm1.pitch(pitch)
     tfm1.trim(0,cut*dur)
     tfm1.tempo(w1,audio_type="s")  # cutting and changing tempo & pitch of first part

     tfm2.pitch(pitch)
     tfm2.trim(cut*dur)
     tfm2.tempo(w2,audio_type="s")  # cutting and changing tempor of second part
     tfm1.build(filein,temp1.name) # buildind the file
     tfm2.build(filein,temp2.name) # building the file
     # horrible hack, because the @&*$ pysox forgot to implement the splice operation
     #os.system("sox "+temp1.name+" "+temp2.name+" "+fileout+" splice")
     cbn = sox.combine.Combiner()
     #cbn.set_globals(verbosity=SOXVERBOSITY)
     cbn.build([temp1.name,temp2.name],fileout,'concatenate')        # do the transform
     if DEBUG: print("Warping by","%.2f" % w1,"till","%.3f" % (cut*dur),"sec then by","%.2f" % w2,"; also, deltapitch=","%.2f" % pitch)
 #    print(w1,w2,pitch,cut,dur,cut*dur,w1*cut*dur+w2*(1-cut)*dur)


def pure_pitch_warping(filein,fileout,Pmax):
     pitch=np.random.uniform(-Pmax,Pmax)       # pitch warp factor
     dur=sox.file_info.duration(filein)  # file duration

     tfm1 = sox.Transformer()
     #tfm1.set_globals(verbosity=SOXVERBOSITY)
     tfm1.pitch(pitch)
     tfm1.build(filein,fileout)
     if DEBUG: print("Deltapitch=","%.2f" % pitch)
 #    print(w1,w2,pitch,cut,dur,cut*dur,w1*cut*dur+w2*(1-cut)*dur)



# off-line version of the SpecAugment (100 for time masking seems a lot; reduced to 80 by default)
def WavAugmentOffline(filenamein,filenameout,
                      time_warping_para=1.8,
                      pitch_warping_para=2,
                      frequency_masking_para=27,
                      time_masking_para=80,
                      frequency_mask_num=1,
                      time_mask_num=1):
    temp1=tempfile.NamedTemporaryFile('w+b',suffix=".wav") # temporary files for the initial part

    infile_duration=sox.file_info.duration(filenamein)
    if(time_warping_para>1):
        time_warping(filenamein,temp1.name,time_warping_para,pitch_warping_para)
        filenamein=temp1.name
    else:
        pure_pitch_warping(filenamein,temp1.name,pitch_warping_para)
        filenamein=temp1.name

    if frequency_mask_num>=1:
            for i in range(frequency_mask_num):
                temp2=tempfile.NamedTemporaryFile('w+b',suffix=".wav") # temporary files for the initial part
                frequency_masking(filenamein,temp2.name,frequency_masking_para)
                filenamein=temp2.name

    if time_mask_num>=1:
            for i in range(time_mask_num):
                temp3=tempfile.NamedTemporaryFile('w+b',suffix=".wav") # temporary files for the initial part
                time_masking(filenamein,temp3.name,time_masking_para)
                filenamein=temp3.name

    tfm = sox.Transformer() # copying the result in filenameout
    #tfm.set_globals(verbosity=SOXVERBOSITY)
    tfm.build(filenamein,filenameout)
    outfile_duration=sox.file_info.duration(filenameout)
    print("Duration: ",infile_duration,"->",outfile_duration)



# mimics the SpecAugment function that can work on-line
# this may be slow because it calls the offline function with temporary files
def WavAugment(data, samplerate, **params):

    tempinfile = tempfile.NamedTemporaryFile('w+b', suffix=".wav")
    sf.write(tempinfile, data, samplerate, subtype='PCM_16')
    tempoutfile = tempfile.NamedTemporaryFile('w+b', suffix=".wav")
    WavAugmentOffline(tempinfile.name, tempoutfile.name, **params)
    modifieddata, samplerate = sf.read(tempoutfile)
    tempinfile.close()
    tempoutfile.close()
    return(modifieddata, samplerate)


def LoadWaveAugmented(pathIn, fullPathOut = None):

    seq, samepleRate = torchaudio.load(pathIn)
    seq = seq.view(-1).numpy()
    tempinfile = tempfile.NamedTemporaryFile('w+b', suffix=".wav")
    if fullPathOut is None:
        tempoutfile = tempfile.NamedTemporaryFile('w+b', suffix=".wav")
    sf.write(tempinfile, seq, samepleRate, subtype='PCM_16')
    name = fullPathOut if fullPathOut is not None else tempoutfile.name
    WavAugmentOffline(tempinfile.name, name,time_warping_para=1,time_masking_para=10)
    tempinfile.close()
    if fullPathOut is None:
        modifieddata, samplerate = sf.read(tempoutfile)
        tempoutfile.close()
        return torch.tensor(modifieddata).float()


if __name__ == "__main__":

    from dataset import findAllSeqs
    parser = argparse.ArgumentParser(description='Dataset augment')
    parser.add_argument('pathDB', type=str)
    parser.add_argument('pathOut', type=str)
    parser.add_argument('--recursionLevel', type=int, default=2)
    parser.add_argument('--extension', type=str, default='.flac')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    inSeqs = [x[1] for x in
               findAllSeqs(args.pathDB, extension=args.extension,
                           recursionLevel=args.recursionLevel)[0]]

    if args.debug:
        inSeqs = inSeqs[:10]

    if not os.path.isdir(args.pathOut):
        os.mkdir(args.pathOut)

    for index, seqName in enumerate(inSeqs):

        fullPath = os.path.join(args.pathDB, seqName)
        fullPathOut = os.path.join(args.pathOut, seqName)

        dirOut = os.path.dirname(fullPathOut)
        if not os.path.isdir(dirOut):
            os.makedirs(dirOut, exist_ok=True)

        LoadWaveAugmented(fullPath, fullPathOut)
