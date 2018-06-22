# Vocal Melody Extraction


This repository includes the source code of the melody extraction algorithm from:

Wei-Tsung Lu and Li Su, “Vocal melody extraction with semantic segmentation and audio-symbolic domain transfer learning,” International Society of Music Information Retrieval Conference (ISMIR), September 2018.

### Dependencies

This repository requires following packages:

- python 3.6
- numpy
- tensorflow
- keras
- mido

### Usage

```
usage: VocalMelodyExtraction.py [-h] [-m model_name] [-i input_file]
                                            
required arguments:

optional arguments:
  -h
  -m  model_name     path to existing model (default = "transfer_audio_directly" )
  -i  input_file     path to input file (default = "train01.wav" )
```

### Todos

 - Add codes for training phase

License
----

BSD




