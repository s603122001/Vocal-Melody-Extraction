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
  -m  model_name     path to existing model (default = "Seg" )
  -i  input_file     path to input file (default = "train01.wav" )
```

### Pretrained Models

Click [here] to download 3 pretrained models.

### Todos

 - Add codes for training phase

License
----

MIT

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [here]:https://drive.google.com/open?id=13kApyZ5lJEGE5CDwaeEuxVuw9sZy_xae

