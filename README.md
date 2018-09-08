# Vocal Melody Extraction


This repository includes the source code of the melody extraction algorithm from:

Wei-Tsung Lu and Li Su, “Vocal melody extraction with semantic segmentation and audio-symbolic domain transfer learning,” International Society of Music Information Retrieval Conference (ISMIR), September 2018.

Wei-Tsung Lu and Li Su, "Deep Learning Models for Melody Perception: An Investigation on Symbolic Music Data," Proc. Asia Pacific Signal and Infor. Proc. Asso. Annual Summit and Conf. (APSIPA ASC), November 2018.

### Dependencies

This repository requires following packages:

- python 3.6
- numpy
- tensorflow
- keras
- mido

### Usage

```
usage: VocalMelodyExtraction.py [-h][-p phase]
                                [-t model_type][-d data_type][-da dataset_path][-la label_path]
                                [-ms model_path_symbolic][-w window_width][-b batch_size_train][-e epoch]
                                [-n steps][-o output_model_name]
                                [-m model_path] [-i input_file][-bb batch_size_train]
  required arguments:
  -da dataset_path              path to data set 
  -la label_path                path to dataset label
  -ms model_path_symbolic       path to symbolic model 
  
  optional arguments:
  -h                
  -p  phase                     phase: training or testing (default: "testing) 
  -t  model_type                model type: seg or pnn (default: "seg")
  -d  data_type                 data type: audio or symbolic (default: "audio") 
  -w  window_width              width of the input feature (default: 128)
  -b  batch_size_train          batch size during training (default: 12)
  -e  epoch                     number of epoch (default: 5)
  -n  steps                     number of step per epoch (default: 6000)
  -o  output_model_name         name of the output model (default: "out")
  -m  model_path                path to existing model (default: "Seg")
  -i  input_file                path to input file (default: "train01.wav")
  -bb batch_size_train          batch size during testing (default: 10)
```

### Pretrained Models

Click [here] to download the pretrained models.

### Todos

 - Add codes for symbolic model training
 - Data set handling

License
----

MIT

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [here]:https://drive.google.com/open?id=13kApyZ5lJEGE5CDwaeEuxVuw9sZy_xae

