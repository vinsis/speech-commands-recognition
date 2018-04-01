## About
This is an attempt to learn features from raw audio directly without any feature extraction. I also tried not to use MaxPooling and relied solely on Global Average Pooling followed by a fully connected layer. After a few quick attempts I was able to achieve ~88% accuracy. Top scorers on Kaggle scored ~90% accuracy (although I suspect the test set used was different). The model I used is dead simple:

```
ModelCNN(
  (main): Sequential(
    (0): Conv1d(1, 32, kernel_size=(90,), stride=(6,))
    (1): ReLU(inplace)
    (2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True)
    (3): Conv1d(32, 64, kernel_size=(31,), stride=(6,))
    (4): ReLU(inplace)
    (5): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True)
    (6): Conv1d(64, 128, kernel_size=(11,), stride=(3,))
    (7): ReLU(inplace)
    (8): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True)
    (9): Conv1d(128, 256, kernel_size=(7,), stride=(2,))
    (10): ReLU(inplace)
    (11): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True)
    (12): Conv1d(256, 512, kernel_size=(5,), stride=(2,))
    (13): ReLU(inplace)
    (14): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True)
    (15): AvgPool1d(kernel_size=(47,), stride=(47,), padding=(0,), ceil_mode=False, count_include_pad=True)
  )
  (fc): Linear(in_features=512, out_features=30, bias=True)
)
```


The combination of global average pooling followed by fully connected layer can be used to detect where in the entire file the word is being spoken.

## Speech Commands Recognition
Single word speech recognition using PyTorch

### Data source
Warden P. Speech Commands: A public dataset for single-word speech recognition, 2017. Available from http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz

__Note__: Since the unzipped data is > 2GB in size, I have only uploaded a small sample of the entire dataset. In each folder, there are only ten files (instead of 1000+). 

### Speech Commands Data Set v0.01
> This is a set of one-second .wav audio files, each containing a single spoken
English word. These words are from a small set of commands, and are spoken by a
variety of different speakers. The audio files are organized into folders based
on the word they contain, and this data set is designed to help train simple
machine learning models.
