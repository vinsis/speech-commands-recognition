## Timeline of network design

As my first approach, I did not want to:
1) Manually extract features (fourier transform, mel spectogram et al) from the audio clips
2) Use max-pooling
I decided to use fully convolutional neural network for word identification.

### 1. First working model

My [first model](https://github.com/vinsis/speech-commands-recognition/blob/b92911b51f021bf7258545a0f46ad2aee713896d/model.py) refused to converge. The loss would just linger where it started.

### 2. Working model that converged

I added a batch normalization. But the error still refused to go down. Increasing the learning rate to 0.01 did the magic. [Here](https://github.com/vinsis/speech-commands-recognition/blob/d0497bf66f701bc0f3633f55bc54af2b77b724b5/model.py) is what the model looked like at this point.

### 3. Making the model easy to interpret

In order to make the model easy to interpret and easy to intuit, I decided to apply global average pooling followed by a dense layer. In this case, the pooled layers would act as weighted features which would be fed to the final dense layer to get the output. (I was thinking those features to be somehow representative of phonemes making up a word.)

[Model at this point](https://github.com/vinsis/speech-commands-recognition/blob/8e4f8a0e26932a89bf8b43149a8e0516b22c431d/model.py)

During the training, error would plateau at around 0.4. At this point, after 5 epochs, I was able to get an accuracy of ~80% on validation set. Not great but promising!

### 4. Increasing the number of parameters

Since the error would plateau during training (in spite of tweaking the learning rate and applying a learning rate scheduler), I decided to increase the number of parameters. I increased the number of channels in each convolution operation.

[Model with increased number of channels](https://github.com/vinsis/speech-commands-recognition/blob/d2c238933c514887dc01baf93c6c53d2528c7669/model.py)

This led to a faster convergence (~85% after 3 epochs). However, the error would still plateau at around 0.35 - 0.4.

It was easy to get to this point but after this point, making further improvements got challenging. I was running my model on my Mac without a GPU and it was getting slower and slower.

### 5. Increasing the number of parameters even more

To make sure my model wasn't constrained by the number of parameters, I decided to go ballistic and significantly increased the number of parameters. I was able to achieve accuracies of 88.5% and 87.5% on validation and test sets within six epochs.

I trained the model further with learning rate 1e-6 but it didn't improve. I guess I am done experimenting.

## Improving accuracy further

Here are some of the things one can try. Note that these steps are written keeping in mind that I am still trying to refrain from MaxPooling and  pre-processing audio (using fourier transform etc):

* Changing the model parameters (kernel size, stride, number of dense layers etc)
* Tweaking hyperparameters like momentum, L2 weight decay
* Adding noise to training data to make the learning model more robust