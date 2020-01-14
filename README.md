# Mnist Classificatoin

Scratching the surface of neural network / deep learning

## installation

just pull, everything is in math approach, no 3rd party package is used

```
git pull    https://github.com/Yimaha/Mnist-digit-classification-python.git
```

## introduction

This is my second attempt to classify Mnist data. The first Attempt is in my 
[javascript version](https://github.com/Yimaha/MNIST-digit-recognistion-Neural-Network) 

The intention of this repository is to learn different method to train a neural network, such as 
utilizing different initialization function, implementing regulation, and deal with
data over-fitting.

Well, as stupid as I was, My first attempt was in javascript, which is a horrible language for data science. On top of that,
**not even a basic linear algebra library was used in my first iteration.** As you may have expected, I encountered
a lot of issue in terms of implementation, but it really helped me understand the math behind the network(and force me 
to review linear algebra). 

Later on, I decided to use [Tensorflow.js](https://www.tensorflow.org/js) as my linear algebra library. which was a 
great idea. My network finally started working for the first time in it's life which reached a 
terminal accuracy of **95%**.

But, as I learn more about the subject, I notice that [Tensorflow.js](https://www.tensorflow.org/js) is just not enough. The speed of training is 
extremely slow, on top of that I frequently encounter memory management issues since I am no master in Tensorflow yet.

Therefore, to fix that issue, I switched to **python**, which is the same language used by the online resource 
I am learning my concepts from. It's performance, math libs, and 3rd party modules (pytorch, tensorflow, pandas) has been a great addition
to my NN tool box.

Now I am trying to implement that last bit of the textbook, which is **deep learning**. It should increase my accuracy to above **99.5%**

Oh btw here is the book: [Neural Networks and Deep Learning by Micheal Nielsen](http://neuralnetworksanddeeplearning.com/index.html)

Shout out to [Micheal Nielsen](http://michaelnielsen.org/) and [3b1b](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw), who inspired me to start learning neural network and deep learning

## terminal accuracy
This is my highest accuracy at the moment: 

***98.3%***

## Folder Structure

**data** folder contains all of the mnist train, validation, and test data.

**solutions** folder contains all the solutions to different types of network. **most of them are playground file other than network, network2, and network3(incoming)**.

**utils** folder contains all the utilities used, such as calculation utilities, or Mnist file reading utilities.


