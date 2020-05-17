# news-headlines-sarcasm-detector
This project is a solution to a Kaggle problem from 2018 (https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection). The data is a collection of news headlines, some from the "Huffington Post" and the rest from satirical newspaper "The Onion". The goal of the Kaggle problem is to build a classifier that predicts if a given headline is sincere (IE, more similar to The Huffington Post) or if a headline is satire (IE, more like The Onion).

I found a solution that takes advantage of preexisting word embeddings: BERT! (https://arxiv.org/abs/1810.04805) \
Bert is a language representation model that came out in 2018. Upon it's release, BERT was topping the leader boards on a number of NLP tasks and can be used for a variety of purposes. One of its more popular uses is for sentence classification! I treated each headline like it was a sentence and then applied a pretrained BERT model to produce word embeddings. Once I had word embeddings for each headline, I ran the BERT embeddings through a recurrent neural network (more specifically a RNN with a Gated Recurrent Unit aka GRU) and finally ran the final RNN hidden state through a linear layer to predict either sarcasm or sincerity.

### Dependencies
All of this is done in Python with PyTorch, Huggingface's Transformers package, and TorchText.\
1) PyTorch installation is hardware specific, check their website to get the best instructions: https://pytorch.org/ 
2) Install Transformers via pip: `pip install transformers`
3) Install TorchText via pip: `pip install torchtext`

 
### Appendix
Some resources I used which helped me a lot on this project:
1) Ben Trevett's Github page, contains a lot of tutorials and examples in PyTorch: https://github.com/bentrevett \
 In particular, his sentiment analysis repository where he applied BERT was very helpful. 
2) ML Explained TorchText tutorial. Supplemental  to Ben Trevett's tutorials, this went a little more in depth on what the TorchText classes do: http://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/
