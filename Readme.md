# Super Reviews using RNN and SHAP

Before the advent of E-commerce and the Internet, when purchasing a product people, when not relying on their judgment, would often gather others opinion by word of mouth, from their circle or leveraging the knowledge of the shopkeeper. Nowadays, when deciding which smartphone or which digital camera we should buy, we can get information and people rating everywhere, from reviews, blogs, online forums, social media. We went from having scarce information to having too much data on a product. In fact many products on popular E-commerce platforms have hundreds or thousands customer reviews for some products. So many that for a customer becomes cumbersome and confusing to read them all.

Multi-document text summarization can come to the rescue to generate summaries of multiple reviews to serve as a guide for customers, by extracting relevant sentences that are highly representative of specific topics. Those sentences would be part of a super review that summarizes the many different reviews for a product without human intervention.

This project explores a possible approach to generate summaries from multiple reviews in three steps:
    • Sentiment Analysis with Recurrent Neural Networks to predict customers rating from free text reviews.
    • Features importance calculation, where features are vocabulary words, using the framework SHAP.
    • Words ranking and sentences selection to synthetize a “super review”.

## Project Outcome

The algorithm proposed is able to provide meaningful summaries, on top of a Sentiment Model that using Recurrent Neural Networks, and in particular an LSTM (Long Short Term Memory) network, is able to achieve 91% accuracy on the task of predicting positive vs negative user rating. 

The goodness of the generated reviews is evaluated only from the point of view of their informative power, looking at the customer total votes of the reviews selected by the algorithm to compose the super review. The score was then compared to the summaries produced by a simpler alternative method. It was found that the latter outperforms the proposed algorithm with products having less than 400 reviews, although the difference between the two algorithms becomes not statistically significant when the number of reviews increase.

## How to run Code

The basic requirements are in the requirements.txt file, you can install with the following command line instruction _pip install -r requirements.txt_. We used Tensorflow 1.14 rather than Tensorflow due to compatibility issues with Shap.

Overall this code was built in a conda environment, so using jupyterlab would minimize any dependency issues.

To run the code, the following steps:

1. Download the GloVe word embedding from http://nlp.stanford.edu/data/glove.6B.zip, unzip and only take the *glove.6B.100d.txt* file.
2. Run 01.Musical Instruments Reviews EDA.ipynb to download and unzip the necessary data (for this only the first couple of cells in the notebook are necessary)
3. Run 02.Sentiment_Model.ipynb for the LSTM Model
4. Run 03.Super Reviews to calculate shap values and synthetize reviews using the proposed algorithm