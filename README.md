# Naive Bayes’ classifier for test classification
* Use python ./naive_bayes.py <test filename ><pred filename > to test.
### Python Implementation:
1. Dictionary Creation.
* Each line of training data is read and is split with ’tab’. The label is currently ignored.
* Read line is stripped of every character but A-Z, a-z, ’whitespace’ and 0. Then, the line is converted to lowercase characters.
* Each word with length > 2 is stemmed using Porter Stemmer (NLTK) and is added to the dictionary, if not already present.
2. Feature Vector creation. (bag-of-words model)
* Each word in the dictionary is a feature.
* In this model, each sample is represented as the bag of its words, disregarding grammar and even word order but keeping multiplicity.
* For each line in training data, a vector with the count of features (dictionary words) is created. 
3. Fitting the classifier.
* Multinomial and Bernoulli models are imported from sklearn.naive bayes package used to fit the training feature vectors and class labels.

### Naive Bayes classifier for multinomial models

* The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification).
* Assumption: the probability of each word event in a document is independent of the word’s context and position in the document.
* The distribution is parametrized by vectors θy = (θy1, . . . , θyn) for each class y, where n is the number of features (dictionary size) and θyi is the probability P (xi | y) of feature i appearing in a sample belonging to class y.
* The parameters θy is estimated by a smoothed version of maximum likelihood, i.e. relative frequency counting: θˆ y i = N y i + α
Ny + αn
where Nyi =  x∈T xi is the number of times feature i appears in a sample of class y in the training set T, and
Ny =  |T | Nyi is the total count of all features for class y. i=1
* For α = 1.0, resulting estimate (θˆyi) will be between the empirical estimate Nyi/Ny, and the uniform probability 1/n.
Naive Bayes classifier for multivariate Bernoulli models.
* Like MultinomialNB, this classifier is suitable for discrete data. This class requires samples to be represented as binary-valued feature vectors i.e., there may be multiple features but each one is assumed to be a binary-valued (Bernoulli, boolean) variable.
* Assumption: the probability of each word occurring in a document is independent of the occurrence of other words in a document.
      1
* The decision rule for Bernoulli naive Bayes is based on
P(xi |y)=P(i|y)xi +(1−P(i|y))(1−xi)
which differs from multinomial NB’s rule in that it explicitly penalizes the non-occurrence of a feature i that is
an indicator for class y, where the multinomial variant would simply ignore a non-occurring feature.

### Results:
* Naive Bayes classifier for multivariate Bernoulli model. – Training data accuracy : 0.98875
  – Test data accuracy : 0.981
* Naive Bayes classifier for multinomial model.
  – Training data accuracy : 0.99 – Test data accuracy : 0.982

### Verdict :
Multinomial distribution fits better to given problem :
* Multiplicity of each feature (dictionary words) in data should be taken into account as both positive and negative words may occur in large reviews (data point). So, the Multinomial model is more accurate for data sets that have a large variance in review length.
* Our dictionary has not included various positive and negative words (as they were not in training set) but can be used in test data. Bernoulli model explicitly penalizes the non-occurrence of a feature whereas Multinomials implicitly encode this information in the probability distributions of words for each class.
Though it is fair not to be able to classify on basis of new words but for short reviews with less known words, the explicit penalty may hurt the performance.
