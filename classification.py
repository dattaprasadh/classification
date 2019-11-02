#!/usr/bin/env python
# coding: utf-8

# # Classification
# 
# In this course we are going to learn the concepts of Classification in Machine Learning.
# 
# ## Problem Statement
# 
# The task of this Course is to classify handwritten digits.
# 
# ![Classifying Handwritten digits](https://www.wolfram.com/mathematica/new-in-10/enhanced-image-processing/HTMLImages.en/handwritten-digits-classification/smallthumb_10.gif)
# 
# 
# ## About the MNIST dataset
# 
# It is a set of 70,000 small images of digits handwritten by high school students and employees of the US Census Bureau. Each image is labeled with the digit it represents. This set has been studied so much that it is often called the “Hello World” of Machine Learning: whenever people come up with a new classification algorithm, they are curious to see how it will perform on MNIST. Whenever someone learns Machine Learning, sooner or later they tackle MNIST.
# 
# 
# 

# ## Fetching the data
# 
# Scikit-Learn provides many helper functions to download popular datasets. MNIST is one of them. The following code fetches the dataset

# In[2]:


from sklearn.datasets import fetch_mldata
# fetch_mldata downloads data in the file structure scikit_learn_data/mldata/mnist-original.mat 
# in your home directory.
# you can also copy from our dataset using rsync -avz /cxldata/scikit_learn_data .
mnist = fetch_mldata("MNIST original")
mnist


# ### Structure of Datasets loaded by Scikit Learn
# 
# Datasets loaded by Sklearn have a dictionary structure. They have the following keys :
# 
# 1. A DESCR key describing the dataset
# 1. A data key containing an array with one row per instance and one column per feature
# 1. A target key containing an array with the labels
# 

# ### Looking into our Datasets
# 
# Out datasets consists of 70,000 images and each image has 784 features. A image consists of 28x28 pixels, and each pixel is a value from 0 to 255 describing the pixel intensity. 0 for white and 255 for black.
# 
# 
# ![MNIST dataset image](https://www.cntk.ai/jup/cntk103a_MNIST_input.png)

# In[3]:


X, y = mnist["data"], mnist["target"]
X.shape
X[1].shape


# Since there are 70,000 images with 28x28 pixels
# 

# In[4]:


y.shape


# This is label for each of the image. Since there are 70,000 images, hence there are 70,000 labels.

# ### Looking at one of the datasamples
# 
# To view the image of a single digit,all we need to do is grab an instance’s feature vector, reshape it to a 28×28 array, and display it using Matplotlib’s imshow() function.

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt


# In[6]:




some_digit = X[36000]   # Selecting the 36,000th image.
some_digit_image = some_digit.reshape(28, 28) # Reshaping it to get the 28x28 pixels

plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()


# In[7]:


plt.imshow(255-some_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest")


# The image looks like a 5. Let's verify it.

# In[8]:


some_digit_image.shape


# In[9]:


X[36000].shape


# In[10]:


y[36000]


# ## <font color='red'>Go to slide show</font>

# ## Test train split
# 
# > Train set is the data on which our Machine learning model is trained.
# 
# > Test set is the data on which our model is finally evaluated
# 
# ![Test Train Split](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Machine_learning_nutshell_--_Split_into_train-test_set.svg/640px-Machine_learning_nutshell_--_Split_into_train-test_set.svg.png)
# 
# We need to split the data into test and train data. The MNIST dataset is actually already split into a training set (the first 60,000 images) and a test set (the last 10,000 images)
# 

# In[11]:


X_train=X[5000:,:]
X_train.shape


# In[12]:


X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# Also we need to **shuffle** our training data so that it ensures that we don't miss out any digit in a cross validation fold.

# In[13]:


import numpy as np
np.random.seed(42)
shuffle_index = np.random.permutation(60000)
print (shuffle_index[1])
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# ### <font color='red'>Go to slide show</font>

# ## Classifying our Digits
# 
# ### Training a Binary Classifier
# 
# A perfect binary classifier looks something like this.
# ![Binary Classifier](https://classeval.files.wordpress.com/2015/06/perfect-classifier.png?w=480&h=192)
# 
# Let us first simplify our problem and make a model that only predicts if the digit is 5 or not. This will be a example of a "Binary Classifier".
# 
# Lets create target vector for our classification.

# In[14]:


y_train_5 = (y_train == 5)
print (y_train_5)
y_test_5 = (y_test == 5)


# Now let’s pick a classifier and train it. A good place to start is with a **Stochastic Gradient Descent (SGD)** classifier, using **Scikit-Learn’s SGDClassifier class**. This classifier has the advantage of being
# capable of handling very large datasets efficiently. This is in part because SGD deals with training instances independently, one at a time (which also makes SGD well suited for online learning). Let’s create an **SGDClassifier** and train it on an example set:

# Now, for binary classification '5' and 'Not 5', we train the SGD Classifier on the training dataset.

# In[15]:


from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42, max_iter=900) # if you want reproducible results set the random_state value.
sgd_clf.fit(X_train, y_train_5)


# ### Testing using sample data
# We remember that a digit 5 was stored in some_digit and we test the classifier on this sample data

# Since it gives a output of **True**, hence our binary classifier correctly identified the digit 5 from our dataset.

# In[16]:


sgd_clf.predict([X[39000]])


# In[19]:


y[39000]


# ### <font color='red'>Go to slideshow</font>

# ## Measuring the performace of our Classifier
# 
# Evaluating a classifier is often significantly trickier than evaluating a regressor.
# 
# ________
# + **Measuring Accuracy Using Cross-Validation**
# ______________
# We will use the cross_val_score() function to evaluate your **SGDClassifier model** using **K-fold cross-
# validation**, with three folds. **K-fold cross-validation** means splitting the training set into K-folds (in this case, three), then making predictions and evaluating them on each fold using a model trained on the remaining folds.
# 
# Here is a example how a 10 fold cross validation works.
# 
# ![A 10 fold cross validation](https://sebastianraschka.com/images/faq/evaluate-a-model/k-fold.png)

# In[50]:


get_ipython().run_line_magic('pinfo', 'cross_val_score')


# In[ ]:


from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=2, scoring="accuracy") 
# Since we need 3 folds hence we will set sv to 3


# In[ ]:


help(cross_val_score)


# This gives us the accuracy for all the 3 folds which is above 95% , which is a good accuracy. But accuracy is generally not the preferred performance measure for classifiers, especially when you are dealing with skewed datasets.
# 
# **What is a Skewed dataset ?**
# When some classes are much more frequent than others, then the dataset is said to be skewed.
# 
# Let's verify the fact that accuracy is not best for Skewed dataset. In our case only 10% of the data is 5. So its a skewed dataset. So if a classifier always predicts 5 than its accuracy is 90%. Lets see it in action.
# 
# 

# In[17]:


from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
    
never_5_clf = Never5Classifier()
never_5_pred = never_5_clf.predict(X_train)

cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")


# So in this case our accuracy is above 90%. **Hence we need a better measure of performace for our classifier.**

# ### <font color='red'>Go to slide show</font>

# ____________
# + **Measuring performace using a Confusion Matrix**
# ____________
# 
# A much better way to evaluate the performance of a classifier is to look at the **confusion matrix**. The general idea is to count the number of times instances of class A are classified as class B. For example, to know the number of times the classifier confused images of 5s with 3s, you would look in the 5th row and 3rd column of the confusion matrix.
# 
# To compute the confusion matrix, you first need to have a set of predictions, so they can be compared to the actual targets. You could make predictions on the test set, but let’s keep it untouched for now.

# In[18]:


from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
y_train_pred


# Just like the cross_val_score() function, cross_val_predict() performs K-fold cross-validation,
# but instead of returning the evaluation scores, it returns the predictions made on each test fold

# In[19]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)


# Each row in a confusion matrix represents an actual class, while each column represents a predicted class.

# The first row of this matrix considers non-5 images (the negative class):
# 
#  + 53,606 of them were correctly classified as non-5s (they are called true negatives)
#  + The remaining 973 were wrongly classified as 5s (false positives). 
#  
# The second row considers the images of 5s (the positive class): 
# 
# + 1,607 were wrongly classified as non-5s (false negatives)
# + The remaining 3,814 were correctly classified as 5s (true positives).
# 

# ### <font color='red'>Go to slide show</font>

# ______________
# + **Measuring accuracy using Precision, Recall and F1 score**
# ________

# **What is Precision, Recall on slides**
# 
# ![Formula for Precision and Recall](http://www2.isprs.org/tl_files/isprs/wg34/images/Precision_Recall_formula.png)
# 
# > **True - Positive**  means the classifier **correctly** classified the **Positive** class.
# 
# > **True - Negative**  means the classifier **correctly** classified the **Neative** class.
# 
# > **False - Positive**  means the classifier **incorrectly** classified a **Negative class** as **Positive Class.**
# 
# > **False - Negative**  means the classifier **incorrectly** classified a **Positive class** as **Negative Class.**
# 

# In[20]:


from sklearn.metrics import precision_score, recall_score

print ("Precision score is : " , precision_score(y_train_5, y_train_pred))
print ("Recall score is : " , recall_score(y_train_5, y_train_pred))


# In[21]:


never_5_pred = never_5_pred.reshape(-1,)
print ("Precision score is : " , precision_score(y_train_5, never_5_pred))
print ("Recall score is : " , recall_score(y_train_5, never_5_pred))


# In[22]:


from sklearn.base import BaseEstimator
class Always5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.ones((len(X), 1), dtype=bool)
    
always_5_clf = Always5Classifier()
always_5_pred = always_5_clf.predict(X_train)

always_5_pred = always_5_pred.reshape(-1,)
print ("Precision score is : " , precision_score(y_train_5, always_5_pred))
print ("Recall score is : " , recall_score(y_train_5, always_5_pred))


# ### <font color='red'>Go to slide show</font>

# It is often convenient to combine precision and recall into a single metric called the F1 score, in particular if you need a simple way to compare two classifiers. The F1 score is the harmonic mean of
# precision and recall.
# 
# 
# ![F1 score](https://hassetukda.files.wordpress.com/2012/08/f12.jpg)

# 2/F1 = 1/recall + 1/precision
# 
# 1. f1/2 < smaller of recall and precision
# 

# In[23]:


from sklearn.metrics import f1_score

f1_score(y_train_5, y_train_pred)


# The F1 score favors classifiers that have similar precision and recall. 
# 
# **Increasing precision reduces recall, and vice versa.**

# *Raising the **threshold** decreases **Recall** *

# In[56]:


get_ipython().run_line_magic('pinfo', 'sgd_clf.decision_function')


# In[60]:


y_scores = sgd_clf.decision_function([some_digit])
y_scores


# In[64]:


threshold = 0
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred


# In[62]:


# Setting the threshold to 20000
threshold = 20000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred


# This confirms that raising the threshold decreases recall.

# ### <font color='red'>Go to slide show</font>

# **Precision - Recall curve**
# 
# Let's try to find the score from the decision_function. Using these scores we will plot a relation between Precision and Recall, by setting different values of the Threshold.

# In[28]:


get_ipython().run_line_magic('pinfo', 'cross_val_predict')


# In[29]:


y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
method="decision_function")
len(y_scores)


# In[30]:


y_scores_1 = y_scores[:,1]


# In[31]:


from sklearn.metrics import precision_recall_curve
precision_recall_curve(y_train_5, y_scores_1)


# In[32]:


# Now we will use these values of y_scores to find different values for precison and recall for varying thresholds.

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores_1)


# In[33]:


# Plotting our results

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.figure(figsize=(16,5))
    # Removing last value to avoid divide by zero in precision computation
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()


# Now you can simply select the threshold value that gives you the best precision/recall tradeoff for your task.
# 
# Another way to select a good precision/recall tradeoff is to plot precision directly against recall directly.

# In[34]:


def plot_precision_vs_recall(precisions, recalls):
    plt.figure(figsize=(18,7))
    plt.plot(recalls[:-1], precisions[:-1], "b-", label="Precision")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])

plot_precision_vs_recall(precisions, recalls)
plt.show()


# ### <font color='red'>Go to slide show</font>

# **ROC curve**
# 
# 
# > The receiver operating characteristic (ROC) curve is another common tool used with binary classifiers.
# 
# > It is very similar to the precision/recall curve, but instead of plotting precision versus recall, the ROC curve plots the **true positive rate (another name for recall)** against the **false positive rate**.

# In[36]:


# Making the ROC curve


from sklearn.metrics import roc_curve

# y_scores_1 = y_scores[:,1]
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores[:, 1])

def plot_roc_curve(fpr, tpr, label=None):
    plt.figure(figsize=(18,5))
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(fpr, tpr)
plt.show()


# Once again there is a tradeoff: **the higher the recall (TPR), the more false positives (FPR) the classifier produces**. The dotted line represents the ROC curve of a purely random classifier; a good classifier stays as far away from that line as possible (toward the top-left corner).
# 
# 
# One way to compare classifiers is to measure the **area under the curve (AUC)**. A perfect classifier will have a ROC AUC equal to 1, whereas a purely random classifier will have a ROC AUC equal to 0.5.

# In[40]:


# Scikit-Learn provides a function to compute the ROC AUC:

from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores[:, 1])


# ## Comparision of SGDClassifier and RandomForestClassifier on the basis of ROC-AUC
# 
# Let’s train a RandomForestClassifier and compare its ROC curve and ROC AUC score to the
# SGDClassifier.
# 
# First, you need to get scores for each instance in the training set. But due to the way it
# works, the **RandomForestClassifier class** does not have a **decision_function()**
# method. Instead it has a **predict_proba()** method.
# 
# Scikit-Learn classifiers generally have one or the other.
# 
# > The **predict_proba()** method returns an array containing a row per instance and a column per
# class, each containing the probability that the given instance belongs to the given class (e.g., 70% chance that the image represents a 5):

# In[41]:


from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
y_probas_forest


# In[42]:


# But to plot a ROC curve, you need scores, not probabilities.
# A simple solution is to use the positive class’s probability as the score:

y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, "b:", label="SGD")
ax.plot(fpr_forest, tpr_forest, linewidth=2, label="Random Forest")
ax.plot([0, 1], [0, 1], 'k--')
ax.legend(loc=4);
ax.set_title('ROC curves');


# As you can see above, the **RandomForestClassifier’s ROC curve** looks much better than the **SGDClassifier’s** as it comes much closer to the top-left corner. As a result, its ROC AUC score is also significantly better.
# 
# 

# In[48]:


# ROC auc score of SGDClassifier

from sklearn.metrics import roc_auc_score
print ("The ROC AUC value for SGDClassifier : ", roc_auc_score(y_train_5, y_scores[:, 1]))
print ("The ROC AUC value for Random Forest Classifier is : " , roc_auc_score(y_train_5, y_scores_forest))


# ### <font color='red'>Go to slide show</font>

# ## Multiclass Classification
# 
# 
# > Binary classifiers distinguish between two classes, **multiclass classifiers (also called multinomial classifiers)** can distinguish between more than two classes.
# 
# ![Multiclass classification](https://raw.githubusercontent.com/ritchieng/machine-learning-stanford/master/w3_logistic_regression_regularization/multiclass_classification.png)
# 
# There are basically two strategies using which you can use multiple binary classifiers for multiclass classification.
# 
# + **One-versus-all (OvA) strategy also called one-versus-the-rest**
# 
#     For example, one way to create a system that can classify the digit images into 10 classes (from 0 to 9) is to train 10 binary classifiers, one for each digit (a 0-detector, a 1-detector, a 2-detector, and so on). Then when you want to classify an image, you get the decision score from each classifier for that image and you select the class whose classifier outputs the highest score.
# 
# 
# + **One-versus-one (OvO) strategy**
# 
#     This is another strategy in which we train a binary classifier for every pair of digits: one to distinguish 0s and 1s, another to distinguish 0s and 2s, another for 1s and 2s, and so on. If there are N classes, you need to train N × (N – 1) / 2 classifiers.

# Scikit-Learn detects when you try to use a binary classification algorithm for a multiclass classification task, and it automatically runs OvA (except for SVM classifiers for which it uses OvO). Let’s try this with the SGDClassifier:

# In[57]:


sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])


# Under the hood, Scikit-Learn actually trained **10 binary classifiers**, got their decision scores for **the image**, and selected the class with the highest score.
# 
# To see that this is indeed the case, you can call the decision_function() method. Instead of returning just one score per instance, it now **returns 10 scores**, one per class:

# In[62]:


y_train[1000]


# In[61]:


some_digit_scores = sgd_clf.decision_function([X_train[1000]])
some_digit_scores


# In[47]:


# The highest score is indeed the one corresponding to class 5:
print ("The index of the maxmimum score is ", np.argmax(some_digit_scores))


# **To force ScikitLearn to use one-versus-one or one-versus-all, you can use the OneVsOneClassifier or OneVsRestClassifier classes**
# 
# To create a multiclass classifier using the OvO strategy, based on a SGDClassifier, we can do the following :

# In[63]:


from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42, max_iter=20))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit])


# In[65]:


len(ovo_clf.estimators_)


# ### <font color='red'>Go to slide show</font>

# **Evaluating the accuracy of SGDClassifier**
# 

# In[68]:


cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")


# Simply scaling the inputs increases accuracy to above 90%

# In[72]:


100*(0.91196761-0.87312537)/0.87312537


# In[69]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")


# ## Error Analysis
# 
# First we'll make the Confusion Matrix. For this we need predictions.

# In[70]:


y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx


# It is often convenient to represent the confusion matrix using Matplotlib’s matshow() function.

# In[71]:


plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()


# Most images are on the main diagonal, which means that they were classified correctly.

# ### <font color='red'>Go to slide show</font>

# ** Calculating error rates instead of absoluter error and plotting **
# 
# We need to divide each value in the confusion matrix by the number of images in the corresponding class, so you can compare error rates instead of absolute number of errors (which would make abundant classes look unfairly bad). We perform the normalization and plot the results again.

# In[72]:


row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()


# ## What are your observations?

# **Observing the error plot and making inferences**
# 
# There are several observations that can be made using the above plotted graph and the potential remedies to the source of error can also be identified. [Refer Slide# ]

# In[73]:


# EXTRA
import os
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")


# In[75]:


cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
plt.show()


# ### <font color='red'>Go to slide show</font>

# ## Multilabel Classification

# 
# In some cases you may want your classifier to output multiple classes for each instance.
# 
# For example, consider a face-recognition classifier: what should it do if it recognizes several people on the same picture? Of course it should attach one label per person it recognizes. Say the classifier has been trained to recognize three faces, **Alice, Bob, and Charlie**; then when it is shown a picture of **Alice and Charlie**, it should output **[1, 0, 1] (meaning “Alice yes, Bob no, Charlie yes”).**
# 
# ![image.png](attachment:image.png)
# 
# Such a classification system that outputs multiple binary labels is called a **Multilabel classification system.**
# 

# **Doing Multilabel Classification using Scikit Learn**

# In[76]:


x = np.array([1, 2])
y = np.array([3, 4])
np.c_[x,y]


# In[77]:


y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)

y_multilabel = np.c_[y_train_large, y_train_odd]  # np.c_ is used to concatenate the two arrays element wise
y_multilabel


# **y_multilabel** array containing two target labels for each digit image
# 
# + the first indicates whether or not the digit is large (7, 8, or 9) 
# + and the second indicates whether or not it is odd. 

# In[81]:


from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)


# **KNeighborsClassifier** supports **multilabel classification** but not all classifiers do.

# In[82]:


knn_clf.predict([some_digit])


# Since the digit 5 is indeed not large (False) and odd (True).
# 
# There are many ways to evaluate a multilabel classifier, and selecting the right metric really depends on your project.
# 
# Here we will use F1 score as a example.

# In[ ]:


y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)
f1_score(y_train, y_train_knn_pred, average="macro")


# In[ ]:


# NOTE: To be reviewed
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)
f1_score(y_train, y_train_knn_pred, average="weighted")


# ## Multioutput Classification
# 
# It is simply a generalization of multilabel classification where each label can be **multiclass (i.e., it can have more than two possible values).**
# 
# 
# Here we'll build a system that removes noise from images. It will take as input a noisy digit image, and it will output a clean digit image, represented as an array of pixel intensities, just like the MNIST images.
# 
# Notice that the classifier’s output is **multilabel (one label per pixel)** and each label can have **multiple values (pixel intensity ranges from 0 to 255)**. It is thus an example of a **multioutput classification system.**
# 
# 

# For this example we'll be adding noise to our MNIST dataset.
# We will be generating random integer using **randint()** and adding to original image.

# In[10]:


import numpy.random as rnd

noise_train = rnd.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise_train
noise_test = rnd.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise_test
y_train_mod = X_train
y_test_mod = X_test


# Let's view the noisy image

# In[11]:


def plot_digit(array):
    array_image = array.reshape(28, 28)
    plt.imshow(array_image, cmap = matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()
    
plot_digit(X_test_mod[4000])


# In[12]:


plot_digit(y_test_mod[4000])


# Now we will clean the image using KNN classifier. It is a example of Multioutput classification. A single label is Multilabel as it has 784 classes and each of the 784 pixel can have values from 0 to 255, hence it is a Multioutput classification example.

# In[14]:


from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()


# In[15]:


knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[4000]])
plot_digit(clean_digit)


# In[89]:


knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[4000]])
plot_digit(clean_digit)


# In[17]:


knn_clf.kneighbors_graph()


# In[ ]:





# In[ ]:




