![image](https://user-images.githubusercontent.com/61338647/169692725-9d521899-e6ee-4484-a8da-ecaab68d1802.png)

The insurance raw data is on google drive as follow: https://drive.google.com/file/d/1OOsw7EqX1BJqP4h-0D6SXjET4B81cDau/view?usp=sharing

### Assignment overview
When using machine learning for classification, things are easiest if classes are “balanced” – that is, when the number of observations belonging to each of the classes are of the same order of magnitude. Unfortunately, this is often not the case. In this assignment, you will work with a dataset of car-insurance claims and try to classify claims into fraudulent (1) and non- fraudulent (0). There are more than 10,000 claims in the dataset, but only around 100 are fraudulent. Nevertheless, we want to create a model that helps the insurance provider target its investigation efforts. For this, we consider two options: (i) synthetically creating new data to make the dataset more balanced, and (ii) using an auto-encoder to represent “normal” (non- fraudulent) claims and applying it to distinguish fraudulent claims – a form of anomaly detection.

### Task description
1. Briefly discuss why it is more difficult to find a good classifier on such a dataset than on one where, for example, 5,000 claims are fraudulent, and 5,000 are not. In particular, consider what happens when undetected fraudulent claims are very costly to the insurance company.
2. Load the dataset "Insurance_claims.csv" and clean it as appropriate for use with machine learning algorithms. A description of the features can be found at the end of this document.
3. Start by creating a (deep) neural network in TensorFlow and train it on the data. Using training and validation sets, find a model with high accuracy, then evaluate it on the test set. In particular, record both the accuracy and AUC. Briefly discuss what issues you observe based on the metrics.

### first approach 
will be to create new (synthetic) data points representing fraudulent claims and remove some of the non-fraudulent claims. This will allow us to create a more balanced dataset. The state-of-the-art method we apply is SMOTE (Synthetic Minority Oversampling Technique).

4. The file "SMOTE.ipynb" explains the process in detail and shows how to change the dataset with an example. You can copy and adjust the code to make it work within your analysis. You can adjust the "sampling_strategy" parameters as you see fit, particularly if you want to fine-tune your model in part 5.
5. Create a new (deep) neural network and train it on your enhanced dataset. Use training and validation sets derived from the enhanced dataset to find a model with high accuracy. Evaluate your final model on a test set consisting only of original data. Again, record the accuracy and AUC. Briefly discuss the changes you would expect in the metrics and the actual changes you observe. Would you say that you are now doing better at identifying fraudulent claims?

### second approach
will be to use an autoencoder to learn what "normal" (non-fraudulent) data "looks like."

6. Using the original data, create a training and set that contains only non-fraudulent claims, as well as validation and test sets that contain non-fraudulent and fraudulent claims. Make sure to spread fraudulent claims evenly across validation and test sets.
7. Using TensorFlow, create an autoencoder, ensuring that the middle hidden layer has fewer neurons than your input has features. Use training and validation sets to find a model that represents its input data well. In particular, you will want to predict your validation set observations. For each observation, you can measure the difference between the original observations and the predicted one, using, for example, the mean squared error of all features of the observation. Plot the errors for all your validation set observations in a histogram - in a good model, this error should be much higher for fraudulent claims than non-fraudulent ones.
8. Use your trained autoencoder to predict the test set and define the corresponding losses. Create a histogram of your test set claims, clearly marking fraudulent and non- fraudulent claims. Discuss how you could use this to decide whether a transaction is fraudulent or not. Can you also derive an AUC in this approach - if yes, how does it perform compared to the previous approaches?

### Transparency of deep learning approaches.
9. As you know, it is difficult to understand precisely why a neural network makes a specific prediction. Discuss why this might be problematic when the neural network prediction leads to a fraud investigation by the insurance company. What alternatives can you envision that make use of the techniques we have applied and allow for more interpretability and transparency?

### 
10. Use your synthetically extended dataset and train a simple model, such as logistic regression or a decision tree that allows you to interpret why fraud is suspected. Keep track of the accuracy and AUC on a test set made from original data only. How does your model perform compared to the previous models you have developed? Does your model allow you to answer a customer who asks, "why am I being investigated"?
