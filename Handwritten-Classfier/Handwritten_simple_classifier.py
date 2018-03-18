
# coding: utf-8


import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import numpy as np
learn = tf.contrib.learn
tf.logging.set_verbosity(tf.logging.INFO)


#loading the Data_sets from mnist
mnist = learn.datasets.load_dataset("mnist")

#separating data's as training and testing 
train_data = mnist.train.images
train_label = np.asarray(mnist.train.labels, dtype=np.int32)
test_data = mnist.test.images
test_label = np.asarray(mnist.test.labels, dtype=np.int32)



min_examples = 10000
data = train_data[:min_examples]
label = train_label[:min_examples]
print(len(data[0]))


def display(i):
    img = test_data[i]
    plt.title("Example: {} , Label: {}".format(i, test_label[i]))
    plt.imshow(img.reshape(28,28), cmap = plt.cm.gray_r)


display(0)
display(1)
display(8)

#well start with a simple classifier
feature_columns = learn.infer_real_valued_columns_from_input(data)
classifier = learn.LinearClassifier(n_classes=10, feature_columns= feature_columns)

#gradientDescent and all the weights are handled by this fit method underthehood
classifier.fit(data, label, batch_size = 100, steps = 1000)


#Testing our model once trained
classifier.evaluate(test_data, test_label)
# we achieved 91% of accurcay for our model      
print classifier.evaluate(test_data, test_label)['accuracy']                                      


#once get's right time to explore the model
def testing(n):
    new_samples = np.array([test_data[n]], dtype=int) 
    y = list(classifier.predict(new_samples, as_iterable=True)) 
    print("Predicted {0}, Label: {1}".format(str(y), test_label[n]))
    display(n)


testing(5)

testing(8)

testing(1)

#Visualizing learned weights
weights  = classifier.weights_
f, axes = plt.subplot(2, 5, figsize=(10,4))
axes = axes.reshape(-1)
for i in range(len(axes)):
    a = axes[i]
    a.imshow(weights.T[i].reshape(28,28), cmap = plt.cm.seismic)
    a.set_title(i)
    a.set_xticks(())
    a.set_yticks(())
plt.show()

