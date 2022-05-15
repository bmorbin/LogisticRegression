
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.datasets import make_blobs


def cross_entropy_loss(w, X, y):
    """
    Computes the loss (equation 1)
    :param w: weight vector
    :type: np.ndarray(shape=(1+d, ))
    :param X: design matrix
    :type X: np.ndarray(shape=(N, 1+d))
    :param y: class labels
    :type y: np.ndarray(shape=(N, ))
    :return loss: loss (equation 1)
    :rtype: float
    """    
    
    ### ===> Your code begins here
    N = y.shape[0]
    loss = (1/N)*sum(np.log(1+np.exp(-y*sum(np.transpose(w*X)))))
    return loss
    ### ===> Your code ends here



def cross_entropy_gradient(w, X, y):
    """
    Computes the gradient of the loss function (equation 2)
    :param w: weight vector
    :type: np.ndarray(shape=(1+d, ))
    :param X: design matrix
    :type X: np.ndarray(shape=(N, 1+d))
    :param y: class labels
    :type y: np.ndarray(shape=(N, ))
    :return grad: gradient (equation 2)
    :rtype: np.ndarray(shape=(1+d, ))
    """
    
    ### ===> Your code begins here 
    N = y.shape[0]
    
    den = 1+np.exp(y*sum(np.transpose(w*X)))
    num = y * np.transpose(X)
    summa = sum(np.transpose(num/den))

    grad = -(1/N)*summa

    return grad
    ### ===> Your code ends here



def train_logistic(X, y, learning_rate = 1e-1, w0 = None,\
                        num_iterations = 300, return_history = False):
    """
    Computes the weight vector applying the gradient descent technique
    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: class label
    :type y: np.ndarray(shape=(N, ))
    :return: weight vector
    :rtype: np.ndarray(shape=(1+d, ))
    :return: the history of loss values (optional)
    :rtype: list of float
    """    
     
    ### ===> Your code begins here
    X_with_ones = np.c_[np.ones(X.shape[0]), X]
    if w0==None: w0 = np.random.normal(loc = 0, scale = 1, size = X_with_ones.shape[1])
    history_loss = list()

    if return_history: 
        history_loss.append(cross_entropy_loss(w0, X_with_ones, y))

    wt = w0
    for t in range(0, num_iterations):
      # compute the gradient
      grad_t = cross_entropy_gradient(wt,X_with_ones,y)
      # setting the direction to move
      v_t = -grad_t
      # updating the weights
      wt = wt + learning_rate*v_t

      if return_history: 
        history_loss.append(cross_entropy_loss(wt, X_with_ones, y))
    
    if return_history: return(wt, history_loss)
    else: return wt
    ### ===> Your code ends here


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict_logistic(X, w):
    """
    Computes the logistic regression prediction
    :param X: design matrix
    :type X: np.ndarray(shape=(N,d))
    :param w: weight vector
    :rtype: np.ndarray(shape=(1+d,))
    :return: predicted classes 
    :rtype: np.ndarray(shape=(N,))
    """
    
    ### ===> Your code begins here
    X_with_ones = np.c_[np.ones(X.shape[0]), X]
    product = w @ np.transpose(X_with_ones)
    predict_score = sigmoid(product)
    predict_classes = np.array([-1 if s<0.5 else +1 for s in predict_score])

    return predict_classes
    ### ===> Your code ends here



# Create two blobs
N = 300
X, y = make_blobs(n_samples=N, centers=2, cluster_std=1, n_features=2, random_state=2)

# change labels 0 to -1
y[y==0] = -1

print("X.shape =", X.shape, "  y.shape =", y.shape)


fig = plt.figure(figsize=(6,6))

# plot negatives in red
plt.scatter(X[y==-1,0], \
            X[y==-1,1], \
            alpha = 0.5,\
            c = 'red')

# and positives in blue
plt.scatter(x=X[y==1,0], \
            y=X[y==1,1], \
            alpha = 0.5, \
            c = 'blue')

P=+1
N=-1
legend_elements = [ Line2D([0], [0], marker='o', color='r',\
                    label='Class %d'%N, markerfacecolor='r',\
                    markersize=10),\
                    Line2D([0], [0], marker='o', color='b',\
                    label='Class %d'%P, markerfacecolor='b',\
                    markersize=10) ]

plt.legend(handles=legend_elements, loc='best')
plt.show()


np.random.seed(567)

# ==> Replace the right hand side below with a call to the
# train_logistic() function defined above. Use parameter return_history=True

w_logistic, loss = train_logistic(X,y, return_history=True,
                                  learning_rate=1e1,
                                  num_iterations=3000)

# ==> Your code insert ends here

print()
print("Final weight:\n", w_logistic)
print()
print("Final loss:\n", loss[-1])

plt.figure(figsize = (12, 8))
plt.plot(loss)
plt.xlabel('Iteration #')
plt.ylabel('Cross Entropy Loss')
plt.show()


x1min = min(X[:,0])
x1max = max(X[:,0])
x2min = min(X[:,1])
x2max = max(X[:,1])

y_pred = predict_logistic(X, w_logistic)

fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(121)
ax1.set_title("Ground-truth")

# plot negatives in red
ax1.scatter(X[y==-1,0], \
            X[y==-1,1], \
            alpha = 0.5, \
            c = 'red')

# and positives in blue
ax1.scatter(x=X[y==1,0], \
            y=X[y==1,1], \
            alpha = 0.5, \
            c = 'blue')

ax2 = fig.add_subplot(122)

ax2.set_title("Prediction")
ax2.scatter(x = X[:,0], y = X[:,1], c = -y_pred, cmap = 'coolwarm')
ax2.legend(handles=legend_elements, loc='best')
ax2.set_xlim([x1min-1, x1max+1])
ax2.set_ylim([x2min-1, x2max+1])

p1 = (x1min, -(w_logistic[0] + (x1min)*w_logistic[1])/w_logistic[2])
p2 = (x1max, -(w_logistic[0] + (x1max)*w_logistic[1])/w_logistic[2])

lines = ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], '-')
plt.setp(lines, color='g', linewidth=4.0)

plt.show()

print(f"{len(y[(y!=y_pred)])} are misclassified points.")

#==============================================================================
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0).fit(X, y)

w_logistic2 = np.concatenate((clf.intercept_,np.reshape(clf.coef_, newshape=(2,))))
y_pred2 = clf.predict(X)


fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(121)
ax1.set_title("Ground-truth")

# plot negatives in red
ax1.scatter(X[y==-1,0], \
            X[y==-1,1], \
            alpha = 0.5, \
            c = 'green')

# and positives in blue
ax1.scatter(x=X[y==1,0], \
            y=X[y==1,1], \
            alpha = 0.5, \
            c = 'purple')

ax2 = fig.add_subplot(122)

ax2.set_title("Prediction")
ax2.scatter(x = X[:,0], y = X[:,1], c = -y_pred2, cmap = 'coolwarm')
ax2.legend(handles=legend_elements, loc='best')
ax2.set_xlim([x1min-1, x1max+1])
ax2.set_ylim([x2min-1, x2max+1])

p1 = (x1min, -(w_logistic2[0] + (x1min)*w_logistic2[1])/w_logistic2[2])
p2 = (x1max, -(w_logistic2[0] + (x1max)*w_logistic2[1])/w_logistic2[2])

lines = ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], '-')
plt.setp(lines, color='g', linewidth=4.0)

plt.show()



