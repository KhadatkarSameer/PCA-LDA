import numpy as np
import os
from pathlib import Path
from numpy import asarray
from PIL import Image
import scipy.linalg as la
from  matplotlib import pyplot

#input: Centered feature vector
#returns U(eigenvector-matrix) and L(diagonal-matrix with diagonal elements as eigen-values)
def highdim_PCA(fvector_centered):
    # print(fvector[:,0].shape)
    D, N = fvector_centered.shape

    ST = fvector_centered.T @ fvector_centered
    ST /= N
    S = fvector_centered @ fvector_centered.T
    S /= N
    # print(S.shape)
    eigenValues, eigenVectors = la.eig(ST)

    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx].real
    eigenVectors = eigenVectors[:, idx]

    u = np.zeros((D, N))
    for i in range(N):
        u[:, i:i + 1] = ((1 / ((N * eigenValues[i]) ** 2)) * (fvector_centered @ eigenVectors[:, i:i + 1])).reshape(
            (10201, 1))
        u[:, i:i + 1] = u[:, i:i + 1] / la.norm(u[:, i:i + 1])

    # print((S @ u[:, 0] - eigenValues[0] * u[:, 0]).max())

    L = np.zeros((N, N))
    for i in range(N):
        if eigenValues[i] > 0:
            L[i][i] = eigenValues[i] ** (-0.5)
        else:
            L[i][i] = 0
    return u,L

#input: U matrix, L matrix, dimension, Centered feature vector
#returns Whitening transform and Whitenned data
def compute_whitening_data(u,L,k,fvector_centered):
    D,N=u.shape
    W = L[:k, :k] @ u[:, :k].T
    y_n = np.zeros((k, N))
    for i in range(N):
        y_n[:, i:i + 1] = W @ fvector_centered[:, i:i + 1]
    return y_n, W

#input: Whitening transform of training data, Centered feature vector of test data
#returns Whitenned test data
def whitened_feature_vector_for_test_data(images,images_bar,W,k):
    D, N = images.shape
    fvector_centered = np.zeros((D, N))
    for i in range(N):
        fvector_centered[:, i:i + 1] = (images[:, i:i + 1] - images_bar)
    y_n = np.zeros((k, N))
    for i in range(N):
        y_n[:, i:i + 1] = W @ fvector_centered[:, i:i + 1]
    return y_n

#input: train data
#returns Centered train data
def center_train_data(images):
    D, N = images.shape

    images_bar = np.zeros((D, 1))
    for i in range(N):
        images_bar += images[:, i:i + 1]
    images_bar /= N
    fvector_centered = np.zeros((D, N))
    for i in range(N):
        fvector_centered[:, i:i + 1] = (images[:, i:i + 1] - images_bar)
    return fvector_centered ,images_bar

#input: test data, mean of train ddta
#returns Centered test data
def center_test_data(images,images_bar):
    D, N = images.shape
    fvector_centered = np.zeros((D, N))
    for i in range(N):
        fvector_centered[:, i:i + 1] = (images[:, i:i + 1] - images_bar)
    return fvector_centered

#input: folder of images
#returns feature vectors of data as matrix and there actual class labels
def load_images_from_folder(folder):
    listc = os.listdir(folder)
    length = len(listc)
    images=np.zeros((101*101,length))
    i=0
    for filename in listc:
        image = Image.open(os.path.join(folder,filename))
        data = asarray(image)
#         data = data[:100,:100]
#         print(data.shape)
        for k in range(101*101):
            images[k][i]=data[k%101][k//101]
        if (filename[10:13]=='hap'):
            listc[i]=0
        if (filename[10:13]=='sad'):
            listc[i]=1
        i+=1
    return images,listc

#input: Whitenned datd
#returns Direction of LDA
def get_LDA_diection(y_n):
    D_new, N = y_n.shape
    happy_num = 0
    sad_num = 0
    happy_bar = np.zeros((D_new, 1))
    sad_bar = np.zeros((D_new, 1))
    for i in range(N):
        if listc[i] == 1:
            sad_bar += y_n[:, i:i + 1]
            sad_num += 1
        if listc[i] == 0:
            happy_bar += y_n[:, i:i + 1]
            happy_num += 1
    happy_bar /= happy_num
    sad_bar /= sad_num
    S_B = (sad_bar - happy_bar) @ (sad_bar - happy_bar).T
    # print(S_B.shape)
    S_W = np.zeros(S_B.shape)
    # print(((y_n[:, 0:1] - sad_bar) @ (y_n[:, 0:1] - sad_bar).T).shape)
    for i in range(N):
        if listc[i] == 1:
            S_W += (y_n[:, i:i + 1] - sad_bar) @ (y_n[:, i:i + 1] - sad_bar).T
        if listc[i] == 0:
            S_W += (y_n[:, i:i + 1] - happy_bar) @ (y_n[:, i:i + 1] - happy_bar).T

    S_L = la.inv(S_W) @ S_B
    dire = la.inv(S_W) @ (sad_bar - happy_bar)
    dire = dire / la.norm(dire)

    return dire

########################################################################################################################
image_path = Path('Data/emotion_classification/train')
images,listc = load_images_from_folder(image_path.as_posix())
images_centered, images_bar = center_train_data(images)
U,L = highdim_PCA(images_centered)

test_path = Path('Data/emotion_classification/test')
test,list_test=load_images_from_folder(test_path.as_posix())
no_test_data=len(list_test)

for k in range(1,images_centered.shape[1]):
    y_n, W = compute_whitening_data(U,L,k,images_centered)
    dire = get_LDA_diection(y_n)
    plotcord = y_n.T @ dire
    plotcord1 = []
    plotcord2 = []
    r = 0
    t = 0
    for i in range(y_n.shape[1]):
        if (listc[i] == 0):
            plotcord1.append(plotcord[i])
            r += 1
        else:
            plotcord2.append(plotcord[i])
            t += 1
    clkey = -1
    plotcord1 = np.array(plotcord1)
    plotcord2 = np.array(plotcord2)
    directory = {}
    happy_class_mean = plotcord1.mean()
    sad_class_mean = plotcord2.mean()

    #to find which class lies on which side of threshold
    if happy_class_mean < sad_class_mean:
        if plotcord1.max()<plotcord2.min():  #for finding whether data is sperated or not
            print(f"Well seperated data at k = {k}")
            threshold = (plotcord2.min() + plotcord1.max()) / 2  #to find threshold
            fig1 = pyplot.plot(plotcord1, np.zeros((len(plotcord1))), "o", label="Happy")
            fig2 = pyplot.plot(plotcord2, np.zeros((len(plotcord2))), "o", label="Sad")
            fig3 = pyplot.plot(np.full((50), threshold), np.linspace(-0.5, 0.5, num=50), "|", label="Threshold")
            pyplot.title(f"LDA of Training Data at k = {k}", fontsize=25)
            pyplot.legend(loc="upper left", fontsize=15)
            pyplot.show()
            print(f"seperability = {sad_class_mean-happy_class_mean}") #to find seperability
            clkey=0
        else:
            print(f"Data not well seperated for k = {k}")
            fig1 = pyplot.plot(plotcord1, np.zeros((len(plotcord1))), "o", label="Happy")
            fig2 = pyplot.plot(plotcord2, np.zeros((len(plotcord2))), "o", label="Sad")
            pyplot.title(f"LDA of Training Data at k = {k}", fontsize=25)
            pyplot.legend(loc="upper left", fontsize=15)
            pyplot.show()
    else:
        if plotcord2.max() < plotcord1.min(): #for finding whether data is sperated or not
            print(f"Well seperated data at k = {k}")
            threshold = (plotcord1.min() + plotcord2.max()) / 2 #to find threshold
            fig1 = pyplot.plot(plotcord1, np.zeros((len(plotcord1))), "o", label="Happy")
            fig2 = pyplot.plot(plotcord2, np.zeros((len(plotcord2))), "o", label="Sad")
            fig3 = pyplot.plot(np.full((50), threshold), np.linspace(-0.5, 0.5, num=50), "|", label="Threshold")
            pyplot.legend(loc="upper left", fontsize=15)
            pyplot.title(f"LDA of Training Data at k = {k}", fontsize=25)
            pyplot.show()
            print(f"seperability = {happy_class_mean - sad_class_mean}") #to find seperability
            clkey = 1
        else:
            print(f"Data not well seperated for k = {k}")
            fig1 = pyplot.plot(plotcord1, np.zeros((len(plotcord1))), "o", label="Happy")
            fig2 = pyplot.plot(plotcord2, np.zeros((len(plotcord2))), "o", label="Sad")
            pyplot.legend(loc="upper left", fontsize=15)
            pyplot.title(f"LDA of Training Data at k = {k}", fontsize=25)
            pyplot.show()
    ############################################### Testing ########################################################################
    if clkey!= -1:
        y_n_t = whitened_feature_vector_for_test_data(test, images_bar, W, k)
        t = y_n_t.T @ dire
        cl_predict = np.zeros((no_test_data))
        plotcord3 = []
        plotcord4 = []
        for i in range(no_test_data):
            if clkey == 0:
                if t[i] < threshold:
                    cl_predict[i] = int(0)
                    plotcord3.append(t[i])
                else:
                    cl_predict[i] = int(1)
                    plotcord4.append(t[i])
            else:
                if t[i] < threshold:
                    cl_predict[i] = int(1)
                    plotcord4.append(t[i])
                else:
                    cl_predict[i] = int(0)
                    plotcord3.append(t[i])

        print("Actual Classes")
        print(list_test)
        print("Predicted Classes")
        print(cl_predict)

        accuracy = 0
        for i in range(no_test_data):
            if cl_predict[i] == list_test[i]:
                accuracy += 1

        print(f"threshold = {threshold}")
        accuracy = accuracy * 100 / no_test_data
        print(f'Accuracy for k={k} is {accuracy}%')
    print('------------------------------------------')
    ########################################################################################################################
