from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance_matrix as dist
from random import sample, shuffle
import pandas as pd, numpy as np

def k_medoids_c(samples, k= 5):
    medoids = []
    medoids.extend(sample(samples.tolist(), k))

    dMatrix = dist(medoids, samples, p=float("inf"))
    error = sum(np.min(dMatrix, axis= 1))

    while True:
        for i in range(k):
            medoid = sample(samples.tolist(), 1)[0]
            while medoid in medoids:
                medoid = sample(samples.tolist(), 1)[0]
            
            medoids[i], medoid = medoid, medoids[i]

            alter_dMatrix = dist(medoids, samples, p=float("inf"))
            alter_error = sum(np.min(alter_dMatrix, axis= 1))

            if error < alter_error:
                medoids[i] = medoid
            else:
                error = alter_error
                dMatrix = alter_dMatrix
                
        if error == alter_error:
            break

    return np.argmin(dMatrix.tolist(), axis= 0) + 1
    
def gaussian_nb(x_train, y_train, x_test):
    #Train
    labels, counts = np.unique(y_train, return_counts=True)
    prior = dict(zip(labels, counts / len(y_train)))

    mean = {}
    var = {}
    for label in labels:
        features = x_train[y_train == label]
        mean[label] = np.mean(features, axis= 0)
        var[label] = np.var(features, axis= 0) + 1e-32

    #Test
    predicts = []
    for sample in x_test:
        posterior = [
            np.prod((1 / np.sqrt(2 * np.pi * var[label])) * np.exp(-((sample - mean[label]) ** 2) / (2 * var[label]))) * prior[label]
            for label in labels
            ]  
        predicts.append(np.argmax(posterior))

    return labels[predicts]
    
def splitter(samples, test_size = 0.3):
    if 0 >= test_size >= 1:
        raise ValueError
    
    shuffle(samples)
    splitArg = round((1 - test_size) * (len(samples)))
  
    try:
        train = samples[:splitArg, :]
        test = samples[splitArg:, :]
    except ValueError:
        print("\nInvalid test_size !!!")
    except:
        print("\nSample length error !!!")
    
    return train[:, :-1], test[:, :-1], train[:, -1], test[:, -1]

def accuracy(y_test, predict):
    s = 0
    l = len(y_test)
    for i in range(l):
        if y_test[i] == predict[i]:
            s += 1

    return s/l

if __name__=="__main__":
    dataset = np.array(pd.read_csv("sorlie.csv", header=None))
    x, y = dataset[:, :-1], dataset[:, -1]

    classes = k_medoids_c(x, k= 5)
    print(f"\nK-medoids:\n{classes}")
    x = np.hstack((x, classes.reshape(len(classes), 1)))
    pd.DataFrame.to_csv(pd.DataFrame(x), "sorlie2.csv")
    
    #holdout
    rep = 100
    accuracy = 0
    for _ in range(rep):
        #x_train, x_test, y_train, y_test = splitter(dataset, test_size= 0.3)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3)

        pred = gaussian_nb(x_train, y_train, x_test)
        
        #accuracy += accuracy(y_test, pred) / rep
        accuracy += accuracy_score(y_test, pred) / rep

    print(f"\nAccuracy of gaussian in {rep} iterations: {accuracy * 100}\n")