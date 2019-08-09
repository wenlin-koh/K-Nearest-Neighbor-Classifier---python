from knn import *
import time

def knnTest(knnFn):
  print("\nRunning {}".format(knnFn.__name__))
  kList = [1, 5, 10, 15, 20]
  percentTTList = [0.7, 0.6, 0.4]
  cvList = [5, 10, 15]
  
  accuracyList = []
  performanceList = []
  

  # load data
  X = loadData("abalone.data")
  # perform normalization
  X_norm = dataNorm(X)
  
  # euclidean test
  # perform knn euclidean variant using each k
  for k in kList:
    k_accuracy = []
    k_runningTime = []

    # perform knn euclidean variant using each percentTT
    for percentTT in percentTTList:
      X_norm_copy = np.copy(X_norm)
      print("\nrunning {:.1f}-{:.1f} Train-and-Test split with k = {}".format(percentTT, 1.0 - percentTT, k))
      
      # start time
      start = time.time()
      #data split: train-and-test
      X_split = splitTT(X_norm_copy, percentTT)
  
      # test split
      if testNorm(X_split) == testNorm([X_norm_copy]):
        print("Splitting is correct")
      
      # perform knn and collect accuracy
      acc = knnFn(X_split[0], X_split[1], k)
      
      # end time
      end = time.time()
  
      print("Accuracy = {}".format(acc))
      print("Performance time = {} seconds".format(end - start))
      k_accuracy.append(acc)
      k_runningTime.append(end - start)
  
    # perform knn euclidean variant using each cv value
    for cv in cvList:
      X_norm_copy = np.copy(X_norm)
      print("\nrunning {}-fold Cross Validation split with k = {}".format(cv, k))
      
      # start time
      start = time.time()
      # data split: Cross-validation
      X_split = splitCV(X_norm_copy, cv)
    
      # test split
      if testNorm(X_split) == testNorm([X_norm_copy]):
        print("Normalization is correct")
    
      # Build training and testing list from X_split
      trainingList = []
      testingList = []
    
      for i in range(cv):
        testingList.append(X_split[i])
        trainingSet = []
        for j in range(cv):
          if i == j:
            continue
          trainingSet.append(X_split[j])
        
        # concatenate the training sets to form one 2D array of training data
        trainingArr = np.vstack(trainingSet[0])
        for j in range(1, len(trainingSet)):
          trainingArr = np.vstack((trainingArr, trainingSet[j]))
        trainingList.append(trainingArr)
    
      # perform knn and collect accuracy
      sumAcc = 0.0
      for i in range(cv):
        sumAcc += knnFn(trainingList[i], testingList[i], k)
      
      acc = sumAcc / cv
      
      # end time
      end = time.time()
    
      print("Accuracy = {}".format(acc))
      print("Performance time = {} seconds".format(end - start))
      k_accuracy.append(acc)
      k_runningTime.append(end - start)
  
    accuracyList.append(k_accuracy)
    performanceList.append(k_runningTime)
  
  id = 0
  for a in accuracyList:
    print("Accuracy for k = {} : {}".format(kList[id], [round(n, 6) for n in accuracyList[id]]))
    id += 1

  id = 0
  for p in performanceList:
    print("Run-time for k = {} : {}".format(kList[id], [round(n, 6) for n in performanceList[id]]))
    id += 1

knnTest(knnMinkow)