import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for r,p in zip(real_labels,predicted_labels):
        if r == 1:
            if p == 1:
                tp += 1
            elif p == 0:
                fn += 1
        elif r == 0:
            if p == 1:
                fp += 1
            elif p == 0:
                tn += 1
    f1 = tp/(tp+(0.5*(fp+fn)))
    return f1
    raise NotImplementedError


class Distances:
    @staticmethod
    # TODO
    def canberra_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        cd = 0
        for p1,p2 in zip(point1,point2):
            if (abs(p1)+abs(p2)) != 0:
                cd += abs(p1-p2)/(abs(p1)+abs(p2))
        return cd
        raise NotImplementedError

    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        md = 0
        for p1,p2 in zip(point1,point2):
            md += abs((p1-p2)**3)
        md = md**(1/3)
        return md
        raise NotImplementedError

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        ed = 0
        for p1,p2 in zip(point1,point2):
            ed += (p1-p2)**2
        ed = ed**(1/2)
        return ed
        raise NotImplementedError

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        ipd = 0
        for p1,p2 in zip(point1,point2):
            ipd += p1*p2
        return ipd
        raise NotImplementedError

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        csd = 0
        p1s = 0
        p2s = 0
        for p1,p2 in zip(point1,point2):
            csd += p1*p2
            p1s += p1*p1
            p2s += p2*p2
        p1s = p1s**(1/2)
        p2s = p2s**(1/2)
        if (p1s*p2s) != 0:
            csd = 1-(csd/(p1s*p2s))
        else:
            csd = 1-csd
        return csd
        raise NotImplementedError

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        gkd = 0
        for p1,p2 in zip(point1,point2):
            gkd += (p1-p2)**2
        gkd = -(np.exp((-0.5)*gkd))
        return gkd
        raise NotImplementedError


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [canberra > minkowski > euclidean > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None
        bestf = float("-inf")
        
        for d in distance_funcs:
            for k in range(1,min(len(x_train),30),2):
                knnmodel = KNN(k, distance_funcs[d])
                knnmodel.train(x_train, y_train)
                pred = knnmodel.predict(x_val)
                f1 = f1_score(y_val,pred)
                if f1 > bestf:
                    bestk = k
                    bestf = f1
                    bestd = d
                    bestmodel = knnmodel
        
        # You need to assign the final values to these variables
        self.best_k = bestk
        self.best_distance_function = bestd
        self.best_model = bestmodel
        #raise NotImplementedError

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [canberra > minkowski > euclidean > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None
        bestf = float("-inf")
        self.best_scaler = None
        
        for s in scaling_classes.keys():
            if s == "min_max_scale":
                minmax_scaler = MinMaxScaler()
                x_t = minmax_scaler(x_train)
                x_v = minmax_scaler(x_val)
            else:
                normal_scaler = NormalizationScaler()
                x_t = normal_scaler(x_train)
                x_v = normal_scaler(x_val)
            for d in distance_funcs.keys():
                for k in range(1,min(len(x_train),30),2):
                    knnmodel = KNN(k, distance_funcs[d])
                    knnmodel.train(x_t, y_train)
                    pred = knnmodel.predict(x_v)
                    f1 = f1_score(y_val,pred)
                    if f1>bestf:
                        bestk = k
                        bestf = f1
                        bestd = d
                        bests = s
                        bestmodel = knnmodel
                
        
        # You need to assign the final values to these variables
        self.best_k = bestk
        self.best_distance_function = bestd
        self.best_scaler = bests
        self.best_model = bestmodel
        #raise NotImplementedError


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        normal_f = []
        for f in features:
            ss = 0
            for i in f:
                ss += i**2
            ss = ss**(1/2)
            temp = []
            for i in f:
                if ss != 0:
                    i = i/ss
                temp.append(i)
            normal_f.append(temp)
        return normal_f
        raise NotImplementedError


class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.a = 0
        pass

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        minmax_f = []
        if self.a == 0:
            self.minf = []
            self.maxf = []
            for i in range(len(features)):
                if i == 0:
                    for j in range(len(features[i])):
                        self.minf.append(features[i][j])
                        self.maxf.append(features[i][j])
                else:
                    for j in range(len(features[i])):
                        if self.minf[j] > features[i][j]:
                            self.minf[j] = features[i][j]
                        if self.maxf[j] < features[i][j]:
                            self.maxf[j] = features[i][j]
            for i in range(len(features)):
                temp = []                
                for j in range(len(features[i])):
                    x = features[i][j]
                    if (self.maxf[j] - self.minf[j]) != 0:
                        x = (x - self.minf[j])/(self.maxf[j] - self.minf[j])
                        
                    else:
                        x = (x - self.minf[j])
                    temp.append(x)
                minmax_f.append(temp)
            self.a = 1
        elif self.a == 1:
            for i in range(len(features)):
                temp = []                
                for j in range(len(features[i])):
                    x = features[i][j]
                    if (self.maxf[j] - self.minf[j]) != 0:
                        x = (x - self.minf[j])/(self.maxf[j] - self.minf[j])
                        
                    else:
                        x = (x - self.minf[j])
                    temp.append(x)
                minmax_f.append(temp)
        return minmax_f
        raise NotImplementedError
