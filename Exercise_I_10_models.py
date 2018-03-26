from sklearn.datasets import load_svmlight_file
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from statistics import mean
import datetime

# disable warnings
import warnings
warnings.filterwarnings("ignore")

# initial parameter
num_fold = 10

# load data
data = load_svmlight_file('datafile.txt')
X, y = shuffle(data[0].toarray(), data[1], random_state=0)

# classifiers
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

for i in range(len(classifiers)):
    start = datetime.datetime.now()
    p = mean(cross_val_score(classifiers[i], X, y, cv=num_fold, scoring='precision'))
    r = mean(cross_val_score(classifiers[i], X, y, cv=num_fold, scoring='recall'))
    f = 2 * r * p / (r + p)
    end = datetime.datetime.now()
    print('classifier = ', names[i], 'precision = ', p, 'recall = ', r, 'f1-score = ', f, 'time cost = ', end - start)
