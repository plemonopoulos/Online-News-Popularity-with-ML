import Tkinter
import numpy
import pandas
from Tkinter import *
import tkMessageBox
import subprocess
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression


#####################################################################################################################
## define action listeners
def graphs():
    subprocess.call(['xdg-open','/home/apostolis/0. ergasia/graphs/graphs visualization/heatmap.png'])
    subprocess.call(['xdg-open','/home/apostolis/0. ergasia/graphs/graphs visualization/histogram1.png'])
    subprocess.call(['xdg-open','/home/apostolis/0. ergasia/graphs/graphs visualization/histogram2.png'])
    subprocess.call(['xdg-open','/home/apostolis/0. ergasia/graphs/graphs visualization/histogram3.png'])
    subprocess.call(['xdg-open','/home/apostolis/0. ergasia/graphs/graphs visualization/pie.png'])
    subprocess.call(['xdg-open','/home/apostolis/0. ergasia/graphs/graphs visualization/days.png'])
    subprocess.call(['xdg-open','/home/apostolis/0. ergasia/graphs/graphs visualization/chan.png'])
    subprocess.call(['xdg-open','/home/apostolis/0. ergasia/graphs/graphs visualization/type.png'])
    subprocess.call(['xdg-open','/home/apostolis/0. ergasia/graphs/graphs preprocess/outliers_yes.png'])
    subprocess.call(['xdg-open','/home/apostolis/0. ergasia/graphs/graphs preprocess/outliers_no_zoom.png'])
    subprocess.call(['xdg-open','/home/apostolis/0. ergasia/graphs/graphs preprocess/outliers_no.png'])

    subprocess.call(['xdg-open','/home/apostolis/0. ergasia/graphs/graphs preprocess/Feature scores.png'])

def linearRegression():
    subprocess.call(['xdg-open','/home/apostolis/0. ergasia/performances/performances Regression/LinearRegression.pdf'])

def decisionTreeRegressor():
    subprocess.call(['xdg-open','/home/apostolis/0. ergasia/performances/performances Regression/DecisionTree.pdf'])

def randomForestRegressor():
    subprocess.call(['xdg-open','/home/apostolis/0. ergasia/performances/performances Regression/RandomForest.pdf'])

def linearRegressionFS():
    subprocess.call(['xdg-open','/home/apostolis/0. ergasia/performances/performances Regression/LinearRegressionFS.pdf'])

def decisionTreeRegressorFS():
    subprocess.call(['xdg-open','/home/apostolis/0. ergasia/performances/performances Regression/DecisionTreeFS.pdf'])

def randomForestRegressorFS():
    subprocess.call(['xdg-open','/home/apostolis/0. ergasia/performances/performances Regression/RandomForestFS.pdf'])

def logisticRegression():
    subprocess.call(['xdg-open','/home/apostolis/0. ergasia/performances/performances Classification/LogisticRegression.pdf'])
def knn():
    subprocess.call(['xdg-open','/home/apostolis/0. ergasia/performances/performances Classification/KNearestNeighbours.pdf'])
def naiveBayes():
    subprocess.call(['xdg-open','/home/apostolis/0. ergasia/performances/performances Classification/NaiveBayes.pdf'])

def logisticRegressionFS():
    subprocess.call(['xdg-open','/home/apostolis/0. ergasia/performances/performances Classification/LogisticRegressionFS.pdf'])
def knnFS():
    subprocess.call(['xdg-open','/home/apostolis/0. ergasia/performances/performances Classification/KNearestNeighboursFS.pdf'])
def naiveBayesFS():
    subprocess.call(['xdg-open','/home/apostolis/0. ergasia/performances/performances Classification/NaiveBayesFS.pdf'])



def predReg():
    print("prediction regression")
    reg = RegressionClass()

def predClass():
    print("prediction classification")
    cl = ClassificationClass()


## create window for regression
class RegressionClass(Tkinter.Tk):
    def __init__(self):
        #Tkinter.Tk.__init__(self)
        
        predReg = Tkinter.Tk()
        predReg.title("Prediction for Regression")
        predReg.geometry("400x400")
        predReg.configure(bg = '#6699ff')

        def runRegression():
            print("run regression")

            myinput = E1.get()
            wordList = myinput.split()
            wordList = numpy.array(wordList)
            wordList = wordList.astype(numpy.float)

            data2 = pandas.read_csv("/home/apostolis/0. ergasia/datasets/OnlineNewsPopularity_Regression FS.txt")
            data2 = numpy.array(data2)
            X2 = data2[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
            y2 = data2[:, [22]]
            y2 = y2.reshape((numpy.shape(y2)[0],))
            print(numpy.shape(X2))
            print(numpy.shape(y2))
  
            print(wordList)
            DT = DecisionTreeRegressor(random_state=42).fit(X2, y2)
            pred = DT.predict(wordList)
            print("prediction = " + str(pred[0]))
            predReg.destroy()
            tkMessageBox.showinfo("Prediction", "Prediction = " + str(pred[0]) + " shares")

        label = Label(predReg, text="Input: ")
        label.place(x = 50 , y = 50)
        E1 = Entry(predReg, bd =2)
        E1.place(x = 100, y = 50)

        ok = Tkinter.Button(predReg, text ="    Run    ", command = runRegression)
        ok.place(x = 100, y = 300)
        quit = Tkinter.Button(predReg, text ="    Quit    ", command = predReg.destroy)
        quit.place(x = 200, y = 300)

## create window for classification
class ClassificationClass(Tkinter.Tk):
    def __init__(self):
        #Tkinter.Tk.__init__(self)
        
        predCL = Tkinter.Tk()
        predCL.title("Prediction for Classification")
        predCL.geometry("400x400")
        predCL.configure(bg = '#6699ff')

        def runClassification():
            print("run classification")

            myinput = E1.get()
            wordList = myinput.split()
            wordList = numpy.array(wordList)
            wordList = wordList.astype(numpy.float)
            
            data2 = pandas.read_csv("/home/apostolis/0. ergasia/datasets/OnlineNewsPopularity_Classification.txt")
            data2 = numpy.array(data2)
            X2 = data2[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55]]
            y2 = data2[:, [56]]
            y2 = y2.reshape((numpy.shape(y2)[0],))
            print(numpy.shape(X2))
            print(numpy.shape(y2))
  
            print(wordList)
            LogReg = LogisticRegression(dual=True).fit(X2, y2)
            pred = LogReg.predict(wordList)
            print("prediction = " + str(pred[0]))
            predCL.destroy()
            if pred[0] == 1:
                tkMessageBox.showinfo("Prediction", "Prediction = popular article")
            if pred[0] == 0:
                tkMessageBox.showinfo("Prediction", "Prediction = unpopular article")

        label = Label(predCL, text="Input: ")
        label.place(x = 50 , y = 50)
        E1 = Entry(predCL, bd =2)
        E1.place(x = 100, y = 50)

        ok = Tkinter.Button(predCL, text ="    Run    ", command = runClassification)
        ok.place(x = 100, y = 300)
        quit = Tkinter.Button(predCL, text ="    Quit    ", command = predCL.destroy)
        quit.place(x = 200, y = 300)



## create the graphical user interface
top = Tkinter.Tk()
top.title("My GUI")
top.geometry("600x450")
top.configure(bg = '#6699ff')

## graphs
Bgraphs = Tkinter.Button(top, text ="Dataset Visualization", command = graphs)
Bgraphs.place(x = 200, y = 20)

## tools regression
BLinReg = Tkinter.Button(top, text ="Linear Regression     ", command = linearRegression)
BLinReg.place(x = 20, y = 100)

BDecTree = Tkinter.Button(top, text ="DecisionTreeRegressor     ", command = decisionTreeRegressor)
BDecTree.place(x = 180, y = 100)

BRanFor = Tkinter.Button(top, text ="Random Forest Regressor     ", command = randomForestRegressor)
BRanFor.place(x = 370, y = 100)

## tools regression FS
BLinRegFS = Tkinter.Button(top, text ="Linear Regression FS", command = linearRegressionFS)
BLinRegFS.place(x = 20, y = 140)

BDecTreeFS = Tkinter.Button(top, text ="DecisionTreeRegressor FS", command = decisionTreeRegressorFS)
BDecTreeFS.place(x = 180, y = 140)

BRanForFS = Tkinter.Button(top, text ="Random Forest Regressor FS", command = randomForestRegressorFS)
BRanForFS.place(x = 370, y = 140)

## tools classification
BLogReg = Tkinter.Button(top, text ="Logistic Regression     ", command = logisticRegression)
BLogReg.place(x = 20, y = 230)

BKNN = Tkinter.Button(top, text ="K Nearest Neighbours     ", command = knn)
BKNN.place(x = 200, y = 230)

BNaiveBayes = Tkinter.Button(top, text ="NAive Bayes     ", command = naiveBayes)
BNaiveBayes.place(x = 390, y = 230)

## tools classification FS
BLogRegFS = Tkinter.Button(top, text ="Logistic Regression FS", command = logisticRegressionFS)
BLogRegFS.place(x = 20, y = 270)

BKNNFS = Tkinter.Button(top, text ="K Nearest Neighbours FS", command = knnFS)
BKNNFS.place(x = 200, y = 270)

BNaiveBayesFS = Tkinter.Button(top, text ="NAive Bayes FS", command = naiveBayesFS)
BNaiveBayesFS.place(x = 390, y = 270)

## tools prediction
BRegression = Tkinter.Button(top, text ="Regression", command = predReg)
BRegression.place(x = 100, y = 380)

BClassification = Tkinter.Button(top, text ="Classification", command = predClass)
BClassification.place(x = 300, y = 380)

top.mainloop()
