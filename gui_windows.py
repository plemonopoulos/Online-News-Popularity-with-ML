import tkinter
import numpy
import pandas
from tkinter import *
import tkinter.messagebox
import subprocess
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression


## define action listeners
def graphs():
    subprocess.Popen(["C://Users//Petros//Desktop//Graphs.pdf"], shell=True)
    #subprocess.Popen(["C://Users//Petros//Desktop//graphsvisualization//heatmap.png"], shell=True)
    #subprocess.Popen(["C://Users//Petros//Desktop//graphsvisualization//histogram1.png"], shell=True)
    #subprocess.Popen(["C://Users//Petros//Desktop//graphsvisualization//histogram2.png"], shell=True)
    #subprocess.Popen(["C://Users//Petros//Desktop//graphsvisualization//histogram3.png"], shell=True)
    #subprocess.Popen(["C://Users//Petros//Desktop//graphsvisualization//pie.png"], shell=True)
    #subprocess.Popen(["C://Users//Petros//Desktop//graphsvisualization//days.png"], shell=True)
    #subprocess.Popen(["C://Users//Petros//Desktop//graphsvisualization//chan.png"], shell=True)
    #subprocess.Popen(["C://Users//Petros//Desktop//graphsvisualization//type.png"], shell=True)
    #subprocess.Popen(["C://Users//Petros//Desktop//graphspreprocess//outliers_yes.png"], shell=True)
    #subprocess.Popen(["C://Users//Petros//Desktop//graphspreprocess//outliers_no.png"], shell=True)
    #subprocess.Popen(["C://Users//Petros//Desktop//graphspreprocess//Feature scores.png"], shell=True)

def linearRegression():
	subprocess.Popen(["C://Users//Petros//Desktop//performances//performances Regression//LinearRegression.pdf"], shell=True)

def decisionTreeRegressor():
	subprocess.Popen(["C://Users//Petros//Desktop//performances//performances Regression//DecisionTree.pdf"], shell=True)

def randomForestRegressor():
	subprocess.Popen(["C://Users//Petros//Desktop//performances//performances Regression//RandomForest.pdf"], shell=True)

def linearRegressionFS():
	subprocess.Popen(["C://Users//Petros//Desktop//performances//performances Regression//LinearRegressionFS.pdf"], shell=True)

def decisionTreeRegressorFS():
	subprocess.Popen(["C://Users//Petros//Desktop//performances//performances Regression//DecisionTreeFS.pdf"], shell=True)

def randomForestRegressorFS():
	subprocess.Popen(["C://Users//Petros//Desktop//performances//performances Regression//RandomForestFS.pdf"], shell=True)

def logisticRegression():
	subprocess.Popen(["C://Users//Petros//Desktop//performances//performances Classification//LogisticRegression.pdf"], shell=True)

def knn():
	subprocess.Popen(["C://Users//Petros//Desktop//performances//performances Classification//KNearestNeighbours.pdf"], shell=True)

def naiveBayes():
	subprocess.Popen(["C://Users//Petros//Desktop//performances//performances Classification//NaiveBayes.pdf"], shell=True)


def logisticRegressionFS():
	subprocess.Popen(["C://Users//Petros//Desktop//performances//performances Classification//LogisticRegressionFS.pdf"], shell=True)

def knnFS():
	subprocess.Popen(["C://Users//Petros//Desktop//performances//performances Classification//KNearestNeighboursFS.pdf"], shell=True)

def naiveBayesFS():
	subprocess.Popen(["C://Users//Petros//Desktop//performances//performances Classification//NaiveBayesFS.pdf"], shell=True)



def predReg():
    print("prediction regression")
    reg = RegressionClass()

def predClass():
    print("prediction classification")
    cl = ClassificationClass()


class RegressionClass(tkinter.Tk):
    def __init__(self):
        
        predReg = tkinter.Tk()
        predReg.title("Prediction for Regression")
        predReg.geometry("400x400")
        predReg.configure(bg = '#6699ff')

        def runRegression():
            print("run regression")

            myinput = E1.get()
            wordList = myinput.split()
            wordList = numpy.array(wordList)
            wordList = wordList.astype(numpy.float)
			
            data2 = pandas.read_csv("C://Users//Petros//Desktop//datasets//OnlineNewsPopularity_Regression FS.txt")
            data2 = numpy.array(data2)
            X2 = data2[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
            y2 = data2[:, [22]]
            y2 = y2.reshape((numpy.shape(y2)[0],))
            print(numpy.shape(X2))
            print(numpy.shape(y2))

            print(wordList)
            DT = DecisionTreeRegressor(random_state=42).fit(X2, y2)
            pred = DT.predict(wordList.reshape(1, -1))
            print("prediction = " + str(pred[0]))
            predReg.destroy()
            tkinter.messagebox.showinfo("Prediction", "Prediction = " + str(pred[0]) + " shares")

        label = Label(predReg, text="Input: ")
        label.place(x = 50 , y = 50)
        E1 = Entry(predReg, bd =2)
        E1.place(x = 100, y = 50)

        ok = tkinter.Button(predReg, text ="    Run    ", command = runRegression)
        ok.place(x = 100, y = 300)
        quit = tkinter.Button(predReg, text ="    Quit    ", command = predReg.destroy)
        quit.place(x = 200, y = 300)

class ClassificationClass(tkinter.Tk):
    def __init__(self):
        
        predCL = tkinter.Tk()
        predCL.title("Prediction for Classification")
        predCL.geometry("400x400")
        predCL.configure(bg = '#6699ff')

        def runClassification():
            print("run classification")

            myinput = E1.get()
            wordList = myinput.split()
            wordList = numpy.array(wordList)
            wordList = wordList.astype(numpy.float)
            
            data2 = pandas.read_csv("C://Users//Petros//Desktop//datasets//OnlineNewsPopularity_Classification.txt")
            data2 = numpy.array(data2)
            X2 = data2[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55]]
            y2 = data2[:, [56]]
            y2 = y2.reshape((numpy.shape(y2)[0],))
            print(numpy.shape(X2))
            print(numpy.shape(y2))
  
            print(wordList)
            LogReg = LogisticRegression(dual=True).fit(X2, y2)
            pred = LogReg.predict(wordList.reshape(1, -1))
            print("prediction = " + str(pred[0]))
            predCL.destroy()
            if pred[0] == 1:
                tkinter.messagebox.showinfo("Prediction", "Prediction = popular article")
            if pred[0] == 0:
                tkinter.messagebox.showinfo("Prediction", "Prediction = unpopular article")

        label = Label(predCL, text="Input: ")
        label.place(x = 50 , y = 50)
        E1 = Entry(predCL, bd =2)
        E1.place(x = 100, y = 50)

        ok = tkinter.Button(predCL, text ="    Run    ", command = runClassification)
        ok.place(x = 100, y = 300)
        quit = tkinter.Button(predCL, text ="    Quit    ", command = predCL.destroy)
        quit.place(x = 200, y = 300)



## create the graphical user interface
top = tkinter.Tk()
top.title("My GUI")
top.geometry("600x450")
top.configure(bg = '#6699ff')


## graphs
Bgraphs = tkinter.Button(top, text ="Dataset Visualization", command = graphs)
Bgraphs.place(x = 200, y = 20)

## tools regression
BLinReg = tkinter.Button(top, text ="Linear Regression     ", command = linearRegression)
BLinReg.place(x = 20, y = 100)

BDecTree = tkinter.Button(top, text ="DecisionTreeRegressor     ", command = decisionTreeRegressor)
BDecTree.place(x = 180, y = 100)

BRanFor = tkinter.Button(top, text ="Random Forest Regressor     ", command = randomForestRegressor)
BRanFor.place(x = 370, y = 100)

## tools regression FS
BLinRegFS = tkinter.Button(top, text ="Linear Regression FS", command = linearRegressionFS)
BLinRegFS.place(x = 20, y = 140)

BDecTreeFS = tkinter.Button(top, text ="DecisionTreeRegressor FS", command = decisionTreeRegressorFS)
BDecTreeFS.place(x = 180, y = 140)

BRanForFS = tkinter.Button(top, text ="Random Forest Regressor FS", command = randomForestRegressorFS)
BRanForFS.place(x = 370, y = 140)

## tools classification
BLogReg = tkinter.Button(top, text ="Logistic Regression     ", command = logisticRegression)
BLogReg.place(x = 20, y = 230)

BKNN = tkinter.Button(top, text ="K Nearest Neighbours     ", command = knn)
BKNN.place(x = 200, y = 230)

BNaiveBayes = tkinter.Button(top, text ="NAive Bayes     ", command = naiveBayes)
BNaiveBayes.place(x = 390, y = 230)

## tools classification FS
BLogRegFS = tkinter.Button(top, text ="Logistic Regression FS", command = logisticRegressionFS)
BLogRegFS.place(x = 20, y = 270)

BKNNFS = tkinter.Button(top, text ="K Nearest Neighbours FS", command = knnFS)
BKNNFS.place(x = 200, y = 270)

BNaiveBayesFS = tkinter.Button(top, text ="NAive Bayes FS", command = naiveBayesFS)
BNaiveBayesFS.place(x = 390, y = 270)

## tools prediction
BRegression = tkinter.Button(top, text ="Regression", command = predReg)
BRegression.place(x = 100, y = 380)

BClassification = tkinter.Button(top, text ="Classification", command = predClass)
BClassification.place(x = 300, y = 380)

top.mainloop()
