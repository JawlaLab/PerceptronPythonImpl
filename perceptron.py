# Created by JawlaLab/Himanshu July 03, 2024
# Code is not optimized for GPU usaage.
import random
import numpy as np
import matplotlib.pyplot as plt

#Creating a class of perceptron.
class perceptron:
    #constructor 
    def __init__(self,inputNodes = 2, weightLow = 0, weightHigh=1, bias = True):
        #creating the list to appened the accuracy data into it to track progress
        self.accuracy =[]
        #number of input nodes
        self.inputNodes = inputNodes
        # weightLow - WeightHigh defines the initial range of the weights in a preceptron.
        self.weightLow = weightLow
        self.weightHigh = weightHigh
        # if bias is there or not
        self.bias = bias
        #creating a list to store the value of weights
        self.weights =[]
        # append "inputNodes" number of weights to the list,each weight value for each input node
        for i in range(0,inputNodes):
            # getting a random number from range weightLow to weightHigh and append it to the list weights
            randomNumber = weightLow + (random.random()*(weightHigh -weightLow))
            self.weights.append(randomNumber)
        #If there is a bias then append an extra weight for bias,(input will be considered as one)
        if(bias):
            randomNumber = weightLow + (random.random()*(weightHigh -weightLow))
            self.weights.append(randomNumber)
        self.weightsnp = np.array(self.weights)
        

    #print Values of the object
    def printValues(self):
        print("Number of Input Nodes: ",self.inputNodes)
        print(f"Value of weight ranges from {self.weightLow} to {self.weightHigh}")
        if self.bias: print("Bias Applied")
        print("Value of weights are: ", self.weights)
        

    #defining the step function (as activation function)
    def stepFunction(self,summedVal):
        if (summedVal>=0):
            return 1
        else:
            return 0
        
        
    #get output from the perceptron
    def getOutput(self,inputValsa):
        inputVals = inputValsa[:]
        if (len(inputVals)!= self.inputNodes):
            print("Number of input Values are not proper, please enter proper number of input Values")
        else:
            if (self.bias):
                inputVals.append(1)
            inputValsNp = np.array(inputVals)
            self.finalArray = inputValsNp*self.weightsnp
            self.outputSum = np.sum(self.finalArray)
            self.finalOutput = self.stepFunction(self.outputSum)
            return(self.finalOutput)
    
    
    #get output list for multiple sets of inputs
    def getoutputMulti(self,inputarray):
        outputs =[]
        for i in range(0,len(inputarray)):
            outputs.append(self.getOutput(inputarray[i]))
        return outputs
    
    
    #Compare the output with the desired output
    def checkResult(self,testSet):
        inputSet= testSet[0]
        desiredOutputSet=testSet[1]
        currentOutputSet = self.getoutputMulti(inputSet)
        if (len(desiredOutputSet)!=len(currentOutputSet)):
            print("The Set mismatch")
        else:
            
            correctOutputCounts = 0
            for i in range(0,len(currentOutputSet)):
                if(currentOutputSet[i]==desiredOutputSet[i]):
                    correctOutputCounts+=1
            successpercentage = (correctOutputCounts/len(currentOutputSet))*100
            return successpercentage
        
        
    # This function only works with Two Inputs
    def visualizeForTwoInputs(self,inputdataplot=[]):
        if (self.inputNodes!=2):
            print("This Function only works when there are only two Inputs")
        else:
            xs = np.linspace(0, 100, 100)
            ys = ((-1*self.weightsnp[0])*xs - self.weightsnp[2])/self.weightsnp[1]
            plt.plot(xs, ys)
            if (len(inputdataplot)>0):
                inputdata = inputdataplot[0]
                outputdata= inputdataplot[1]
                #print("input data: ",inputdata)
                #print("output data: ",outputdata)
                positivedatax=[]
                positivedatay=[]
                negitivedatax=[]
                negitivedatay=[]
                for i in range(0,len(outputdata)):
                    if (outputdata[i]==1):
                        positivedatax.append(inputdata[i][0])
                        positivedatay.append(inputdata[i][1])
                    if (outputdata[i]==0):
                        negitivedatax.append(inputdata[i][0])
                        negitivedatay.append(inputdata[i][1])
                plt.plot(positivedatax,positivedatay,'s')
                plt.plot(negitivedatax,negitivedatay,'s')
                plt.show()
                
                
    #Training of perceptron
    def learnPerceptron(self, dataset):
        weights = self.weightsnp
        #defining the step size
        self.stepSize =0.00002
        inputdataset  = dataset[0]
        Doutputdataset = np.array(dataset[1])

        for i in range(0,len(Doutputdataset)):
            errorF = Doutputdataset- np.array(self.getoutputMulti(inputdataset))
            if (errorF[i]!=0):
                # updating the weights
                weightsnp = self.stepSize*np.array(inputdataset[i]+[1])*errorF[i]
                self.weightsnp = self.weightsnp + weightsnp
                self.weights = self.weightsnp.tolist()
   

    #iterate learnPerceptron function
    def learniter(self,dataset,itera):
        for i in range(0,itera):
            self.learnPerceptron(dataset)
            self.accuracy.append(self.checkResult(dataset))
            #self.visualizeForTwoInputs(dataset)
        plt.plot(self.accuracy)
        plt.show()
        
        
a = perceptron(weightLow=-1,weightHigh = 1,inputNodes=2)
print(a.getOutput([1,1]))
a.printValues()
print(a.getoutputMulti([[1,2],[2,1]]))
#a.visualizeForTwoInputs()
checkTestSet = [[[47, 76], [23, 48], [30, 23], [5, 30], [26, 21], [2, 11], [1, 30], [1, 17], [78, 50], [70, 98], [43, 3], [94, 81], [1, 10], [39, 12], [52, 77], [18, 37], [94, 44], [47, 100], [84, 10], [74, 32], [63, 29], [67, 66], [66, 45], [45, 44], [78, 17], [32, 48], [46, 59], [16, 42], [0, 9], [15, 43], [53, 25], [71, 55], [46, 36], [98, 81], [94, 85], [27, 70], [70, 9], [63, 65], [63, 11], [11, 15], [47, 4], [78, 16], [57, 4], [21, 96], [22, 29], [9, 39], [38, 56], [61, 94], [1, 59], [98, 56], [86, 70], [64, 63], [100, 69], [2, 36], [45, 53], [99, 29], [50, 45], [55, 42], [57, 100], [94, 100], [58, 86], [75, 58], [14, 91], [85, 71], [30, 95], [24, 28], [98, 49], [75, 77], [83, 88], [41, 17], [70, 97], [91, 47], [14, 23], [43, 49], [28, 26], [0, 98], [7, 86], [47, 61], [85, 84], [32, 4], [67, 67], [37, 24], [34, 2], [51, 13], [48, 82], [81, 46], [85, 28], [60, 30], [50, 92], [35, 31], [84, 65], [67, 49], [90, 13], [99, 74], [88, 99], [54, 86], [1, 89], [84, 63], [78, 53], [52, 15]], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0]]
print(a.checkResult(checkTestSet))
a.visualizeForTwoInputs(checkTestSet)
a.learniter(checkTestSet,50)
