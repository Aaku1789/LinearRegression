import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, tag) -> None:
        self._tag = tag
        self._data = None

    def setDataset(self, path:str, csv=True):
        if csv:
            self._df = pd.read_csv(path)
        print(f"Set the dataset of Linear Regression \"{self._tag}\"")
    
    def _showFeatureOrLabelOptions(self):
        if self._df is None:
            raise Exception("Try to set the dataset first using setDataset")
        print("Possible feature/label options: ")
        for col in self._df.columns:
            print(col)
    
    def getHyperParameters(self):
        learningR = float(input("Enter the learning rate: "))
        if learningR<=0:
            raise Exception("Learning Rate cannot be less than or equal to 0")
        self._learningRate = learningR
        batchS = int(input("Enter Batch Size: "))
        if batchS <= 0:
            raise Exception("Batch Size cannot be zero or a negative value.")
        self._batchSize = batchS
        epoch = int(input("Enter number of Epochs: "))
        if epoch <=0:
            raise Exception("Epochs cannot be zero or negative.")
        self._epochs = epoch

    def setFeature(self, featureName = None):
        '''Passing no value for featureName asks the user for the feature.'''
        if self._df is None:
            raise Exception("Dataframe not found. try calling setDataset() first")
        if featureName is None:
            self._showFeatureOrLabelOptions()
            featureName = input("Enter feature column name: ")
        if self._data is None:
            self._data = np.zeros((len(self._df), 2))
        if featureName not in self._df.columns:
            raise Exception(f"Dataframe does not have a column named \"{featureName}\"")
        self._data[:, 0] = np.array(self._df[featureName])
        print(f"Set feature of Linear Regression \"{self._tag}\" as \"{featureName}\": {len(self._df[featureName])} entries.")
        self._featureName = featureName
    
    def setLabel(self, labelName = None):
        '''Passing no value for featureName asks the user for the label.'''
        if self._df is None:
            raise Exception("Dataframe not found. try calling setDataFrame() first")
        if labelName is None:
            self._showFeatureOrLabelOptions()
            labelName = input("Enter label column name: ")
        if self._data is None:
            self._data = np.zeros((len(self._df), 2))
        if labelName not in self._df.columns:
            raise Exception(f"Dataframe does not have a column named \"{labelName}\"")
        self._data[:, 1] = np.array(self._df[labelName])
        print(f"Set label of Linear Regression \"{self._tag}\" as \"{labelName}\": {len(self._df[labelName])} entries.")
        self._labelName = labelName
    
    def _generateBatches(self, batchsize) -> list:
        np.random.shuffle(self._data)
        batchBoundaries = [batchsize*i for i in range((len(self._data)//batchsize)+1)]
        if len(self._data)%batchsize != 0:
            batchBoundaries.append(len(self._data))
        # print(batchBoundaries)
        batches = list()
        counter = 0
        while counter<(len(batchBoundaries)-1):
            batches.append(self._data[batchBoundaries[counter]:batchBoundaries[counter+1]])
            counter+=1
        return batches

        # Old code
        # data = np.array([zip(self._x, self._y)])
        # np.random.shuffle(data)
        # batchBoundaries = [batchsize*i for i in range(len(self._x)//batchsize)]
        # batchBoundaries.append(self._x)
        # xBatch = list()
        # yBatch = list()
        # counter = 0
        # while counter<(len(batchBoundaries)-1):
        #     xBatch.append([element[0] for element in data[batchBoundaries[counter]:batchBoundaries[counter+1]]])
        #     yBatch.append([element[1] for element in data[batchBoundaries[counter]:batchBoundaries[counter+1]]])
        #     counter+=1

    def _lossGradient(self):
        x = self._currBatch[:, 0]
        y = self._currBatch[:, 1]
        slopeGradient = np.sum(2*x*((self._weight*x) + self._bias - y), dtype=np.float32)
        biasGradient = np.sum(2*((self._weight*x)+self._bias-y), dtype=np.float32)
        return slopeGradient, biasGradient

    def _loss(self):
        x = self._data[:, 0]
        y = self._data[:, 1]
        return np.sum((y - (self._weight*x) - self._bias)**2)

    def train(self, learningRate = None, batchSize = None, epochs = None):
        self._learningRate = learningRate if learningRate is not None else self._learningRate
        self._batchSize = batchSize if batchSize is not None else self._batchSize
        self._epochs = epochs if batchSize is not None else self._epochs
        if self._learningRate <= 0 or self._epochs <= 0 or self._batchSize <= 0:
            raise Exception("Hyperparameters cannot be zero or negative")
        self._lossValue = np.zeros((self._epochs, 2))
        self._lossValue[:,0] = np.arange(start=1, stop = self._epochs+1)
        currentEpoch = 1
        self._weight: float = 0
        self._bias: float = 0
        print(self._learningRate, self._batchSize, self._epochs)
        while currentEpoch <= self._epochs:
            # Code for one epoch
            batches = self._generateBatches(self._batchSize)
            print(f"Training epoch {currentEpoch}/{self._epochs} ... {len(batches)} batches... Loss = ", end="")
            for self._currBatch in batches:
                # Code for one batch
                gradientW, gradientB = self._lossGradient()
                self._weight -= gradientW*self._learningRate
                self._bias -= gradientB*self._learningRate
            self._lossValue[currentEpoch-1, 1] = self._loss()
            print(self._lossValue[currentEpoch-1, 1])
            currentEpoch+=1


    def _getLinePoints(self):
        if self._bias==0:
            return (0, 0), (1, self._weight)
        return (-(self._bias)/self._weight, 0), (0,self._bias)
    
    def showPlot(self):
        self._dataPlot()
        self._lossPlot()
        plt.suptitle(f"Linear Regression: {self._tag}\nLearning Rate: {self._learningRate} | Batch Size: {self._batchSize} | Epochs: {self._epochs}")
        plt.show()

    def _dataPlot(self):
        pointX, pointY = self._getLinePoints()
        plt.subplot(1, 2, 1)
        plt.scatter(self._data[:,0], self._data[:,1], alpha=0.1)
        plt.axline(pointX, pointY)
        plt.xlabel(self._featureName)
        plt.ylabel(self._labelName)
        plt.title(f"Weight = {self._weight} | Bias = {self._bias}")

    def _lossPlot(self):
        plt.subplot(1, 2, 2)
        plt.plot(self._lossValue[:, 0], self._lossValue[:, 1])
        plt.title("Loss curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

if __name__ == "__main__":
    lr = LinearRegression("Taxi fares in Chicago")
    lr.setDataset("data.csv")
    lr.setFeature()
    lr.setLabel()
    lr.getHyperParameters()
    lr.train()
    lr.showPlot()
