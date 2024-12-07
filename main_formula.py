import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filepath = "data.csv"

def showFeatureOrLabelOptions(df: pd.DataFrame):
    print("Possible feature/label options: ")
    for col in df.columns:
        print(col)

def inputCSV(filename):
    return pd.read_csv(filename)

def setFeature(dataframe: pd.DataFrame):
    colName = input("Enter feature column name: ")
    if colName in dataframe.columns:
        plt.xlabel(colName)
        return np.array(dataframe[colName])
    else:
        print("feature not found")
        return None
    
def setLabel(dataframe: pd.DataFrame):
    colName = input("Enter label column name: ")
    if colName in dataframe.columns:
        plt.ylabel(colName)
        return np.array(dataframe[colName])
    else:
        print("label not found")
        return None

def train(x, y, n):
    print("training model (non iteratively)")
    sumX=float(np.sum(x))
    sumY=float(np.sum(y))
    sumXY = float(np.sum(x*y))
    sumX2 = float(np.sum(x**2))
    b = ((sumX*sumXY)-(sumY*sumX2))/(((sumX)**2) - (n*sumX2))
    w = ((sumX*sumY)-(n*sumXY))/((sumX*sumX) - (n*sumX2))
    return w, b

def getPoints(w, b):
    if b==0:
        return (0, 0), (1, w)
    return (-b/w, 0), (0,b)

def graph(x, y, w, b, point1, point2):
    plt.scatter(x, y, alpha=0.1)
    plt.title(f"Weight: {w:.4f} | Bias: {b:.4f}")
    plt.axline(point1, point2)
    plt.show()

# """CODE WITHOUT FUNCTIONS: Prototype"""
# count = 1
# num = int(input("Enter number of points: "))
# x = list()
# y = list()
# while count<=num:
#     x.append(float(input(f"Enter x coordinates of point {count}: ")))
#     y.append(float(input(f"Enter y coordinates of point {count}: ")))
#     count+=1
# x = np.array(x)
# y = np.array(y)
# sumX=float(np.sum(x))
# sumY=float(np.sum(y))
# sumXY = float(np.sum(x*y))
# sumX2 = float(np.sum(x**2))

# b = ((sumX*sumXY)-(sumY*sumX2))/(((sumX)**2) - (num*sumX2))
# w = ((sumX*sumY)-(num*sumXY))/(((sumX)**2) - (num*sumX2))

# print(w, b)

# plt.axline((-b/w, 0), (0,b))
# plt.scatter(x, y)
# plt.title("Hey")
# plt.show()

if __name__ == "__main__":
    myDf = inputCSV(filepath)
    myNum = len(myDf)
    showFeatureOrLabelOptions(myDf)
    myX = setFeature(myDf)
    myY = setLabel(myDf)
    myW, myB = train(myX, myY, myNum)
    myPoint1, myPoint2 = getPoints(myW, myB)
    graph(myX, myY, myW, myB, myPoint1, myPoint2)
    