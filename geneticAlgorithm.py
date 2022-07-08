import random
import numpy as np
from sklearn.linear_model import *
import pandas
import house
import csv
import math
import warnings

"""
    This will run the genetic algorithm
    population: the population to run the genetic algorithm on
    fitness: the function that rates our population

    returns another generation
"""
def GeneticAlgorithm(population, fitness, lowestScore, highestScore, objectListType):
    nextPopulation = []
    weightedList = WeightedBy(population, fitness, objectListType)
    weightedPairs = WeightedRandomChoices(population, weightedList, fitness, lowestScore, highestScore, objectListType)

    #For Elitism
    nextPopulation.append(weightedPairs[0][0])
    nextPopulation.append(weightedPairs[0][1])

    for pair in weightedPairs:

        #Incorporates Elitism by carrying on the best two options
        if pair == weightedPairs[0]:
            continue

        nextPair = Reproduce(pair[0], pair[1])
        mutationChance = random.random()
        if mutationChance < .02 and mutationChance > .01:
            mutationObject = random.choice(nextPair)
            mutationObject.randomMutation()
        elif mutationChance < .01:
            nextPair[0].randomMutation()
            nextPair[1].randomMutation()

        nextPopulation.append(nextPair[0])
        nextPopulation.append(nextPair[1])

    return nextPopulation
        
    

"""
This function will generate a new list that has items paired up with each other based on score
The closest scored neighbors are paired together. The last item is dropped from the generation and the
second from last item in the generation is paired with the best item in the generation.

pairs neighbors, excludes last two neighbors and instead does second to last element paired with best element
"""
def WeightedRandomChoices(population, weights, fitness, lowestScore, highestScore, objectListType):
    resultPairs = []
    numOfHouses = len(weights)

    firstHouse = None
    secondHouse = None

    for house in weights:
        if firstHouse == None:
            firstHouse = house
        elif secondHouse == None:
            secondHouse = house
        else:
            resultPairs.append([firstHouse, secondHouse])
            firstHouse = house
            secondHouse = None

    
    #Generates two additional random instead of first and last
    num1 = 19
    num2 = 20

    randomOne = GenerateHouseInRange(num1, lowestScore, highestScore, objectListType, fitness)
    randomTwo = GenerateHouseInRange(num2, lowestScore, highestScore, objectListType, fitness)

    resultPairs.append([randomOne,randomTwo])
    return resultPairs
        

"""
Returned a population list with their fitness scores added to their object
"""
def WeightedBy(population, fitness, objectListType):
    weightedList = []
    populationList = []

    for item in population:
        if objectListType == "Partial":
            data = np.array(item.objectToList("Partial"))
            score = fitness.predict(data.reshape(1, -1))
            item.setProjectedSalePrice(score)
            weightedList.append(item)

        if objectListType == "Multi":
            data = np.array(item.objectToList("Multi"))
            data = data.reshape(1, -1)
            score = fitness.predict(data)
            item.setProjectedSalePrice(score)
            weightedList.append(item)

    organizedHouses = OrganizeListByFitness(weightedList, objectListType)
    
    return organizedHouses

"""
Returns a list with the population ordered from highest score to lowest score
"""
def OrganizeListByFitness(weightedList, objectListType):
    weights = weightedList
    organizedHouses = []
    for num in range(len(weightedList)):
        bestItem = None
        bestValue = None
        
        for weight in weights:
            if bestItem == None:
                bestValue = weight.ProjectedSalePrice
                bestItem = weight
            else:
                nextValue = weight.ProjectedSalePrice

                if objectListType == "Partial":
                    if bestValue < nextValue:
                        bestItem = weight
                        bestValue = nextValue


                if objectListType == "Multi":
                    if bestValue[0][0] < nextValue[0][0]:
                        bestItem = weight
                        bestValue = nextValue

        weights.remove(bestItem)

        organizedHouses.append(bestItem)
    return organizedHouses

"""
    returns an individual that has been created from randomly splitting and combining two other objects
"""
def Reproduce(parentOne, parentTwo):

    parentOneList = parentOne.objectToList("Full")
    parentTwoList = parentTwo.objectToList("Full")

    splittingIndex = random.randint(0, len(parentOneList))

    firstList = parentOneList[:splittingIndex] + parentTwoList[splittingIndex:]
    secondList = parentTwoList[:splittingIndex] + parentOneList[splittingIndex:]

    firstResultHouse = house.house([], [])
    firstResultHouse.listToObject(firstList, "Full")

    secondResultHouse = house.house([], [])
    secondResultHouse.listToObject(secondList, "Full")

    return [firstResultHouse, secondResultHouse]

"""
    This function is responsible for creating one house that meets the constraints passed
    in to the function.
    
    Id: the id of the item to be created
    lowestScore: low end of the score range
    highestScore: high end of the score range
    objectListType: the list of parameters to give to the fitness function to predict a score
    fitness: the function to determine the score
"""
def GenerateHouseInRange(Id, lowestScore, highestScore, objectListType, fitness):
    newHouse = house.house([], [])
    newHouse.randomlyFillObj()
    newHouse.normalizeData([])
    newHouse.setId(Id)
    data = np.array(newHouse.objectToList(objectListType))
    score = 0

    if objectListType == "Partial":
        score = fitness.predict(data.reshape(1, -1))

    if objectListType == "Multi":
        data = data.reshape(1, -1)
        score = fitness.predict(data)


    if objectListType == "Partial":
        while not (lowestScore < score and score < highestScore):
           newHouse.randomlyFillObj()
           newHouse.normalizeData([])
           newHouse.setId(Id)
           data = np.array(newHouse.objectToList(objectListType))
           score = fitness.predict(data.reshape(1, -1))

    if objectListType == "Multi":
        while not (lowestScore < score[0][0] and score[0][0] < highestScore):
            newHouse.randomlyFillObj()
            newHouse.normalizeData([])
            newHouse.setId(Id)
            data = np.array(newHouse.objectToList(objectListType))
            score = fitness.predict(data.reshape(1, -1))

    newHouse.setProjectedSalePrice(score)
    return newHouse

"""
    This function is responsible for creating a new generation with x number of items that
    meets the score constraints.
    
    numOfItems: number of items to put in generation
    lowestScore: low end of the score range
    highestScore: high end of the score range
    objectListType: the list of parameters to give to the fitness function to predict a score
    fitness: the function to determine the score
"""
def RandomlyGeneratePopulation(numOfItems, lowestScore, highestScore, objectListType, fitness):
    houses = []
    for num in range(numOfItems):
        newHouse = GenerateHouseInRange(num, lowestScore, highestScore, objectListType, fitness)
        houses.append(newHouse)
    
    return houses

"""
    This function is responsible for taking a newly created generation and making sure that each item
    in the generation meets a certain score constraint. If the constraints are not meet then that item
    is discared and replaced with a randomly generated item that does meet the constraints.

    generation: list of houses that were just created by genetic algorithm
    lowestScore: low end of the score range
    highestScore: high end of the score range
    objectListType: the list of parameters to give to the fitness function to predict a score
    fitness: the function to determine the score
"""
def FixOutliersInGeneration(generation, lowestScore, highestScore, objectListType, fitness):
    newGeneration = []

    if objectListType == "Partial":
        for item in generation:
            data = np.array(item.objectToList(objectListType))
            score = fitness.predict(data.reshape(1, -1))

            if not (lowestScore < score and score < highestScore):
                newHouse = GenerateHouseInRange(item.Id, lowestScore, highestScore, objectListType, fitness)
                newGeneration.append(newHouse)
            else:
                newGeneration.append(item)

        return newGeneration

    if objectListType == "Multi":
        for item in generation:
            data = np.array(item.objectToList(objectListType))
            data = data.reshape(1, -1)
            score = fitness.predict(data)

            if not (lowestScore < score[0][0] and score[0][0] < highestScore):
                newHouse = GenerateHouseInRange(item.Id, lowestScore, highestScore, objectListType, fitness)
                newGeneration.append(newHouse)
            else:
                newGeneration.append(item)

        return newGeneration

def sorterHelper(givenHouse):

    """
    To sort by price [0][0]
                sqft [0][1]
          garagesqft [0][2]
    """

    """
    To sort by multiple change comments
    """
    
    #SQFT + GARAGE SQFT
    #value = givenHouse.ProjectedSalePrice [0][1] + givenHouse.ProjectedSalePrice [0][2]
    
    #PRICE + SQFT
    value = givenHouse.ProjectedSalePrice [0][0] + givenHouse.ProjectedSalePrice [0][1]

    #PRICE
    #value = givenHouse.ProjectedSalePrice[0][1]

    return value

def sorter(generationInput, objectListType):
    sortedOutput = generationInput
    sortedOutput.sort(reverse = True, key = sorterHelper)
    return sortedOutput

"""
Prints out everything necessary for submission ish
"""
def output(numOfItems, lowestScore, highestScore, sortedGeneration, objectListType):
    itemIndex = 1;

    print("\nAfter " + str(numOfItems) + " generations of " + str(numOfItems) + " child designs these are the resulting houses.")
    print("They are ranked in order from best to worst: ")
    print("\nThe price range used was between $" + str(lowestScore) + " and $" + str(highestScore))

    for item in sortedGeneration:
        
        print("\nHouse " + str(itemIndex) + ": ")

        if objectListType == "Partial":
            print("This house has a sale price of: " + str(round(item.ProjectedSalePrice[0], 2)))

        if objectListType == "Multi":
            print("This house has a sale price of: " + str(round(item.ProjectedSalePrice[0][0], 2)))
            print("This house has a square footage of: " + str(math.trunc(item.ProjectedSalePrice[0][1])))
            print("This house has a garage square footage of: " + str(math.trunc(item.ProjectedSalePrice[0][2])))

        itemIndex+=1

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    #Min House Price
    #Range 1: 35000 - 75000
    #Range 2: 75000 - 115000
    #Range 3: 115000 - 150000
    #Range 4: 150000 - 200000
    #Range 5: 200000 - 250000
    #Range 6: 250000 - 300000
    #Range 7: 300000 - 400000
    #Range 8: 400000 - 500000
    #Range 9: 500000 - 600000
    #Range 10: 600000 - 750000
    #Max House Price

    """
    Change these variables to alter the process
    """
    
    numOfItems = 250
    lowestScore = 400000
    highestScore = 500000

    #Either "Linear" or "Multi"
    typeOfRegression = "Multi"

    """

    """

    normalizedTrainingCSV = pandas.read_csv("normalizedTrainingData.csv")

    if typeOfRegression == "Linear":
        X = normalizedTrainingCSV[house.house.partialColumnNames]
        Y = normalizedTrainingCSV["SalePrice"]
        objectListType = "Partial"
        fitness = LinearRegression()


    if typeOfRegression == "Multi":
        multiVariables = ["SalePrice", "GrLivArea", "GarageArea"]
        np.matrix(multiVariables)
        X = normalizedTrainingCSV[house.house.multiColumnNames]
        Y = normalizedTrainingCSV[multiVariables]
        print(X)
        print(Y)
        objectListType = "Multi"
        fitness = LinearRegression()


    fitness = fitness.fit(X,Y)

    #Change first parameter for number of houses
    currentGeneration = RandomlyGeneratePopulation(numOfItems, lowestScore, highestScore, objectListType, fitness)

    allOriginalGenerations = [[currentGeneration]]
    allGenerations = [[currentGeneration]]
    
    #Change for number of generations
    for generationId in range(numOfItems):
        print("Gen id: " + str(generationId))

        # creates the next generation
        nextGeneration = GeneticAlgorithm(allGenerations[len(allGenerations) - 1][0], fitness, lowestScore, highestScore, objectListType)

        # adds the unfixed generation to the allOriginalGenerations list
        allOriginalGenerations.append([nextGeneration])

        # checks that the next generation is meeting the constraints
        currentGeneration = FixOutliersInGeneration(nextGeneration, lowestScore, highestScore, objectListType, fitness)

        # adds the fixed generation to the allGenerations list
        allGenerations.append([currentGeneration])

    finalPopulation = WeightedBy(currentGeneration, fitness, objectListType)
    
    if typeOfRegression == "Linear":
        sortedGeneration = OrganizeListByFitness(currentGeneration, objectListType)

    if typeOfRegression == "Multi":
        sortedGeneration = sorter(currentGeneration, objectListType)
    
    output(numOfItems, lowestScore, highestScore, sortedGeneration, objectListType)