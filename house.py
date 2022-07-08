import numpy as np
import random

#All information necessary to define a house
class house:

    #All possible descriptors of a house
    columnNames = ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition', 'SalePrice']

    #Removed data. No ID, or Price. Not useful for linear regression
    partialColumnNames = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition']

    #Removed data. No ID, Price, GrLivArea, or GarageArea
    multiColumnNames = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition']

    ridgeVariables = ["SalePrice", "GrLivArea", "GarageArea"]

    #Columns that already have integer only values
    normalizedColumnNames = ["Id", "LotFrontage", "LotArea", "OverallQual",
       "OverallCond", "YearBuilt", "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
       "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea",
       "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr",
       "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF",
       "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold", "SalePrice"]

    #Dictionary with name of attribute for key and possible values for definition
    normalizingDataTable = {
        'MSSubClass': ["0", "20", "30", "40", "45", "50", "60", "70", "75", "80", "85", "90", "120", "160", "180", "190"],
        'MSZoning': ["C (all)", "FV", "I", "RH", "RL", "RP", "RM", "NA"],
        'Street': ["Grvl", "Pave"],
        'Alley': ["Grvl", "Pave", "NA"],
        'LotShape': ["IR3", "IR2", "IR1", "Reg"],
        'LandContour': ["Lvl", "Bnk", "HLS", "Low"],
        'Utilities': ["AllPub", "NoSewr", "NoSeWa", "ELO", "NA"],
        'LotConfig': ["Inside", "Corner", "CulDSac", "FR2", "FR3"],
        'LandSlope': ["Gtl", "Mod", "Sev"],
        'Neighborhood': ["Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr", "Crawfor", "Edwards", 
        "Gilbert", "IDOTRR", "MeadowV", "Mitchel", "NAmes", "NoRidge", "NPkVill", "NridgHt", "NWAmes", "OldTown", 
        "SWISU", "Sawyer", "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker"],
        'Condition1': ["Artery", "Feedr", "Norm", "RRNn", "RRAn", "PosN", "PosA", "RRNe", "RRAe"],
        'Condition2': ["Artery", "Feedr", "Norm", "RRNn", "RRAn", "PosN", "PosA", "RRNe", "RRAe"],
        'BldgType': ["1Fam", "2fmCon", "Duplex", "Twnhs", "TwnhsE", "TwnhsI"],
        'HouseStyle': ["1Story", "1.5Fin", "1.5Unf", "2Story", "2.5Fin", "2.5Unf", "SFoyer", "SLvl"],
        'RoofStyle': ["Flat", "Gable", "Gambrel", "Hip", "Mansard", "Shed"],
        'RoofMatl': ["ClyTile", "CompShg", "Membran", "Metal", "Roll", "Tar&Grv", "WdShake", "WdShngl"],
        'Exterior1st': ["AsbShng", "AsphShn", "BrkComm", "BrkFace", "CBlock", "CemntBd", "HdBoard", "ImStucc", "MetalSd", 
        "Other", "Plywood", "PreCast", "Stone", "Stucco", "VinylSd", "Wd Sdng", "Wd Shng", "NA"],
        'Exterior2nd': ["AsbShng", "AsphShn", "BrkComm", "BrkFace", "CBlock", "CemntBd", "HdBoard", "ImStucc", "MetalSd", 
        "Other", "Plywood", "PreCast", "Stone", "Stucco", "VinylSd", "Wd Sdng", "Wd Shng", "NA"],
        'MasVnrType': ["BrkCmn", "BrkFace", "CBlock", "NA", "Stone"],
        'ExterQual': ["Ex", "Gd", "TA", "Fa", "Po", "NA"],
        'ExterCond': ["Ex", "Gd", "TA", "Fa", "Po", "NA"],
        'Foundation': ["BrkTil", "CBlock", "PConc", "Slab", "Stone", "Wood"],
        'BsmtQual': ["Ex", "Gd", "TA", "Fa", "Po", "NA"],
        'BsmtCond': ["Ex", "Gd", "TA", "Fa", "Po", "NA"],
        'BsmtExposure': ["Gd", "Av", "Mn", "No", "NA"],
        'BsmtFinType1': ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"],
        'BsmtFinType2': ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"],
        'Heating': ["Floor", "GasA", "GasW", "Grav", "OthW", "Wall"],
        'HeatingQC': ["Ex", "Gd", "TA", "Fa", "Po", "NA"],
        'CentralAir': ["N", "Y"],
        'Electrical': ["SBrkr", "FuseA", "FuseF", "FuseP", "Mix", "NA"],
        'KitchenQual': ["Ex", "Gd", "TA", "Fa", "Po", "NA"],
        'Functional': ["Typ", "Min1", "Min2", "Mod", "Maj1", "Maj2", "Sev", "Sal", "NA"],
        'FireplaceQu': ["Ex", "Gd", "TA", "Fa", "Po", "NA"],
        'GarageType': ["2Types", "Attchd", "Basment", "BuiltIn", "CarPort", "Detchd", "NA"],
        'GarageFinish': ["Fin", "RFn", "Unf", "NA"],
        'GarageQual': ["Ex", "Gd", "TA", "Fa", "Po", "NA"],
        'GarageCond': ["Ex", "Gd", "TA", "Fa", "Po", "NA"],
        'PavedDrive': ["Y", "P", "N"],
        'PoolQC': ["Ex", "Gd", "TA", "Fa", "NA"],
        'Fence': ["GdPrv", "MnPrv", "GdWo", "MnWw", "NA"],
        'MiscFeature': ["Elev", "Gar2", "Othr", "Shed", "TenC", "NA"],
        'SaleType': ["WD", "CWD", "VWD", "New", "COD", "Con", "ConLw", "ConLI", "ConLD", "Oth", "NA"],
        'SaleCondition': ["Normal", "Abnorml", "AdjLand", "Alloca", "Family", "Partial"],
        # Normalized Data  w/ lists for random data generation
        "LotFrontage": [*range(0, 351, 1)],
        "LotArea": [*range(1000, 215301, 1)],
        "OverallQual": [*range(1, 11, 1)],
        "OverallCond": [*range(1, 11, 1)],
        "YearBuilt": [*range(1860, 2013, 1)],
        "YearRemodAdd": [*range(1940, 2013, 1)],
        "MasVnrArea": [*range(0, 1701, 1)],
        "BsmtFinSF1": [*range(0, 5750, 1)],
        "BsmtFinSF2": [*range(0, 1551, 1)],
        "BsmtUnfSF": [*range(0, 2401, 1)],
        "TotalBsmtSF": [*range(0, 6251, 1)],
        "1stFlrSF": [*range(250, 4851, 1)],
        "2ndFlrSF": [*range(0, 2251, 1)],
        "LowQualFinSF": [*range(0, 651, 1)],
        "GrLivArea": [*range(250, 5751, 1)],
        "BsmtFullBath": [*range(0, 4, 1)],
        "BsmtHalfBath": [*range(0, 3, 1)],
        "FullBath": [*range(0, 4, 1)],
        "HalfBath": [*range(0, 3, 1)],
        "BedroomAbvGr": [*range(0, 9, 1)],
        "KitchenAbvGr": [*range(0, 4, 1)],
        "TotRmsAbvGrd": [*range(1, 17, 1)],
        "Fireplaces": [*range(0, 4, 1)],
        "GarageYrBlt": [0, *range(1890, 2013, 1)],
        "GarageCars": [*range(0, 7, 1)],
        "GarageArea": [*range(0, 1601, 1)],
        "WoodDeckSF": [*range(0, 951, 1)],
        "OpenPorchSF": [*range(0, 651, 1)],
        "EnclosedPorch": [*range(0, 651, 1)],
        "3SsnPorch": [*range(0, 651, 1)],
        "ScreenPorch": [*range(0, 651, 1)],
        "PoolArea": [*range(0, 851, 1)],
        "MiscVal": [*range(0, 16001, 1)],
        "MoSold": [*range(1, 13, 1)],
        "YrSold": [*range(2003, 2013, 1)]
        }

    CSVRow = []
    excludedColumns = []
    Id = -1
    SalePrice = -1
    ProjectedSalePrice = -1
    

    """
    def __init__(self, Id):
        getattr(self, "Id", Id)
        setattr(self, "Id", Id)
        self.randomlyFillObj()
    """

    #Initializes house object
    def __init__(self, CSVRow, excludedColumns):
        self.CSVRow = CSVRow
        self.excludedColumns = excludedColumns

    #Assigns Id to house object
    def setId(self, Id):
        self.Id = Id

    #Assigns SalePrice to house object
    def setSalePrice(self, SalePrice):
        self.SalePrice = SalePrice

    #Assigns ProjectedSalePrice to house object
    def setProjectedSalePrice(self, ProjectedSalePrice):
        self.ProjectedSalePrice = ProjectedSalePrice

    #Loads up all relevant info for a house
    def loadClass(self, CSVRow, excludedColumns):
        for name in self.columnNames:
            if name not in excludedColumns:
                getattr(self, name, CSVRow[name].values[0])
                setattr(self, name, CSVRow[name].values[0])

            if (name == "Id"):
                print("Working on house with id: " + getattr(self, name))

    #Normalized typos
    def normalizeData(self, excludedColumns):
        for name in self.columnNames:
            #if name in excludedColumns:
                #print("Skipped column with name: " + name)
            if name not in self.normalizedColumnNames and name not in excludedColumns:
                nameValue = getattr(self, name)
                #print("Normalizing " + name + ", prevValue = " + str(nameValue))
                # Fixing mistakes that were made in the given dataset by explicitly fixing spelling inconsistencies between data and data description
                try:
                    # Checking if the nameValue is NaN
                    if (nameValue != nameValue):
                        setattr(self, name, self.normalizingDataTable[name].index("NA"))
                    elif (nameValue == "None"):
                        setattr(self, name, self.normalizingDataTable[name].index("NA"))
                    elif (nameValue == "Brk Cmn"):
                        setattr(self, name, self.normalizingDataTable[name].index("BrkComm"))
                    elif (nameValue == "CmentBd"):
                        setattr(self, name, self.normalizingDataTable[name].index("CemntBd"))
                    elif (nameValue == "WdShing"):
                        setattr(self, name, self.normalizingDataTable[name].index("Wd Shng"))
                    else:
                        setattr(self, name, self.normalizingDataTable[name].index(getattr(self, name)))
                except:
                    print("Error Normalizing " + name + ", prevValue = " + str(nameValue))
            elif name in self.normalizedColumnNames and name not in excludedColumns:
                nameValue = getattr(self, name)
                try:
                    if (nameValue != nameValue):
                        setattr(self, name, 0)
                except:
                    print("Error Normalizing " + name + ", prevValue = " + str(nameValue))
                
    #Prints every attribute
    def print(self):
        for name in self.columnNames:
            print(str(name) + ": " + str(getattr(self, name)))

    #Converts from object to list
    def objectToList(self, listType):
        resultList = []
        if listType == "Full":
            for name in self.columnNames:
                resultList.append(getattr(self, name))
        elif listType == "SemiPartial":
            for name in self.columnNames:
                if not (name == "SalePrice"):
                    resultList.append(getattr(self, name))
        elif listType == "Partial":
            for name in self.columnNames:
                if not (name == "Id" or name == "SalePrice"):
                    resultList.append(getattr(self, name))
        elif listType == "Multi":
            for name in self.columnNames:
                if not (name == "Id" or name == "SalePrice" or name == "GrLivArea" or name == "GarageArea"):
                    resultList.append(getattr(self, name))

        return resultList

    #Converts from list to objec
    def listToObject(self, objList, listType):
        propIndex = 0
        propName = self.columnNames[propIndex]
        for objProp in objList:
            if listType == "Full":
                getattr(self, propName, objProp)
                setattr(self, propName, objProp)
            elif listType == "SemiPartial":
                if not (propName == "SalePrice"):
                    getattr(self, propName, objProp)
                    setattr(self, propName, objProp)
            elif listType == "Partial":
                if not (propName == "Id" or propName == "SalePrice"):
                    getattr(self, propName, objProp)
                    setattr(self, propName, objProp)

            propIndex += 1
            if not (len(self.columnNames) - 1 < propIndex):
                propName = self.columnNames[propIndex]
            
    #Puts all variables into a list
    def classVariableNamesToList(self, listType):
        if listType == "Full":
            return self.columnNames
        elif listType == "Partial":
            return self.partialColumnNames

    #Randomly generate a house
    def randomlyFillObj(self):
        for var in self.partialColumnNames:
            choices = self.normalizingDataTable[var]
            choice = random.choice(choices)
            getattr(self, var, choice)
            setattr(self, var, choice)

    #Randomly selects and modifies attribute
    def randomMutation(self):
        propToMutate = random.choice(self.partialColumnNames)
        newValueFromCol = random.choice(self.normalizingDataTable[propToMutate])
        newValue = self.normalizingDataTable[propToMutate].index(newValueFromCol)
        setattr(self, propToMutate, newValue)