import csv

class DataReader:

    reader : csv.DictReader
    houses : int
    currentRow = []

    def __init__(self, file_name):
        self.file = open(file_name, "r")
        self.reader = csv.DictReader(self.file)
        # Every house has 2 columns: a load and a PV value
        self.houses = (len(self.reader.fieldnames) - 1) // 2

    # Returns an array of [houses] arrays, each containing [amount] load values
    def readAllLoads(self, amount):
        ret = [[] for i in range(self.houses)]

        for i, row in enumerate(self.reader):
            if i == amount: break
            
            for house in range(self.houses):
                ret[house].append(int(row[f'load_house_{house}_W']))
        
        return ret

    # Currently unused
    def getNextLoads(self):
        row = next(self.reader)
        self.currentRow = [
            row[f'load_house_{i}_W'] for i in range(self.houses)
        ]
        return self.currentRow
    
    # Just an abstraction function
    def getLoad(self, house):
        return self.currentRow[house]