import pandas as pd
import time
import random
import sys

#ajoute une nouvelle rangée à un modèle avec +1 à la nouvelle observation
def addRowModel(model, obsLetters, predLetters):

    newrow = pd.DataFrame(index=[obsLetters], columns=model.columns).fillna(0)
    print(newrow.shape)
    model = model.append(newrow)
    model[predLetters][obsLetters] += 1
    return model

#retourne un df dont les sommes des rangées égals 1
def normalizeRow(dataframe):
    return dataframe.div(dataframe.sum(axis=1), axis=0).fillna(0)

#retourne une liste triés des séquences de longueur n dans le predictNext
#retourne un seul exemple de la séquence, si word est vrai, Ã§a le fait par mot
def uniqueSeqStr(txt, n, word=False):
    occurence = set()
    if (not word):
        if (n == 1) :
            occurence = set(txt)
        else :
            for i in range(len(txt)-n+1):
                occurence.add(txt[i:i+n])
    else :
        txt = txt.split()
        if (n == 1):
            occurence = set(txt)
        else :
            for i in range(len(txt)-n+1):
                occurence.add(" ".join(txt[i:i+n]))

    return sorted(list(occurence))

#retourne un dataframe avec les les occurences du groupe de letter et lettre suivante
def createMarkovModel (txt, ngram=1, word=False):

    rowIndex = uniqueSeqStr(txt, ngram, word)
    columnsIndex = uniqueSeqStr(txt, 1, word)
    occMatrix = pd.DataFrame(0, index=rowIndex, columns=columnsIndex)

    if (word) :
        txt = txt.split()

        for i in range(len(txt)-ngram):
            currentLetters = " ".join(txt[i:i+ngram])
            nextLetter = txt[i+ngram]
            occMatrix[nextLetter][currentLetters] += 1

    else :
        for i in range(len(txt)-ngram):

            currentLetters = txt[i:i+ngram]

            nextLetter = txt[i+ngram]
            occMatrix[nextLetter][currentLetters] += 1


    #print(occMatrix)
    return occMatrix

def predictNext(current, model):
    row = model.loc[current]/(model.loc[current].sum())
    row = row.cumsum()
    #print("prob of the letter after " + letter)
    #print(col)
    r = random.random()
    for prob, index in zip(row, row.index):
        if (prob > r) :
            return index

    return random.choice(row.index)
def generateTextByWords(txt, length, ngram=1):
    models = []
    setsOfLetters = []
    txtList = []

    for i in range(1, ngram + 1):
        model = createMarkovModel(txt, i, word=True)
        models.append(model)
        #liste d'ensemble des index des dataframe
        setsOfLetters.append(set(model.index))

    txtList = random.sample(setsOfLetters[ngram-1], 1)[0].split()

    for i in range(length-ngram):
        wordsFound = False
        lengthOfSeq = ngram
        #si on trouve pas la séquence de ngram, on cherche pour une seq de ngram-1 etc

        while(not wordsFound):
            currentWords = " ".join(txtList[i:i + lengthOfSeq])
            #print("Current words : [{0}]".format(currentWords))
            #s'il trouve la séquence dans le texte, prédit, sinon descend `ngram-1`
            if (currentWords in setsOfLetters[lengthOfSeq-1]):

                nextWord = predictNext(currentWords, models[lengthOfSeq-1])
                wordsFound = True
                txtList.append(nextWord)
                #print("[" + currentWords + "] found")
                #print("Predicted word : [{0}]".format(nextWord))

            else :
                #print("[" + currentWords + "] not found")
                lengthOfSeq -= 1

            #print("Generated text : ")
            #print(txtList)


    #print(len(setsOfLetters))
    return " ".join(txtList)
def generateTextByLetters(txt, length, ngram=1):
    models = []
    setsOfLetters = []
    txtList = []

    for i in range(1, ngram + 1):
        model = createMarkovModel(txt, i)
        models.append(model)
        #liste d'ensemble des index des dataframe
        setsOfLetters.append(set(model.index))

    generatedText = random.sample(setsOfLetters[ngram-1], 1)[0]

    for i in range(length-ngram):
        lettersFound = False
        lengthOfSeq = ngram
        #si on trouve pas la séquence de ngram, on cherche pour une seq de ngram-1 etc

        while(not lettersFound):

            currentLetters = generatedText[i:i + lengthOfSeq]

            #s'il trouve la séquence dans le texte, prédit, sinon descend `ngram-1`
            if (currentLetters in setsOfLetters[lengthOfSeq-1]):
                nextLetter = predictNext(currentLetters, models[lengthOfSeq-1])
                lettersFound = True
                #print("[" + currentLetters + "] found")

                generatedText += nextLetter

            else :
                #print("[" + currentLetters + "] not found")
                lengthOfSeq -= 1

    return generatedText


if __name__ == "__main__":

    if (sys.argv[0] == "help" or len(sys.argv) != 6):
        print("\nThe parameters are : \n" +
            "Filename : a filename \n" +
            "Ngram : a number \n" +
            "Word : true if you want by word, false if you want letters\n" +
            "Length : a number, the length of the output \n" +
            "toFile : true if you want the output to go to \\output\\output.txt \n"+
            "set to false otherwise")
        sys.exit()
    filename = sys.argv[1]
    ngram = int(sys.argv[2])
    word = True if (sys.argv[3] == "true") else False
    length = int(sys.argv[4])
    toFile = True if (sys.argv[5] == "true") else False


    with open("input/"+ filename, "r") as f:
        txt = f.read()


    time1 = time.time()
    if (word) :
        generatedText = generateTextByWords(txt, length, ngram)
    else :
        generatedText = generateTextByLetters(txt, length, ngram)
    time2 = time.time()
    textTime = time2-time1

    if (toFile):
        with open("output/output.txt", "w") as f:
            f.write(generatedText)

    else:
        print(generatedText)

    print("\nThe text has been generated in {:10.3f} s \n".format(textTime) +
        "The parameters are : \n"
        "Filename :{}\n".format(sys.argv[1]) +
        "Ngram : {}\n".format(sys.argv[2])  +
        "Word : {}\n".format(sys.argv[3])  +
        "Length : {}\n".format(sys.argv[4])  +
        "toFile : {}\n".format(sys.argv[5]) )
