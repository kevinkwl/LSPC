"""
utilities: 

"""



def plotroc(true_label, pval):
    """
    
    :param true_label: 
    :param pval: 
    :return: 
    """
    pass


def getData(posXs, negXs, posTag=1, negTag=-1):
    Nmin = len(negXs)
    Nmax = len(posXs)

    Xdata = []
    Ydata = []
    for i in range(Nmax):
        posX = posXs[i]
        Xdata.append([])
        Ydata.append([])
        for j in range(Nmin):
            negX = negXs[j]
            Xdata[i].append(posX + negX)
            Ydata[i].append([posTag] * len(posX) + [negTag] * len(negX))
    return Xdata, Ydata