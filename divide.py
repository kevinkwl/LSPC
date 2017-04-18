
def checknew(i,tag,a1):
    if len(a1)==0:
        return 0
    if tag[i][:3] != a1[-1][:3]:
        return 1
    return 0


def addAns(ans,ansx,a,ax):  # checked
    ans.append([])
    ansx.append([])
    l1=len(ans)
    for i in range(len(a)):
        ans[l1-1].append(a[i])
        ansx[l1-1].append(ax[i])

def merge(a1,a1x,a2,a2x): #checked
    for j in range(len(a1)):
        a2.append(a1[j])
        a2x.append(a1x[j])

def throw_remain(a1,a1x,a2,a2x,ans,ansx,k):

    if (len(a2) >= k):
        del a1[:]
        del a1x[:]
        for i in range(k,len(a2)):
            a1.append(a2[i])
            a1x.append(a2x[i])

        del a2[k:]
        del a2x[k:]
        #a1 = a2[k:]
        #a1x = a2x[k:]
        #a2 = a2[:k]
        #a2x = a2x[:k]
        addAns(ans, ansx, a2, a2x)

        del a2[:]
        del a2x[:]
        for i in range(len(a1)):
            a2.append(a1[i])
            a2x.append(a1x[i])
    del a1[:]
    del a1x[:]

def add(i,tag,feature,a1,a1x,a2,a2x,ans,ansx,k):
    if (len(a1) == k):
        addAns(ans, ansx, a1, a1x)
        del a1[:]
        del a1x[:]
    if (checknew(i,tag,a1)):
        if (len(a1)<k):
            merge(a1,a1x,a2,a2x)
            throw_remain(a1,a1x,a2,a2x,ans,ansx,k)
    a1.append(tag[i])
    a1x.append(feature[i])

def divide(tag,feature,size_part,sort_tag=2):
    """
    
    :param tag: a list of patent tags 
    :param feature: a list of feature values
    :param size_part: decomposition task size (data points)
    :param sort_tag: 0 for class-decomposition, 2 for random-decomposition, 1 for nothing
    :return: (posXs, negXs), posXs is a list of partitioned positive class training data
    """


    # where size_n stands for how many data,tag mean where it belongs to like A01G/9/02 and 
    # features contains all the inner data(which is very long)
    size_n=len(tag)                                     # size_part means how many data it contains in a small SVM
    if (sort_tag==0):
        zipped = zip(tag, feature)
        tag, feature = zip(*sorted(zipped, key=lambda x: x[0]))
    if (sort_tag==2):
        zipped = list(zip(tag, feature))
        import numpy as np
        np.random.shuffle(zipped)
        tag, feature = list(zip(*zipped))

    pos1 = []  # This store the integ positive one , if some rest, throw to pos2
    pos1x = []
    pos2 = []
    pos2x = []
    neg1 = []  # like neg1
    neg1x = []
    neg2 = []
    neg2x = []
    anspos = []
    ansposx = []
    ansneg = []
    ansnegx = []
    for i in range(size_n):
        if tag[i][0] == 'A':
            add(i,tag,feature,pos1,pos1x,pos2,pos2x,anspos,ansposx,size_part)
        else:
            add(i,tag,feature,neg1,neg1x,neg2,neg2x,ansneg,ansnegx,size_part)
    merge(pos1,pos1x,pos2,pos2x)
    throw_remain(pos1,pos1x,pos2,pos2x, anspos, ansposx, size_part)
    addAns(anspos, ansposx, pos2,pos2x)

    merge(neg1, neg1x, neg2, neg2x)
    throw_remain(neg1, neg1x, neg2, neg2x, ansneg, ansnegx, size_part)
    addAns(ansneg, ansnegx, neg2, neg2x)
    return ansposx,ansnegx

