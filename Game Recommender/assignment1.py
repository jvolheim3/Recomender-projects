# %% [markdown]
# # Assignment 1  
# 
# ### Import data

# %%
import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy as np
import numpy
import string
import random
import string
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# %%
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

# %%
def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

# %%
def readJSON(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        d = eval(l)
        u = d['userID']
        g = d['gameID']
        yield u,g,d

# %%
# Some data structures that will be useful

# %%
allHours = []
for l in readJSON("./Data/train.json.gz"):
    allHours.append(l)

# %% [markdown]
# ### Partition data 

# %%
hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]
hoursValid[0]

# %% [markdown]
# ### Part 1

# %%
# Any other preprocessing...
gamesPerUser = {}
UserPerGame = {}
for u,g,d in hoursTrain:
    if u not in gamesPerUser:
        gamesPerUser[u] = [g]
    else:
        gamesPerUser[u].append(g)
    if g not in UserPerGame:
        UserPerGame[g] = [u]
    else:
        UserPerGame[g].append(u)

# %% [markdown]
# ### Adding negative pairs

# %%
def ranGame(user):
    '''random game user hasnt played'''
    ran = random.choice(list(UserPerGame.keys()))
    if user not in gamesPerUser:
        return ran

    if len(gamesPerUser[user]) == len(UserPerGame):
        print('all games played')
        return ran
    
    while ran in gamesPerUser[user]:
        ran = random.choice(list(UserPerGame.keys()))
    return ran
    
newHoursValid = []
for u,g,d in hoursValid:
    newHoursValid.append((u,g,1))
    newHoursValid.append((u,ranGame(u),0))
newHoursTrain = []
for u,g,d in hoursTrain:
    newHoursTrain.append((u,g,1))
    newHoursTrain.append((u,ranGame(u),0))


# %% [markdown]
# ### Compute most popular

# %%

gameCount = defaultdict(int)
totalPlayed = 0

for user,game,_ in hoursTrain:
  gameCount[game] += 1
  totalPlayed += 1

mostPopular = [(gameCount[x], x) for x in gameCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
  count += ic
  return1.add(i)
  if count > totalPlayed * 0.686868686868687: break


# %% [markdown]
# ### similarity and training functions

# %%
def Jaccard(s1, s2):
    # ...
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


def similar(u,g,s=1):
    siml = []
    for i in gamesPerUser[u]:
        if i == g:
            continue
        if g in return1:
            siml.append(Jaccard(set(UserPerGame[g]), set(UserPerGame[i]))*s)
        else:
            siml.append(Jaccard(set(UserPerGame[g]), set(UserPerGame[i])))
    if len(siml) == 0:
        return 0
    
    siml.sort(reverse=True)

    if len(siml)//2 >1:
        siml = siml[:len(siml)//2]
    else:
        return siml[0]

    simAvg = sum(siml)/len(siml)
    return simAvg


def getTrainSim(s=1):
    sim3 = []

    for u,g,_ in newHoursTrain:
        if u not in gamesPerUser:
            sim3.append(0)
        else:
            sim3.append(similar(u,g,s))
    return sim3

def getValidSim(s=1):
    sim3 = []

    for u,g,_ in newHoursValid:
        if u not in gamesPerUser:
            sim3.append(0)
        else:
            sim3.append(similar(u,g,s))
    return sim3

def trainModel(thres_start=0.009, thres_end=0.04, N_Tstep=500, s_start=1, s_end=1.1, N_Sstep=3):
    sim1 = getTrainSim()
    print('training model: Thresholds')
    
    best_threshold = 0
    best_acc = 0
    y_actual = [d for _,_,d in newHoursTrain]
    for thres in np.linspace(thres_start, thres_end, N_Tstep):
        y_pred = np.array(sim1) > thres
        acc = sum(y_pred == y_actual)/len(y_actual)
        if acc > best_acc:
            best_acc = acc
            best_threshold = thres
    
    print('best threshold: ', best_threshold, 'Acc: ',best_acc)

    return best_threshold

def validateModel(thres):
    sim1 = getValidSim()
    y_actual = [d for _,_,d in newHoursValid]
    y_pred = np.array(sim1) > thres
    acc = sum(y_pred == y_actual)/len(y_actual)
    print('Accuracy: ', acc)
    return acc


def predict(u,g,s=1,thres=0.001):
    if u not in gamesPerUser:
        return 0
    sim = similar(u,g,s)
    if sim > thres:
        return 1
    return 0

# %% [markdown]
# ### train threshold 

# %%
best_threshold =trainModel()
print( 'best_threshold: ', best_threshold)
print(validateModel(best_threshold))


# %% [markdown]
# ### Logistic regression

# %%
def feature(u,g):
    feat= [1,predict(u,g,1,0.0173)]
    feat.append(gameCount[g]/totalPlayed if g in return1 else 0)
    return feat


X = [feature(u,g) for u,g,_ in newHoursTrain]
y = [d for _,_,d in newHoursTrain]

clf = linear_model.LogisticRegression(class_weight='balanced')


# %%
print('Logistic Regression Model:')
clf.fit(X, y)
print('Logistic Regression Accuracy: ', clf.score(X,y))

# %%
X_valid = [feature(predict(u,g,1,0.0173),gameCount[g]/totalPlayed if g in return1 else 0) for u,g,_ in newHoursValid]

Y_valid_actual = [d for _,_,d in newHoursValid]
print('Logistic Regression Accuracy on Validation Set: ', clf.score(X_valid,Y_valid_actual))

# %% [markdown]
# ### Solution test

# %%
predictions = open("predictions_Played.csv", 'w')
pred = []
for l in open("./Data/pairs_Played.csv"):
    if l.startswith("userID"):
        
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    pred = clf.predict([feature(predict(u,g,1,0.0173),gameCount[g]/totalPlayed if g in gameCount else 0)])[0]
    _ = predictions.write(u + ',' + g + ',' + str(pred) + '\n')

predictions.close()

# %% [markdown]
# # Part 2: hours played prediction 
# ### Preprocessing 
# 

# %%
userIDs = {}
itemIDs = {}
interactions = []
for u,g,d in allHours:
    if u not in userIDs:
        userIDs[u] = len(userIDs)
    if g not in itemIDs:
        itemIDs[g] = len(itemIDs)
    
    interactions.append((u,g,d['hours_transformed']))


# %%
nTrain = int(len(interactions) * 0.75)
interactionsTrain = interactions[:nTrain]
interactionsTest = interactions[nTrain:]

trainHours = [r for u,i,r in interactionsTrain]
globalAverage = sum(trainHours) * 1.0 / len(trainHours)

# %%
itemsPerUser = defaultdict(list)
usersPerItem = defaultdict(list)
for u,i,r in interactionsTrain:
    itemsPerUser[u].append(i)
    usersPerItem[i].append(u)

# %%

hoursPerUser = defaultdict(set)
hoursPerItem = defaultdict(set)

for u,i,d in interactionsTrain:
    hours = d
    hoursPerItem[i].add((u, hours))
    hoursPerUser[u].add((i, hours))


# %% [markdown]
# ### Functions

# %%
def iterate(lam,lr):
    global alpha
    global betaU
    global betaI
    lalpha = alpha
    lbetaU = betaU.copy()
    lbetaI = betaI.copy()

    a = sum(r - (betaU[u] + betaI[i]) for u,i,r in interactionsTrain)
    b = a / len(interactionsTrain)
    alpha = lalpha + lr * (b - lalpha)

    for u in hoursPerUser:
        sm = sum(i[1] - (alpha + betaI[i[0]]) for i in hoursPerUser[u])
        sm = sm / (lam + len(hoursPerUser[u]))
        betaU[u] = lbetaU[u] + lr * (sm - lbetaU[u])

    for u in hoursPerItem:
        sm = sum(i[1] - (alpha + betaU[i[0]]) for i in hoursPerItem[u])
        sm = sm / (lam + len(hoursPerItem[u]))
        betaI[u] = lbetaI[u] + lr * (sm - lbetaI[u])
        
    mse = 0
    for u,g,r in interactionsTrain:
        prediction = alpha + betaU[u] + betaI[g]
        mse += (r - prediction)**2
        
    regularizer = 0
    
    for u in betaU:
        regularizer += betaU[u]**2
    for g in betaI:
        regularizer += betaI[g]**2

    mse /= len(interactionsTrain)
    return mse ,mse + lam*regularizer

def iterate2(lami,lamu,lr):
    global alpha
    global betaU
    global betaI
    lalpha = alpha
    lbetaU = betaU.copy()
    lbetaI = betaI.copy()

    a = sum(r - (betaU[u] + betaI[i]) for u,i,r in interactionsTrain)
    b = a / len(interactionsTrain)
    alpha = lalpha + lr * (b - lalpha)

    for u in hoursPerUser:
        sm = sum(i[1] - (alpha + betaI[i[0]]) for i in hoursPerUser[u])
        sm = sm / (lamu + len(hoursPerUser[u]))
        betaU[u] = lbetaU[u] + lr * (sm - lbetaU[u])

    for u in hoursPerItem:
        sm = sum(i[1] - (alpha + betaU[i[0]]) for i in hoursPerItem[u])
        sm = sm / (lami + len(hoursPerItem[u]))
        betaI[u] = lbetaI[u] + lr * (sm - lbetaI[u])
        
    mse = 0
    for u,g,r in interactionsTrain:
        prediction = alpha + betaU[u] + betaI[g]
        mse += (r - prediction)**2
        
    regularizeru = 0
    regularizeri = 0
    for u in betaU:
        regularizeru += betaU[u]**2
    for g in betaI:
        regularizeri += betaI[g]**2

    mse /= len(interactionsTrain)
    return mse ,mse + lamu*regularizeru + lami*regularizeri

def predict(u,g):
    global alpha
    global betaU
    global betaI
    
    bu = 0
    bi = 0
    if u in betaU:
        bu = betaU[u]
    if g in betaI:
        bi = betaI[g]
    return alpha + bu + bi

def mseValidate():
    mse = 0
    for u,g,r in interactionsTest:
        prediction = predict(u,g)
        mse += (r - prediction)**2
    mse /= len(interactionsTest)
    return mse

# %% [markdown]
# ### Find lambda combinations 

# %%
pairs = []
for i in range(10):
    for j in range(10):
        print('lamda: ', i, 'lamu: ', j)
        betaU = {}
        betaI = {}
        for u in hoursPerUser:
            betaU[u] = 0

        for g in hoursPerItem:
            betaI[g] = 0
        alpha = globalAverage 

        mse,objective = (100,100)
        newMSE,newObjective = iterate2(i,j,.78)
        itera = 0

        while itera < 10 or objective - newObjective > 0.01:
            mse, objective = newMSE, newObjective
            newMSE, newObjective = iterate2(i,j,0.78)
            itera += 1
            print("MSE after "
                + str(itera) + " iterations = " + str(newMSE))
            if itera == 100:
                break
        msev = mseValidate()
        pairs.append((msev,i,j))
        print("MSE on test data = " + str(msev), 'Mse on train data: ', newMSE)

# %%
# sort pairs by mse
pairs.sort()

print(pairs[:10])

# %% [markdown]
# ### Train

# %%
betaU = {}
betaI = {}
for u in hoursPerUser:
    betaU[u] = 0

for g in hoursPerItem:
    betaI[g] = 0
alpha = globalAverage 

mse,objective = (100,100)
newMSE,newObjective = iterate2(1,9,.78)
itera = 0

while itera < 10 or objective - newObjective > 0.01:
    mse, objective = newMSE, newObjective
    newMSE, newObjective = iterate2(1,9,0.78)
    itera += 1
    print("MSE after "
        + str(itera) + " iterations = " + str(mse))
    mse = mseValidate()
    print("MSE on test data = " + str(mse), 'Mse on train data: ', newMSE)
    if itera == 100:
        break
   


# %% [markdown]
# ### Data test output 

# %%
predictions = open("predictions_Hours.csv", 'w')
for l in open("./Data/pairs_Hours.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    if u in betaU and g in betaI:
        pred = predict(u,g)
    elif u in betaU and g not in betaI:
        pred = alpha + betaU[u] +np.mean([b for b in betaI.values()])
    elif u not in betaU and g in betaI:
        pred = alpha + betaI[g] + np.mean([b for b in betaU.values()])
    else:
        pred = globalAverage
   
    _ = predictions.write(u + ',' + g + ',' + str(pred) + '\n')

predictions.close()


