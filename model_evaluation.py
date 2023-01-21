import pandas as pd
import collections
import numpy as np
import matplotlib.pyplot as plt
from dvrl_tiger import setupExperiment, runOneTimeStep, trackValues

config = {"actionEncodeDim": 64,
          "observationEncodeDim": 128,
          "hiddenDim": 64,
          "hDim": 256,  # =zDim for VAE
          "zDim": 256,
          "numParticles": 15,
          "numStepBeforeTrain": 4,
          "totalTrainStep": 10000,
          "actionDim": 3,  # fixed
          "observationDim": 1 #fixed
          }

numParticles = [10, 15, 20]
hiddenDim = [64, 128]
Dim = 256
actionEncodeDim = 64
observationEncodeDim = 128
numStepBeforeTrain = [2, 3, 4, 5]

def run_model(actionEncodeDim, observationEncodeDim, Dim, numParticles, hiddenDim, numStepBeforeTrain):
    config['numParticles'] = numParticles
    config['hiddenDim'] = hiddenDim
    config['hDim'] = config['zDim'] = Dim
    config['actionEncodeDim'] = actionEncodeDim
    config['observationEncodeDim'] = observationEncodeDim
    config['numStepBeforeTrain'] = numStepBeforeTrain

    env, actorCritic, rollouts, currentMemory = setupExperiment(config)
    cumulativeReward = 0
    episode_step = 0
    movingAverageReward = []
    scores = collections.deque(maxlen=50)
    for j in range(config["totalTrainStep"]):
        trackedValues = collections.defaultdict(lambda: [])
        for step in range(config["numStepBeforeTrain"]):
            policyReturn, currentMemory, masks, reward, cumulativeReward, episode_step, done = runOneTimeStep(config,
                                                                                                              actorCritic,
                                                                                                              currentMemory,
                                                                                                              env,
                                                                                                              cumulativeReward,
                                                                                                              episode_step)
            if done:
                scores.append(cumulativeReward)
                movingAverageReward.append(np.mean(scores))
                cumulativeReward = 0
            rollouts.insert(step, reward, masks)
            trackedValues = trackValues(trackedValues, policyReturn)
        actorCritic.learn(rollouts, trackedValues, currentMemory)
    return movingAverageReward


def drawLinePlot(val, numParticles, numStepBeforeTrain):
    for i in hiddenDim:
        plt.plot(val[(numParticles, i, numStepBeforeTrain)], label='hiddenDim:{}'.format(i))
        plt.legend()

def main():
    levelValues = [numParticles, hiddenDim, numStepBeforeTrain]
    levelNames = ['numParticles', 'hiddenDim', 'numStepBeforeTrain']
    ind = pd.MultiIndex.from_product(levelValues, names=levelNames)
    y = {}
    # data = pd.DataFrame(index=ind, columns=['scores'])
    for i in ind:
        y[i] = run_model(actionEncodeDim, observationEncodeDim, Dim, *i)
        # data.loc[i] = [y[i]]

    fig = plt.figure(figsize=(30, 25))
    plotRowNum = len(numParticles)
    plotColNum = len(numStepBeforeTrain)
    plotCounter = 1

    for i in numParticles:
        for j in numStepBeforeTrain:
            ax = fig.add_subplot(plotRowNum, plotColNum, plotCounter)
            drawLinePlot(y, i, j)
            plotCounter += 1
            plt.title('StepBeforeTrain = ' + str(j))
            plt.ylabel('numParticles ' + str(i))

    plt.suptitle('Model Evaluation for DVRL_tiger')
    plt.show()

main()