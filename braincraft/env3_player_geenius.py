# Braincraft challenge — 1000 neurons, 100 seconds, 10 runs, 2 choices, no reward
# Copyright (C) 2025 Nicolas P. Rougier
# Released under the GNU General Public License 3
"""
Example and evaluation of the performances of a random player.
"""
from bot import Bot
from environment_3 import Environment
import time
import random

def identity(x):
    return x

def getSeed():
    s = time.time()
    s = round(1000000*(s - int(s)))
    return s

bot = Bot()

# Fixed parameters
n = 1000
p = bot.camera.resolution

def geenius_trainer():
    """Train a geenius"""
    s = getSeed()
    np.random.seed(s)

    warmup = 0
    f = np.tanh
    g = np.tanh
    leak = 0.85

    # Random search for best model for 5 tries  (not efficient at all)
    model_best = None
    score_best = -1

    numGenerations = 10
    populationSize = 20
    numSurvivors = 5

    survivors = []
    for gen in range(numGenerations):
        print('generation', gen)
        generation = []
        for i in range(populationSize):
            # Evaluate model on 3 runs
            if len(survivors) == 0:
                Win, W, Wout = initialize(numNeurons=3)
                model = Win, W, Wout, warmup, leak, f, g
            else:
                parent = random.choice(survivors)
                model = mutate(copyModel(parent))

            score_mean, score_std = evaluate(model, Bot, Environment, runs=1, debug=False)
            generation.append((score_mean, model))

            if score_mean > score_best:
                score_best = score_mean
                model_best = Win, W, Wout, warmup, leak, f, g
                print('new best score:', score_best)
                # Return temporary best result
                yield model

        # Sort by mean score, descending
        generation = sorted(generation, reverse=True)
        survivors = [model for score, model in generation[:numSurvivors]]

def mutateNonzeroMatrix(matrix, changeFactors):
    numChanges = len(changeFactors)
    nonzeros = matrix[matrix != 0]
    nonzeros[np.random.randint(len(nonzeros), size=numChanges)] *= changeFactors
    matrix[matrix != 0] = nonzeros

def mutate(model, changeRate=0.2, maxChangePowerOf2=1):
    Win, W, Wout, warmup, leak, f, g = model

    for matrix in [Win, W, Wout]:
        numChanges = max(1, round(np.count_nonzero(matrix)*changeRate))
        changeFactors = 2**np.random.uniform(-maxChangePowerOf2, maxChangePowerOf2, size=numChanges)
        mutateNonzeroMatrix(matrix, changeFactors)

    mutatedModel = Win, W, Wout, warmup, leak, f, g
    return mutatedModel

def copyModel(model):
    Win, W, Wout, warmup, leak, f, g = model
    newModel = Win.copy(), W.copy(), Wout.copy(), warmup, leak, f, g
    return newModel

def initialize(numNeurons=5):
    amplitude = 2

    # Initialize zeroed weight matrix
    Win  = np.zeros((n, 2*p+3))
    # Randomize only left, middle and right camera input matrix, plus energy, hit, and bias inputs
    # inputWeightIdx = [0, p//2, p-1, p, p*3//2, 2*p-1, 2*p, 2*p+1, 2*p+2]
    inputWeightIdx = [0, p-1, p, 2*p-1, 2*p]
    Win[:numNeurons, inputWeightIdx] = amplitude*np.random.uniform(-1, 1, [numNeurons, len(inputWeightIdx)]);

    # Initialize zeroed internal weight matrix
    W = np.zeros((n, n))
    # Randomize only numNeuron interal weights
    W[:numNeurons, :numNeurons] = amplitude*np.random.uniform(-1, 1, [numNeurons, numNeurons])

    # Initialize zeroed output weights
    Wout = np.zeros((1,n))
    Wout[:, :numNeurons] = amplitude*np.random.uniform(-1, 1, (1,numNeurons))
    return Win, W, Wout

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    import numpy as np
    from challenge_2 import train, evaluate

    seed = getSeed()
    np.random.seed(seed)

    # Training (100 seconds)
    print(f"Starting training for 100 seconds (user time)")
    model = train(geenius_trainer, timeout=100)

    # Evaluation
    start_time = time.time()
    score, std = evaluate(model, Bot, Environment, debug=True, seed=seed, progress=True)
    elapsed = time.time() - start_time
    print(f"Evaluation completed after {elapsed:.2f} seconds")
    print(f"Final score: {score:.2f} ± {std:.2f}")
