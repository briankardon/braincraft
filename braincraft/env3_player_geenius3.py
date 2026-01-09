# Braincraft challenge — 1000 neurons, 100 seconds, 10 runs, 2 choices, no reward
# Copyright (C) 2025 Nicolas P. Rougier
# Released under the GNU General Public License 3
"""
Example and evaluation of the performances of a random player.
"""
from bot import Bot
from environment_3 import Environment
import time, traceback

class QuickBot(Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.energy = 0.5

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

class Model:
    def __init__(self, randomize=False):
        self.Win = None
        self.W = None
        self.Wout = None
        self.warmup = None
        self.leak = None
        self.f = None
        self.g = None
        self.score = None
        self.fitness = None
        if randomize:
            self.randomize()
    def setComponents(self, Win=None, W=None, Wout=None, warmup=None, leak=None, f=None, g=None):
        self.Win = Win
        self.W = W
        self.Wout = Wout
        self.warmup = warmup
        self.leak = leak
        self.f = f
        self.g = g
    def setFitness(self, fitness):
        self.fitness = fitness
    def getComponents(self):
        return self.Win.copy(), self.W.copy(), self.Wout.copy(), self.warmup, self.leak, self.f, self.g
    def copy(self):
        copy = Model()
        copy.setComponents(*self.getComponents())
        return copy
    def mutate(self, changeRate=0.2, maxChange=1):
        copy = self.copy()
        for matrix in [copy.Win, copy.W, copy.Wout]:
            numNonZero = np.count_nonzero(matrix)
            numChanges = max(1, round(numNonZero*changeRate))
            # print('\nnum changes', numChanges, 'of', numNonZero)
            changeFactors = np.random.uniform(-maxChange, maxChange, size=numChanges)
            mutateNonzeroMatrix(matrix, changeFactors)

        copy.leak += np.random.uniform(-0.02, 0.02)
        copy.leak = np.clip(copy.leak, 0, 1)
        copy.warmup = 0 #np.clip(copy.warmup + np.random.randint(-2, 2), 0, 20)
        return copy
    def mate(self, other):
        # Random "gene" sharing
        Win = crossoverNonzeroMatrices(self.Win, other.Win)
        W = crossoverNonzeroMatrices(self.W, other.W)
        Wout = crossoverNonzeroMatrices(self.Wout, other.Wout)
        leak = (self.leak + other.leak) / 2
        warmup = round((self.warmup + other.warmup) / 2)
        f = self.f
        g = self.g

        child = Model()
        child.setComponents(Win=Win, W=W, Wout=Wout, leak=leak, warmup=warmup, f=f, g=g)
        return child
    def test(self, runs=1):
        score_mean, score_std = evaluate(self.getComponents(), QuickBot, Environment, runs=runs, debug=False)
        self.score = score_mean
    def __str__(self):
        with np.printoptions(precision=2, floatmode='fixed', suppress=True):
            output = f"Win: {self.Win[self.Win != 0]}\nW:   {self.W[self.W != 0]}\nWout: {self.Wout[self.Wout != 0]}"
        return output
    def randomize(self):
        # self.warmup = np.random.randint(0, 10)
        self.warmup = 0 #np.random.randint(0, 10)
        self.f = np.tanh #lambda x: np.maximum(0, x)
        self.g = np.tanh
        self.leak = 0.85

        amplitude = 1

        numNeuronsPerSide = 2
        numSideNeurons = 2*numNeuronsPerSide
        leftNeuronsIdx = list(range(0, numNeuronsPerSide))
        rightNeuronsIdx = list(range(numNeuronsPerSide, numSideNeurons))
        leftCameraInputs = list(range(0, numNeuronsPerSide))
        rightCameraInputs = list(range(p-numNeuronsPerSide, p))

        ### INPUT WEIGHTS
        # Initialize zeroed weight matrix
        self.Win  = np.zeros((n, 2*p+3))
        # Leftmost distance input to left neurons
        self.Win[leftNeuronsIdx, leftCameraInputs] = 0.1 #amplitude*np.random.uniform(-1, 1, size=numNeuronsPerSide)
        # Rightmost distance input to neuron 1
        self.Win[rightNeuronsIdx, rightCameraInputs] = 0.1 #amplitude*np.random.uniform(-1, 1, size=numNeuronsPerSide)
        # Add hit, energy, and bias connections
        # self.Win[:numSideNeurons, 2*p:2*p+3] = amplitude*np.random.uniform(-1, 1, size=[numSideNeurons, 3])

        ### INTERNAL WEIGHTS
        # Randomize internal connections
        self.W = np.zeros((n, n))
        # # Interconnect left neurons
        self.W[0:numNeuronsPerSide, 0:numNeuronsPerSide] = amplitude*np.random.uniform(-1, 1, size=[numNeuronsPerSide, numNeuronsPerSide])
        # # Interconnect right neurons
        self.W[numNeuronsPerSide:numSideNeurons, numNeuronsPerSide:numSideNeurons] = amplitude*np.random.uniform(-1, 1, size=[numNeuronsPerSide, numNeuronsPerSide])
        # Sparse connect left to right
        # self.W[leftNeuronsIdx, rightNeuronsIdx] = amplitude*np.random.uniform(-1, 1, size=[1, numNeuronsPerSide])
        # Sparse connect right to left
        # self.W[rightNeuronsIdx, leftNeuronsIdx] = amplitude*np.random.uniform(-1, 1, size=[1, numNeuronsPerSide])

        # Dense connect left and right
        # self.W[0:numNeuronsPerSide, numNeuronsPerSide:numSideNeurons] = amplitude*np.random.uniform(-1, 1, size=[1, numNeuronsPerSide])
        # Sparse connect right to left
        # self.W[numNeuronsPerSide:numSideNeurons, 0:numNeuronsPerSide] = amplitude*np.random.uniform(-1, 1, size=[1, numNeuronsPerSide])

        # Connect side neurons to processor neurons
        # self.W[:numSideNeurons, numSideNeurons:(numSideNeurons+numProcessors)] = amplitude*np.random.uniform(-1, 1, size=[numSideNeurons, numProcessors])
        # Interconnect processors
        # self.W[numSideNeurons:(numSideNeurons+numProcessors), numSideNeurons:(numSideNeurons+numProcessors)] = amplitude*np.random.uniform(-1, 1, size=[numProcessors, numProcessors])

        ### OUTPUT WEIGHTS
        # Initialize zeroed output weights
        self.Wout = np.zeros((1,n))
        self.Wout[:, :numSideNeurons] = amplitude*np.random.uniform(-1, 1, size=numSideNeurons)

def mutateNonzeroMatrix(matrix, changeFactors):
    numChanges = len(changeFactors)
    nonzeros = matrix[matrix != 0]
    mutateIdx = np.random.choice(range(len(nonzeros)), size=numChanges, replace=False)
    nonzeros[mutateIdx] += changeFactors
    matrix[matrix != 0] = nonzeros

def crossoverNonzeroMatrices(matrix1, matrix2):
    # Matrices must be the same shape with the same # of nonzero in the same places
    newMatrix = np.zeros(matrix1.shape)
    nonZeroMask = (matrix1 != 0)
    numNonzero = np.count_nonzero(matrix1)
    choices = np.random.choice([True, False], size=numNonzero)
    newMatrix[nonZeroMask] = np.where(choices, matrix1[nonZeroMask], matrix2[nonZeroMask])
    return newMatrix

def geenius_trainer():
    """Train a geenius"""
    # s = getSeed()
    s = 1
    np.random.seed(s)

    # Random search for best model for 5 tries  (not efficient at all)
    model_best = None
    score_best = -1

    populationSize = 20
    numSurvivors = 5

    survivors = []
    gen = 0
    while True:
        print('\ngeneration', gen)
        generation = []
        for i in range(populationSize):
            # Evaluate model on 3 runs
            if len(survivors) == 0:
                # Must be first generation
                child = Model(randomize=True)
                if gen == 0 and i == 0:
                    print('Num weights:', np.count_nonzero(child.Win), np.count_nonzero(child.W), np.count_nonzero(child.Wout))
            else:
                if i == 0 and model_best is not None:
                    child = model_best.copy()
                else:
                    # Choose two parents from survivors weighted by their scores
                    weights = np.array([m.fitness for m in survivors])
                    weights = weights / np.sum(weights)
                    parent1, parent2 = np.random.choice(survivors, p=weights, size=2)
                    r = np.random.uniform(0.20, 0.40) #(15 - np.maximum(score_best, 1))/15
                    child = parent1.mate(parent2).mutate(changeRate=r, maxChange=1.0)

            child.test()
            generation.append(child)

            if child.score > score_best:
                score_best = child.score
                model_best = child.copy()
                # print('\nbest Win :', model_best[0][model_best[0] != 0])
                # print('best W   :', model_best[1][model_best[1] != 0])
                # print('best Wout:', model_best[2][model_best[2] != 0])
                print('\n' + str(model_best))
                print('\n****** Overall best score:', score_best)
            # Return best result so far
            yield model_best.getComponents()

        # Sort by mean score, descending
        generation = sorted(generation, reverse=True, key=lambda m:m.score)
        survivors = [model for model in generation[:numSurvivors]]
        # Assign fitness
        selectionPressure = 2
        [model.setFitness((numSurvivors - k)*selectionPressure) for k, model in enumerate(survivors)]
        print('\nGeneration best score:', max([m.score for m in generation]))

        gen = gen + 1

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    import numpy as np
    from pathlib import Path
    from challenge_2 import train, evaluate

    print('RUNNING CODE:')
    with open(Path(__file__).resolve(), 'r') as f:
        print(f.read())

    # Training (100 seconds)
    print(f"Starting training for 100 seconds (user time)")
    model = train(geenius_trainer, timeout=100)

    seed = 12345
    np.random.seed(seed)

    # Evaluation
    start_time = time.time()
    score, std = evaluate(model, Bot, Environment, debug=False, seed=seed, progress=True)
    elapsed = time.time() - start_time
    print(f"Evaluation completed after {elapsed:.2f} seconds")
    print(f"Final score: {score:.2f} ± {std:.2f}")
    input('ready?')
    score, std = evaluate(model, Bot, Environment, runs=1, debug=True, seed=seed, progress=True)
