import numpy as np
import math
from scipy.special import comb
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
# from tqdm import tqdm

class IsingLattice:
    def __init__(self, temperature, initial_state, size):
        self.size = size
        self.T = temperature
        self.system = self._build_system(initial_state)

    @property
    def sqr_size(self):
        return (self.size, self.size)

    def _build_system(self, initial_state):
        if initial_state == 'r':
            system = np.random.choice([-1, 1], self.sqr_size)
        elif initial_state == 'u':
            system = np.ones(self.sqr_size)
        else:
            raise ValueError("Initial state must be 'r' or 'u'")

        return system

    def _bc(self, i):
        if i >= self.size:
            return 0
        if i < 0:
            return self.size - 1
        else:
            return i

    def energy(self, N, M):
        return -2*self.system[N, M]*(
            self.system[self._bc(N - 1), M] + self.system[self._bc(N + 1), M]
            + self.system[N, self._bc(M - 1)] + self.system[N, self._bc(M + 1)]
        )

    @property
    def internal_energy(self):
        e = 0
        E = 0
        E_2 = 0
        for i in range(self.size):
            for j in range(self.size):
                e = self.energy(i, j)
                E += e
                E_2 += e**2
        U = (1./self.size**2)*E
        U_2 = (1./self.size**2)*E_2
        return U, U_2

    @property
    def heat_capacity(self):
        U, U_2 = self.internal_energy
        return (U_2 - U**2)/(self.T)**2
    
    @property
    def entropy(self):
        N = self.size**2
        Np = int((N+np.sum(self.system))/2)
        return math.log(comb(N, Np, exact=True))/N

    @property
    def magnetization(self):
        return np.abs(np.sum(self.system)/self.size**2)


def run(lattice, epochs):
    for epoch in range(epochs):
        N, M = np.random.randint(0, lattice.size, 2)
        E = -1*lattice.energy(N, M)
        if E <= 0.:
            lattice.system[N, M] *= -1
        elif np.exp(-E/lattice.T) > np.random.rand():
            lattice.system[N, M] *= -1


def main(tInitial, tFinal, tStep, initial_state, size, epochs):
    temperature1 = np.arange(0.010, tInitial, 0.05)
    temperature2 = np.arange(tInitial, tFinal, tStep)
    temperature3 = np.arange(tFinal, 6, 0.05)
    temperature = np.concatenate(temperature1, temperature2, temperature3, axis=None)
    E = []
    S = []
    C = []
    M = []
    for T in tqdm(temperature):
        lattice = IsingLattice(temperature=T, initial_state=initial_state, size=size)
        run(lattice, epochs)
        E += [lattice.internal_energy[0]]
        S += [lattice.entropy]
        C += [lattice.heat_capacity]
        M += [lattice.magnetization]
        print("T = %.4f\tE = %.2f\t S = %.2f\t C = %.2f\t M = %.2f" %(T, lattice.internal_energy[0], lattice.entropy, lattice.heat_capacity, lattice.magnetization), end="\r")
    E = np.array(E)
    S = np.array(S)
    C = np.array(C)
    M = np.array(M)
    dataFileName = "data-ESCM-T-Init-"+initial_state+"-size"+str(int(size))+"-iter"+str(int(epochs))+".csv"
    np.savetxt(dataFileName, np.array([Temp, E, S, C, M]), delimiter=',')

if __name__ == "__main__":
    tInitial = 1.0
    tFinal = 3.4
    tStep = 0.01
    initial_state = "r"
    size = 100
    epochs = 1000000
    main(tInitial, tFinal, tStep, initial_state, size, epochs)