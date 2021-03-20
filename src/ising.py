import numpy as np
import math
from scipy.special import comb
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from tqdm import tqdm
import click
import time


class IsingLattice:

    def __init__(self, temperature, initial_state, size):
        self.size = size
        self.T = temperature
        self.system = self._build_system(initial_state)

    @property
    def sqr_size(self):
        return (self.size, self.size)

    def _build_system(self, initial_state):
        """Build the system

        Build either a randomly distributed system or a homogeneous system (for
        watching the deterioration of magnetization

        Parameters
        ----------
        initial_state : str: "r" or other
            Initial state of the lattice.  currently only random ("r") initial
            state, or uniformly magnetized, is supported
        """

        if initial_state == 'r':
            system = np.random.choice([-1, 1], self.sqr_size)
        elif initial_state == 'u':
            system = np.ones(self.sqr_size)
        else:
            raise ValueError(
                "Initial State must be 'r', random, or 'u', uniform"
            )

        return system

    def _bc(self, i):
        """Apply periodic boundary condition

        Check if a lattice site coordinate falls out of bounds. If it does,
        apply periodic boundary condition

        Assumes lattice is square

        Parameters
        ----------
        i : int
            lattice site coordinate

        Return
        ------
        int
            corrected lattice site coordinate
        """
        if i >= self.size:
            return 0
        if i < 0:
            return self.size - 1
        else:
            return i

    def energy(self, N, M):
        """Calculate the energy of spin interaction at a given lattice site
        i.e. the interaction of a Spin at lattice site n,m with its 4 neighbors

        - S_n,m*(S_n+1,m + Sn-1,m + S_n,m-1, + S_n,m+1)

        Parameters
        ----------
        N : int
            lattice site coordinate
        M : int
            lattice site coordinate

        Return
        ------
        float
            energy of the site
        """
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
        temp = np.sum(self.system)
        Np = int((N+temp)/2)
        return math.log(comb(N, Np, exact=True))/N

    @property
    def magnetization(self):
        """Find the overall magnetization of the system
        """
        return np.abs(np.sum(self.system)/self.size**2)


def run(lattice, epochs, video=True):
    """Run the simulation
    """

    FFMpegWriter = manimation.writers['ffmpeg']
    writer = FFMpegWriter(fps=24)

    fig = plt.figure()

    with writer.saving(fig, "ising.mp4", 100):
        for epoch in tqdm(range(epochs)):
        #for epoch in range(epochs):
            # Randomly select a site on the lattice
            N, M = np.random.randint(0, lattice.size, 2)

            # Calculate energy of a flipped spin
            E = -1*lattice.energy(N, M)

            # "Roll the dice" to see if the spin is flipped
            if E <= 0.:
                lattice.system[N, M] *= -1
            elif np.exp(-E/lattice.T) > np.random.rand():
                lattice.system[N, M] *= -1

            if video and epoch % (epochs//100) == 0:
                img = plt.imshow(
                    lattice.system, interpolation='nearest', cmap='jet'
                )
                writer.grab_frame()
                img.remove()

    plt.close('all')


@click.command()
@click.option(
    '--temperature', '-t',
    default=0.5,
    show_default=True,
    help='temperature of the system'
)
@click.option(
    '--tinitial',
    default=0.01,
    show_default=True,
    help='start temperature of the system'
)
@click.option(
    '--tfinal',
    default=4.5,
    show_default=True,
    help='end temperature of the system'
)
@click.option(
    '--tstep',
    default=0.01,
    show_default=True,
    help='temperature step of the system'
)
@click.option(
    '--initial-state', '-i',
    default='r',
    type=click.Choice(['r', 'u'], case_sensitive=False),
    show_default=True,
    help='(R)andom or (U)niform initial state of the system'
)
@click.option(
    '--size', '-s',
    default=100,
    show_default=True,
    help='Number of sites, M, in the MxM lattice'
)
@click.option(
    '--epochs', '-e',
    default=1_000_000,
    type=int,
    show_default=True,
    help='Number of iterations to run the simulation for'
)
@click.option(
    '--video',
    is_flag=True,
    help='Record a video of the simulation progression'
)
def main(temperature, tinitial, tfinal, tstep, initial_state, size, epochs, video):
    lattice = IsingLattice(temperature=temperature, initial_state=initial_state, size=size)
    run(lattice, epochs, video)

#     Temp = np.concatenate((np.arange(0.010, tinitial, 0.1), np.arange(tinitial, tfinal, tstep), np.arange(tfinal, 4.5, 0.1), np.arange(4.5, 8, 0.2)), axis=None)
# #     Temp = np.arange(6, 15, 0.2)
#     E = []
#     S = []
#     C = []
#     M = []
#     startTime = time.time()
#     for T in Temp:
#         print("T=%.4f" %T)
#         lattice = IsingLattice(temperature=T, initial_state=initial_state, size=size)
#         run(lattice, epochs, video)
#         E += [lattice.internal_energy[0]]
#         S += [lattice.entropy]
#         C += [lattice.heat_capacity]
#         M += [lattice.magnetization]
#         print(f"{'Mean Energy [J]:':,<25}{lattice.internal_energy[0]:.2f}")
#         print(f"{'Entropy [J/K]:':,<25}{lattice.entropy:.2f}")
#     endTime = time.time()
#     E = np.array(E)
#     S = np.array(S)
#     C = np.array(C)
#     M = np.array(M)
#     dataFileName = "data-ESCM-T-size"+str(int(size))+"-iter"+str(int(epochs))+".csv"
#     np.savetxt(fileFileName, np.array([Temp, E, S, C, M]), delimiter=',')
    
#     fig, ax1 = plt.subplots(1, 1)
#     Eplot = ax1.plot(Temp, E, 'r-', label="E-T", alpha=0.7)
#     plt.minorticks_on()

#     ax2 = ax1.twinx()
#     Splot = ax2.plot(Temp, S, 'b-', label="S-T", alpha=0.7)

#     lns = Eplot+Splot
#     labs = [l.get_label() for l in lns]
#     ax1.legend(lns, labs, fontsize=15)

#     ax1.grid()
#     ax1.set_title("Enerygy & Entropy - Temperature", fontsize=20)
#     ax1.set_xlabel("Temperature", fontsize=15)
#     ax1.set_ylabel("Temperature", fontsize=15)
#     ax2.set_ylabel("Entropy", fontsize=15)

#     plt.savefig("ES-T.png", bbox_inches='tight', pad_inches=0.5)
#     plt.close()
    
#     print("Duration: %.2f" %(endTime-startTime))
    
#     plt.plot(Temp, E, '.-')
#     plt.savefig("E-T.png")
#     plt.close()

    print(f"{'Mean Energy [J]:':,<25}{lattice.internal_energy[0]:.2f}")
    print(f"{'Entropy [J/K]:':,<25}{lattice.entropy:.2f}")
    print(f"{'Net Magnetization [%]:':,<25}{lattice.magnetization:.2f}")
    print(f"{'Heat Capacity [AU]:':,<25}{lattice.heat_capacity:.2f}")


if __name__ == "__main__":
    plt.ion()
    main()
