""" Runs simulations for deception research """

from simulations.simulation_runner import SimulationRunner
from simulations.dynamics.npop_discrete_replicator import NPopDiscreteReplicatorDynamics as NPDRD


class Runner(SimulationRunner):

    def __init__(self):
        super(Runner, self).__init__()


class SignallingGame(NPDRD):

    def __init__(self):
        super(SignallingGame, self).__init__()
