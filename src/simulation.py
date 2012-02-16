""" Runs simulations for deception research """

import itertools
import math
import multiprocessing as mp
import operator

from simulations.base import listener
from simulations.simulation_runner import SimulationRunner
from simulations.dynamics.npop_discrete_replicator import NPopDiscreteReplicatorDynamics as NPDRD


sender_1 = ((2., 0., 0., 0.),
            (0., 2., 0., 0.),
            (0., 0., 2., 0.),
            (0., 0., 2., 1.))

sender_2 = ((2., 1., 1., 0.),
            (1., 2., 0., 1.),
            (1., 0., 2., 1.),
            (2., 1., 1., 0.))

sender_3 = ((2., 4. / 3., 2. / 3., 0.),
            (4. / 3., 2., 4. / 3., 2. / 3.),
            (2. / 3., 4. / 3., 2., 4. / 3.),
            (2., 4. / 3., 2. / 3., 0.))

receiver_1 = ((2., 0., 0., 0.),
              (0., 2., 0., 0.),
              (0., 0., 2., 0.),
              (0., 0., 2., 3.))

receiver_2 = ((2., 1., 1., 0.),
              (1., 2., 0., 1.),
              (1., 0., 2., 1.),
              (0., 1., 1., 2.))

receiver_3 = ((2., 4. / 3., 2. / 3., 0.),
              (4. / 3., 2., 4. / 3., 2. / 3.),
              (2. / 3., 4. / 3., 2., 4. / 3.),
              (0., 2. / 3., 4. / 3., 2.))

_choices = ["nci", "sim", "dist"]


def lambda_payoffs(lam):
    lam = float(lam)
    return ((2. / (1. + lam), 1. / (1. + lam), 1. / (1. + lam), 0.),
            (1. / (1. + lam), 2. / (1. + lam), 0., 1. / (1. + lam)),
            (1. / (1. + lam), 0., 2. / (1. + lam), 1. / (1. + lam)),
            (0., 1. / (1. + lam), 1. / (1. + lam), 2. / (1. + lam)))


def add_options(this):
    this.oparser.add_option("-r", "--routine", action="store",
                                choices=_choices, dest="routine",
                                help="name of routine to run")
    this.oparser.add_option("--combinatorial", action="store_true",
                                dest="combinatorial", default=False,
                                help="use a combinatorial model")
    this.oparser.add_option("--noncombinatorial", action="store_true",
                                dest="noncombinatorial", default=False,
                                help="use a non-combinatorial model")


def add_comparative_options(this):
    this.oparser.add_option("-l", "--lambda", action="store",
                                type="float", dest="lam",
                                help="value of the lambda parameter")


def check_options(this):
    if not this.options.routine in _choices:
        this.oparser.error("Unknown routine selected")

    if this.options.combinatorial and this.options.noncombinatorial:
        this.oparser.error("You must select either --combinatorial or --noncombinatorial, but not both.")
    elif not this.options.combinatorial and not this.options.noncombinatorial:
        this.oparser.error("You must select either --combinatorial or --noncombinatorial")


def check_comparative_options(this):
    if this.options.lam < 0.:
        this.oparser.error("The lambda parameter must be positive.")


def set_data(this):

    #near common interest
    if this.options.routine == "nci":
        this.data['s_payoffs'] = sender_1
        this.data['r_payoffs'] = receiver_1

    #similarity
    elif this.options.routine == "sim":
        this.data['s_payoffs'] = sender_2
        this.data['r_payoffs'] = receiver_2

    #distance
    elif this.options.routine == "dist":
        this.data['s_payoffs'] = sender_3
        this.data['r_payoffs'] = receiver_3

    else:
        raise ValueError("Unable to determine requested payoffs")

    if this.options.combinatorial:
        this._simulation_class = CombinatorialSignallingGame
        this.data['is_combinatorial'] = True
    else:
        this._simulation_class = NonCombinatorialSignallingGame
        this.data['is_combinatorial'] = False


def set_comparative_data(this):
    this.data['s_payoffs'] = sender_2
    this.data['r_payoffs_c'] = receiver_2
    this.data['r_payoffs_nc'] = lambda_payoffs(this.options.lam)


def done_handler(this):
    print "[Done]"


def go_handler(this):
    print "[Go]"


def output_dir_handler(this):
    print "[Made Output Directory] {0}".format(this.options.output_dir)


def pool_started_handler(this, pool):
    try:
        pool_size = pool._processes
    except AttributeError:
        if this.options.pool_size is None:
            pool_size = mp.cpu_count()
        else:
            pool_size = this.options.pool_size

    print "[Pool Started] {0} parallel computations".format(pool_size)


def start_handler(this):
    print "[Starting simulations]"


def result_handler(this, result):
    try:
        print "[Result] {0} complete in {1} generations".format(this.finished_count,
                                                                result[0])
    except TypeError:
        print "[Result (error)] {0}".format(result)


@listener('done', done_handler)
@listener('go', go_handler)
@listener('made output_dir', output_dir_handler)
@listener('pool started', pool_started_handler)
@listener('start', start_handler)
@listener('result', result_handler)
@listener('oparser set up', add_options)
@listener('options parsed', set_data)
@listener('options parsed', check_options)
class Runner(SimulationRunner):

    def __init__(self, *args, **kwdargs):

        if 'default_handlers' not in kwdargs:
            kwdargs['default_handlers'] = False

        if len(args) == 0:
            args = [None]

        super(Runner, self).__init__(*args, **kwdargs)


@listener('done', done_handler)
@listener('go', go_handler)
@listener('made output_dir', output_dir_handler)
@listener('pool started', pool_started_handler)
@listener('start', start_handler)
@listener('result', result_handler)
@listener('oparser set up', add_comparative_options)
@listener('options parsed', set_comparative_data)
@listener('options parsed', check_comparative_options)
class ComparativeRunner(SimulationRunner):

    def __init__(self, *args, **kwdargs):

        if 'default_handlers' not in kwdargs:
            kwdargs['default_handlers'] = False

        if len(args) == 0:
            args = [ComparativeSignallingGame]

        super(ComparativeRunner, self).__init__(*args, **kwdargs)


def format_matrix(matrix, prefix=None):
    if prefix is None:
        prefix = ""

    ret = prefix + "["
    first = True
    for row in matrix:
        if first:
            ret += "[" + ", ".join(["{0}".format(i) for i in row]) + "]"
            first = False
        else:
            ret += "\n" + prefix + " [" + ", ".join(["{0}".format(i) for i in row]) + "]"
    ret += "]\n"

    return ret


def format_population(this, pop, stable=False, comparative=False):
    ret = ""

    fstr = "\t\t{0:>5}: {1:>5}: {2}\n"

    ret += "\tSenders:\n"
    for sender, proportion in enumerate(pop[0]):
        if abs(proportion - 0.) > this.effective_zero:
            ret += fstr.format(sender, this.types[0][sender], proportion)
            if stable:
                ret += format_matrix(sender_matrix(sender), "\t\t\t")

    ret += "\tReceivers:\n"
    for receiver, proportion in enumerate(pop[1]):
        if abs(proportion - 0.) > this.effective_zero:
            if comparative and receiver >= len(this.types[0]):
                ret += fstr.format("C {0}".format(receiver - len(this.types[0])), this.types[1][receiver], proportion)
            elif comparative:
                ret += fstr.format("NC {0}".format(receiver), this.types[1][receiver], proportion)
            else:
                ret += fstr.format(receiver, this.types[1][receiver], proportion)

            if stable:
                if comparative:
                    if receiver < len(this.types[0]):
                        ret += format_matrix(sender_matrix(receiver), "\t\t\t")
                    else:
                        ret += format_matrix(receiver_matrix(receiver - len(this.types[0])), "\t\t\t")
                elif this.data['is_combinatorial']:
                    ret += format_matrix(receiver_matrix(receiver), "\t\t\t")
                else:
                    ret += format_matrix(sender_matrix(receiver), "\t\t\t")

    return ret


def stable_state_handler(this, genct, thisgen, lastgen, firstgen, comparative=False):

    if this.force_stop:
        fstr = "[Force stop] {0} generations"
    else:
        fstr = "[Stable state] {0} generations"

    print >> this.out, fstr.format(genct)
    print >> this.out, format_population(this, thisgen, stable=True, comparative=comparative)


def stable_state_handler_comparative(*args, **kwdargs):
    stable_state_handler(*args, comparative=True, **kwdargs)


def initial_set_handler(this, initial_pop):
    print >> this.out, "[Initial population set]"
    print >> this.out, format_population(this, initial_pop)


def generation_handler(this, genct, thispop, lastpop):
    print >> this.out, "[Generation] {0} complete.".format(genct)
    print >> this.out, format_population(this, thispop)


def generate_stateact_cache_com(this):

    sacache = {}

    for profile in itertools.product(*this.types):
        sstrat = sender_matrix(profile[0])
        rstrat = receiver_matrix(profile[1])

        sacache[tuple(profile)] = {}

        for state in range(4):
            msg = sstrat[state].index(1)
            act = rstrat[msg].index(1)

            sacache[tuple(profile)][state] = act

    this.data['stateact_cache'] = sacache


def generate_stateact_cache_noncom(this):

    sacache = {}

    for profile in itertools.product(*this.types):
        sstrat = sender_matrix(profile[0])
        rstrat = sender_matrix(profile[1])

        sacache[tuple(profile)] = {}

        for state in range(4):
            msg = sstrat[state].index(1)
            act = rstrat[msg].index(1)

            sacache[tuple(profile)][state] = act

    this.data['stateact_cache'] = sacache


def generate_stateact_cache_comparative(this):

    sacache = {}

    for profile in itertools.product(*this.types):
        sstrat = sender_matrix(profile[0])

        if profile[1] < len(this.types[0]):
            rstrat = sender_matrix(profile[1])
        else:
            rstrat = receiver_matrix(profile[1] - len(this.types[0]))

        sacache[tuple(profile)] = {}

        for state in range(4):
            msg = sstrat[state].index(1)
            act = rstrat[msg].index(1)

            sacache[tuple(profile)][state] = act

    this.data['stateact_cache'] = sacache


def sender_matrix(s):
    return ((
                (s & 1) & ((s & 16) >> 4),
                (s & 1) & (~(s & 16) >> 4),
                (~(s & 1) & ((s & 16) >> 4)) & 1,
                (~(s & 1) & (~(s & 16) >> 4)) & 1
            ),
            (
                ((s & 2) >> 1) & ((s & 32) >> 5),
                ((s & 2) >> 1) & (~(s & 32) >> 5),
                ((~(s & 2) >> 1) & ((s & 32) >> 5)) & 1,
                ((~(s & 2) >> 1) & (~(s & 32) >> 5)) & 1
            ),
            (
                ((s & 4) >> 2) & ((s & 64) >> 6),
                ((s & 4) >> 2) & (~(s & 64) >> 6),
                ((~(s & 4) >> 2) & ((s & 64) >> 6)) & 1,
                ((~(s & 4) >> 2) & (~(s & 64) >> 6)) & 1
            ),
            (
                ((s & 8) >> 3) & ((s & 128) >> 7),
                ((s & 8) >> 3) & (~(s & 128) >> 7),
                ((~(s & 8) >> 3) & ((s & 128) >> 7)) & 1,
                ((~(s & 8) >> 3) & (~(s & 128) >> 7)) & 1
            )
           )


def receiver_matrix(r):
    return ((
                (r & 1) & ((r & 4) >> 2),
                (r & 1) & (~(r & 4) >> 2),
                (~(r & 1) & ((r & 4) >> 2)) & 1,
                (~(r & 1) & (~(r & 4) >> 2)) & 1
            ),
            (
                (r & 1) & ((r & 8) >> 3),
                (r & 1) & (~(r & 8) >> 3),
                (~(r & 1) & ((r & 8) >> 3)) & 1,
                (~(r & 1) & (~(r & 8) >> 3)) & 1
            ),
            (
                ((r & 2) >> 1) & ((r & 4) >> 2),
                ((r & 2) >> 1) & (~(r & 4) >> 2),
                ((~(r & 2) >> 1) & ((r & 4) >> 2)) & 1,
                ((~(r & 2) >> 1) & (~(r & 4) >> 2)) & 1
            ),
            (
                ((r & 2) >> 1) & ((r & 8) >> 3),
                ((r & 2) >> 1) & (~(r & 8) >> 3),
                ((~(r & 2) >> 1) & ((r & 8) >> 3)) & 1,
                ((~(r & 2) >> 1) & (~(r & 8) >> 3)) & 1
            )
            )


@listener('initial set', initial_set_handler)
@listener('generation', generation_handler)
@listener('stable state', stable_state_handler)
@listener('force stop', stable_state_handler)
class CombinatorialSignallingGame(NPDRD):

    def __init__(self, *args, **kwdargs):
        #if 'parameters' not in kwdargs:
        #    parameters = (2, 2)
        #else:
        #    parameters = kwdargs['parameters']

        parameters = (2, 2)

        n = reduce(operator.mul, parameters, 1)

        kwdargs['types'] = [range(n ** n),
                            range(reduce(operator.mul, [p ** p for p in parameters], 1))]

        if 'default_handlers' not in kwdargs:
            kwdargs['default_handlers'] = False

        super(CombinatorialSignallingGame, self).__init__(*args, **kwdargs)

        self.stateact_cache = None

        if 'stateact_cache' not in self.data or not self.data['stateact_cache']:
            generate_stateact_cache_com(self)

        self.stateact_cache = self.data['stateact_cache']
        self.interaction_cache = {}
        self.state_probs = tuple([1. / float(n)] * n)

    def _profile_payoffs(self, profile):
        if profile not in self.interaction_cache:
            self.interaction_cache[profile] = (
                math.fsum(
                    self.data['s_payoffs'][state][act] * self.state_probs[state]
                    for state, act in self.stateact_cache[profile].iteritems()),
                math.fsum(
                    self.data['r_payoffs'][state][act] * self.state_probs[state]
                    for state, act in self.stateact_cache[profile].iteritems())
            )

        return self.interaction_cache[profile]


@listener('initial set', initial_set_handler)
@listener('generation', generation_handler)
@listener('stable state', stable_state_handler)
@listener('force stop', stable_state_handler)
class NonCombinatorialSignallingGame(NPDRD):

    def __init__(self, *args, **kwdargs):
        #if 'parameters' not in kwdargs:
        #    parameters = (2, 2)
        #else:
        #    parameters = kwdargs['parameters']

        parameters = (2, 2)

        n = reduce(operator.mul, parameters, 1)

        kwdargs['types'] = [range(n ** n),
                            range(n ** n)]

        if 'default_handlers' not in kwdargs:
            kwdargs['default_handlers'] = False

        super(NonCombinatorialSignallingGame, self).__init__(*args, **kwdargs)

        self.stateact_cache = None

        if 'stateact_cache' not in self.data or not self.data['stateact_cache']:
            generate_stateact_cache_noncom(self)

        self.stateact_cache = self.data['stateact_cache']
        self.interaction_cache = {}
        self.state_probs = tuple([1. / float(n)] * n)

    def _profile_payoffs(self, profile):
        if profile not in self.interaction_cache:
            self.interaction_cache[profile] = (
                math.fsum(
                    self.data['s_payoffs'][state][act] * self.state_probs[state]
                    for state, act in self.stateact_cache[profile].iteritems()),
                math.fsum(
                    self.data['r_payoffs'][state][act] * self.state_probs[state]
                    for state, act in self.stateact_cache[profile].iteritems())
            )

        return self.interaction_cache[profile]


@listener('initial set', initial_set_handler)
@listener('generation', generation_handler)
@listener('stable state', stable_state_handler_comparative)
@listener('force stop', stable_state_handler_comparative)
class ComparativeSignallingGame(NPDRD):

    def __init__(self, *args, **kwdargs):
        #if 'parameters' not in kwdargs:
        #    parameters = (2, 2)
        #else:
        #    parameters = kwdargs['parameters']

        parameters = (2, 2)

        n = reduce(operator.mul, parameters, 1)

        kwdargs['types'] = [range(n ** n),
                            range(n ** n + reduce(operator.mul, [p ** p for p in parameters], 1))]

        if 'default_handlers' not in kwdargs:
            kwdargs['default_handlers'] = False

        super(ComparativeSignallingGame, self).__init__(*args, **kwdargs)

        self.stateact_cache = None

        if 'stateact_cache' not in self.data or not self.data['stateact_cache']:
            generate_stateact_cache_comparative(self)

        self.stateact_cache = self.data['stateact_cache']
        self.interaction_cache = {}
        self.state_probs = tuple([1. / float(n)] * n)

    def _profile_payoffs(self, profile):
        if profile not in self.interaction_cache:
            if profile[1] < len(self.types[0]):
                r_payoffs = self.data['r_payoffs_nc']
            else:
                r_payoffs = self.data['r_payoffs_c']

            self.interaction_cache[profile] = (
                math.fsum(
                    self.data['s_payoffs'][state][act] * self.state_probs[state]
                    for state, act in self.stateact_cache[profile].iteritems()),
                math.fsum(
                    r_payoffs[state][act] * self.state_probs[state]
                    for state, act in self.stateact_cache[profile].iteritems())
            )

        return self.interaction_cache[profile]
