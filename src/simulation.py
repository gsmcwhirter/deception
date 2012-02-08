""" Runs simulations for deception research """

import itertools
import math
import operator

from simulations.base import listener
from simulations.simulation_runner import SimulationRunner
from simulations.dynamics.npop_discrete_replicator import NPopDiscreteReplicatorDynamics as NPDRD


receiver_sim_2 = ((2., 1., 1., 0.),
                  (1., 2., 0., 1.),
                  (1., 0., 2., 1.),
                  (0., 1., 1., 2.))

receiver_dist_2 = ((2., 4. / 3., 2. / 3., 0.),
                   (4. / 3., 2., 4. / 3., 2. / 3.),
                   (2. / 3., 4. / 3., 2., 4. / 3.),
                   (0., 2. / 3., 4. / 3., 2.))

sender_sim_2_1 = ((0., 1., 1., 2.),
                   (1., 2., 0., 1.),
                   (1., 0., 2., 1.),
                   (0., 1., 1., 2.))

sender_sim_2_2 = ((2., 1., 1., 0.),
                   (2., 1., 0., 1.),
                   (1., 0., 1., 2.),
                   (0., 2., 1., 1.))

sender_dist_2_1 = ((0., 4. / 3., 2. / 3., 2.),
                   (4. / 3., 2., 4. / 3., 2. / 3.),
                   (2. / 3., 4. / 3., 2., 4. / 3.),
                   (0., 2. / 3., 4. / 3., 2.))

sender_dist_2_2 = ((2., 4. / 3., 2. / 3., 0.),
                   (2., 4. / 3., 4. / 3., 2. / 3.),
                   (2. / 3., 4. / 3., 4. / 3., 2.),
                   (0., 2., 4. / 3., 2. / 3.))

_choices = ["simil0", "simil1", "simil2", "dist0", "dist1", "dist2"]


def add_options(this):
    this.oparser.add_option("-r", "--routine", action="store",
                                choices=_choices, dest="routine",
                                help="name of routine to run")


def check_options(this):
    if not this.options.routine in _choices:
        this.oparser.error("Unknown routine selected")


def set_data(this):
    #common interest
    if this.options.routine == "simil0":
        this.data['s_payoffs'] = receiver_sim_2
        this.data['r_payoffs'] = receiver_sim_2

    #sender map 1
    elif this.options.routine == "simil1":
        this.data['s_payoffs'] = sender_sim_2_1
        this.data['r_payoffs'] = receiver_sim_2

    #sender map 3
    elif this.options.routine == "simil2":
        this.data['s_payoffs'] = sender_sim_2_2
        this.data['r_payoffs'] = receiver_sim_2

    #common interest
    elif this.options.routine == "dist0":
        this.data['s_payoffs'] = receiver_dist_2
        this.data['r_payoffs'] = receiver_dist_2

    #sender map 1
    elif this.options.routine == "dist1":
        this.data['s_payoffs'] = sender_dist_2_1
        this.data['r_payoffs'] = receiver_dist_2

    #sender map 3
    elif this.options.routine == "dist2":
        this.data['s_payoffs'] = sender_dist_2_2
        this.data['r_payoffs'] = receiver_dist_2

    else:
        raise ValueError("Unable to determine requested payoffs")


def done_handler(this):
    print "[Done]"


def go_handler(this):
    print "[Go]"


def output_dir_handler(this):
    print "[Made Output Directory] {0}".format(this.options.output_dir)


def pool_started_handler(this, pool):
    print "[Pool Started] {0} parallel computations".format(pool.get_ncpus())


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

        super(Runner, self).__init__(*args, **kwdargs)


def format_population(this, pop):
    ret = ""

    fstr = "\t\t{0:>5}: {1:>5}: {2}\n"

    ret += "\tSenders:\n"
    for sender, proportion in enumerate(pop[0]):
        if abs(proportion - 0.) > this.effective_zero:
            ret += fstr.format(sender, this.types[0][sender], proportion)

    ret += "\tReceivers:\n"
    for receiver, proportion in enumerate(pop[1]):
        if abs(proportion - 0.) > this.effective_zero:
            ret += fstr.format(receiver, this.types[1][receiver], proportion)

    return ret


def stable_state_handler(this, genct, thisgen, lastgen, firstgen):

    if this.force_stop:
        fstr = "[Force stop] {0} generations"
    else:
        fstr = "[Stable state] {0} generations"

    print >> this.out, fstr.format(genct)
    print >> this.out, format_population(this, thisgen)


def initial_set_handler(this, initial_pop):
    print >> this.out, "[Initial population set]"
    print >> this.out, format_population(this, initial_pop)


def generation_handler(this, genct, thispop, lastpop):
    print >> this.out, "[Generation] {0} complete.".format(genct)
    print >> this.out, format_population(this, thispop)


def generate_stateact_cache(this):

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
class SingleSignallingGame(NPDRD):

    def __init__(self, *args, **kwdargs):
        #if 'parameters' not in kwdargs:
        #    parameters = (2, 2)
        #else:
        #    parameters = kwdargs['parameters']

        parameters = (2, 2)

        n = reduce(operator.mul, parameters, 1)

        kwdargs['types'] = [range(n ** n),
                            range(
                                reduce(
                                    operator.mul,
                                    [p ** p for p in parameters],
                                    1
                                )
                            )]

        if 'default_handlers' not in kwdargs:
            kwdargs['default_handlers'] = False

        super(SingleSignallingGame, self).__init__(*args, **kwdargs)

        self.stateact_cache = None

        if 'stateact_cache' not in self.data or not self.data['stateact_cache']:
            generate_stateact_cache(self)

        self.stateact_cache = self.data['stateact_cache']
        self.interaction_cache = {}
        self.state_probs = tuple([1. / float(n)] * n)

    def _interaction(self, index, profile):
        if index != 0 and index != 1:
            raise ValueError("Strategy profile index out of bounds")

        if profile not in self.interaction_cache:
            self.interaction_cache[profile] = (
                math.fsum(
                    self.data['s_payoffs'][state][act] * self.state_probs[state]
                    for state, act in self.stateact_cache[profile].iteritems()),
                math.fsum(
                    self.data['r_payoffs'][state][act] * self.state_probs[state]
                    for state, act in self.stateact_cache[profile].iteritems())
            )

        return self.interaction_cache[profile][index]
