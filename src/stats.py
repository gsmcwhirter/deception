import collections
import itertools
import math
import simulation

_use_stats = True
try:
    from sage.all_cmdline import *
    from sage.finance.time_series import TimeSeries
except ImportError:
    _use_stats = False

from simulations.base import listener
from simulations.statsparser import StatsParser as StatsParserBase


def set_options(this):
    """ Sets the OptionParser options

    Options:

    -n, --nosumm                Do not output summary statistics
    -q, --quiet                 Only output aggregate statistics
    -g STRING, --gphx=STRING    The format string for histogram outputs

    """
    this.oparser.add_option("-n", "--nosumm", action="store_false",
                                    dest="summary", default=True,
                                    help="do not output summary statistics")
    this.oparser.add_option("-q", "--quiet", action="store_true",
                                    dest="quiet", default=False,
                                    help="only output aggregate statistics")
    this.oparser.add_option("-g", "--gphx", action="store", dest="gphx",
                                    type="string", default=None,
                                    help="the filename pattern for histogram drawings")


def handle_result(this, out, count, result):
    """ Saves an unpickled result

    """

    try:
        if this._duplications is None:
            this._duplications = []
    except:
        this._duplications = []

    if this.options.verbose:
        print "[Result] {0} found".format(count)

    this._duplications.append((count, result))


def handle_options(this, out, options):
    """ Saves the unpickled simulations options object

    """

    if this.options.verbose:
        print "[Options]"

    this._result_options = options


def when_done(this, out, comparative=False):
    """ Actually calculates the statistics

    """

    massaged_dups = []

    for dup in this._duplications:
        #initial = dup[1][1]
        final_sender, final_receiver = dup[1][2]
        generations = dup[1][0]

        final_sender = [(j, st) for (j, st) in enumerate(final_sender)
                                    if st >= 10. * this._effective_zero]
        final_receiver = [(j, st) for (j, st) in enumerate(final_receiver)
                                    if st >= 10. * this._effective_zero]

        massaged_dups.append((final_sender, final_receiver, generations))

    print >> out, "Total Duplications: {0}, avg generations: {1:.2}".format(
                        len(massaged_dups),
                        float(sum(i[2] for i in massaged_dups)) / float(len(massaged_dups)))

    splitat = None

    if comparative:
        s_payoffs = simulation.sender_2
        r_payoffs = (simulation.receiver_2, simulation.lambda_payoffs(this._result_options.lam))
        splitat = len(this._duplications[0][1][2][0])

    elif this._result_options.routine == "nci":
        s_payoffs = simulation.sender_1
        r_payoffs = simulation.receiver_1

    elif this._result_options.routine == "sim":
        s_payoffs = simulation.sender_2
        r_payoffs = simulation.receiver_2

    elif this._result_options.routine == "dist":
        s_payoffs = simulation.sender_3
        r_payoffs = simulation.receiver_3

    else:
        raise ValueError("Unknown routine selected")

    if this.options.summary:
        output_summary(this, out, massaged_dups, comparative=comparative, splitat=splitat)

    output_klstats(this, out, massaged_dups, r_payoffs, s_payoffs, comparative=comparative, splitat=splitat)

    return massaged_dups


def when_done_comparative(this, out):
    """ Actually calculates the statistics

    """

    when_done(this, out, comparative=True)


def output_summary(this, out, duplications, comparative=False, splitat=None):
    """ Ouputs summary statistics

    """

    options = this.options
    end_states = {}

    for (final_sender, final_receiver, generations) in duplications:
        end_state = [[], []]

        if options.verbose or len(final_sender) <= 4:
            for i, state in final_sender:
                state_tmp = round(state, 5)
                if state_tmp < 1e-3:
                    state_tmp = 0.
                end_state[1].append((i, state_tmp))
        else:
            end_state[1] = [("mixed", "")]

        for i, state in final_receiver:
            if options.verbose or len(final_receiver) == 1:
                state_tmp = round(state, 5)
                if state_tmp < 1e-3:
                    state_tmp = 0.
            else:
                state_tmp = "some"
            end_state[0].append((i, state_tmp))

        end_state = tuple((tuple(end_state[0]), tuple(end_state[1])))

        if end_state in end_states:
            end_states[end_state][0] += 1
            end_states[end_state][1].append(generations)
        else:
            end_states[end_state] = [1, [generations]]

    ks = end_states.keys()
    ks.sort()

    bthstr = "\t{0:>5}: {1:>15}\t\t{2:>5}: {3}"
    sndstr = "\t{0:>22}\t\t{1:>5}: {2}"
    recstr = "\t{0:>5}: {1:>15}"

    comb_results = 0
    noncomb_results = 0
    mixed_results = 0

    for e in ks:
        has_comb = False
        has_noncomb = False
        print >> out, "-" * 72
        for rec, snd in itertools.izip_longest(e[0], e[1], fillvalue=None):
            print >> out, "\t",

            if rec and rec[0] < splitat:
                has_noncomb = True
            elif rec:
                has_comb = True

            if snd and rec:
                if comparative and rec[0] < splitat:
                    print >> out, bthstr.format("NC {0}".format(rec[0]), rec[1], snd[0], snd[1])
                elif comparative:
                    print >> out, bthstr.format("C {0}".format(rec[0] - splitat), rec[1], snd[0], snd[1])
                else:
                    print >> out, bthstr.format(rec[0], rec[1], snd[0], snd[1])
            elif rec:
                if comparative and rec[0] < splitat:
                    print >> out, recstr.format("NC {0}".format(rec[0]), rec[1])
                elif comparative:
                    print >> out, recstr.format("C {0}".format(rec[0] - splitat), rec[1])
                else:
                    print >> out, recstr.format(*rec)
            elif snd:
                print >> out, sndstr.format("", *snd)

        print >>out, "\t\t\t\t\t({0} times, {1:.2} avg gen)".format(
                            end_states[e][0],
                            float(sum(end_states[e][1])) / float(len(end_states[e][1])))

        if has_comb and has_noncomb:
            mixed_results += end_states[e][0]
        elif has_comb:
            comb_results += end_states[e][0]
        elif has_noncomb:
            noncomb_results += end_states[e][0]

    if comparative:
        print >> out
        print >> out, "Combinatorial Results: {0}".format(comb_results)
        print >> out, "Non-Combinatorial Results: {0}".format(noncomb_results)
        print >> out, "Mixed Results: {0}".format(mixed_results)

    print >> out, "=" * 72


def misinfo(msg, information_content):
    return any(any(info_content2 > 0.
                    for state2, info_content2 in enumerate(information_content)
                    if state2 != state)
                for state, info_content in enumerate(information_content))


def kl_measures(sender_pop, receiver_pop, n=2):
    msgs = list(itertools.product(range(n), range(n)))
    states = list(itertools.product(range(n), range(n)))
    state_probs = [1. / float(len(states))] * len(states)

    all_cprobs_msg_given_state = collections.defaultdict(list)
    all_cprobs_state_given_msg = collections.defaultdict(list)
    information_contents = collections.defaultdict(list)
    for i, msg in enumerate(msgs):
        cprobs_msg_given_state = []
        for j, state in enumerate(states):
            pr = 0.
            for (sender, sender_prob) in sender_pop:
                if simulation.sender_matrix(sender)[j][i] == 1:
                    pr += sender_prob
            cprobs_msg_given_state.append(pr)
        all_cprobs_msg_given_state[i] = cprobs_msg_given_state

        for j, state in enumerate(states):
            if math.fsum(cprobs_msg_given_state) > 0.:
                prob_state_given_msg = ((state_probs[j] * cprobs_msg_given_state[j]) /
                                        math.fsum(state_probs[k] * cprobs_msg_given_state[k]
                                                    for k in xrange(len(states))))
            else:
                prob_state_given_msg = float('inf')

            all_cprobs_state_given_msg[j].append(prob_state_given_msg)

            if prob_state_given_msg > 0. and not math.isinf(prob_state_given_msg):
                information_contents[i].append(math.log(prob_state_given_msg / state_probs[j]))
            else:
                information_contents[i].append(- float('inf'))

    return (information_contents, all_cprobs_state_given_msg, all_cprobs_msg_given_state)


def output_klstats(this, out, duplications, r_payoffs, s_payoffs, comparative=False, splitat=None):

    options = this.options

    times_misinformation = 0
    times_sender_hdeception = 0
    times_receiver_hdeception = 0
    times_full_deception = 0

    pcts_senders_deceptive = []
    pcts_senders_hdeceptive = []
    pcts_interactions_hdeceptive_potential = []
    pcts_interactions_deceptive_potential = []
    pcts_interactions_hdeceptive = []
    pcts_interactions_deceptive = []

    n = 2

    all_states = list(itertools.product(range(n), range(n)))
    state_probs = [1. / float(len(all_states))] * len(all_states)

    for dup_i, (final_sender, final_receiver, generations) in enumerate(duplications):
        misinfo_msgs = []

        (kls, all_cprobs_state_given_msg, all_cprobs_msg_given_state) = kl_measures(final_sender, final_receiver, n)

        if not options.quiet:
            print >> out, "Conditional Probabilities (msg: [pr(msg | state) for state in states]):"
            for msg, cprobs_msg_given_state in all_cprobs_msg_given_state.iteritems():
                print >> out, "\t{0:>5}: {1}".format(msg, cprobs_msg_given_state)
            print >> out, "Conditional Probabilities (msg: [pr(state | msg) for state in states]):"
            for msg, cprobs_state_given_msg in all_cprobs_state_given_msg.iteritems():
                print >> out, "\t{0:>5}: {1}".format(msg, cprobs_state_given_msg)
            print >> out, "KL Information Measures (msg: I(msg): [I(msg, state) for state in states]):"
            for msg, information_content in kls.iteritems():
                print >> out, "\t{0:>5}: {1:>15}: {2}".format(msg,
                                                      math.fsum(state_content * all_cprobs_state_given_msg[msg][state]
                                                        for state, state_content in enumerate(information_content)
                                                        if not math.isinf(state_content) and\
                                                            not math.isinf(all_cprobs_state_given_msg[msg][state])
                                                      ),
                                                      information_content)

        misinfo_msgs = [msg for msg, information_content in kls.iteritems()
                            if misinfo(msg, information_content)]

        if len(misinfo_msgs) > 0:
            times_misinformation += 1

        # sum of population proportions that are deceptive against some receiver
        percent_senders_deceptive = 0.
        # sum of population proportions that are almost deceptive against some receiver (no receiver detriment)
        percent_senders_halfdeceptive = 0.
        # percent of interactions that are deceptive
        percent_interactions_deceptive_potential = 0.
        # percent of interactions that are almost deceptive
        percent_interactions_halfdeceptive_potential = 0.
        #
        percent_interactions_deceptive = 0.
        #
        percent_interactions_halfdeceptive = 0.

        s_decept = False
        r_decept = False
        f_decept = False

        for (sender, sender_prob) in final_sender:
            s_matrix = simulation.sender_matrix(sender)
            s_hdeceptive = False
            s_fdeceptive = False
            for msg in misinfo_msgs:
                states_msg_sent = [state for state, row in enumerate(s_matrix) if row[msg] == 1]

                for (receiver, receiver_prob) in final_receiver:
                    sri_hdeceptive = False
                    sri_fdeceptive = False

                    if comparative and receiver < splitat:
                        r_matrix = simulation.sender_matrix(receiver)
                    elif comparative:
                            r_matrix = simulation.receiver_matrix(receiver - splitat)
                    elif this._result_options.combinatorial:
                        r_matrix = simulation.receiver_matrix(receiver)
                    else:
                        r_matrix = simulation.sender_matrix(receiver)
                    action = r_matrix[msg].index(1)

                    for state in states_msg_sent:
                        if comparative and receiver < splitat:
                            receiver_payoff_actual = r_payoffs[0][state][action]
                            receiver_payoff_ifknew = max(r_payoffs[0][state])

                            actions_ifknew = [act for act, payoff in enumerate(r_payoffs[0][state])
                                                if payoff == receiver_payoff_ifknew]
                        elif comparative:
                            receiver_payoff_actual = r_payoffs[1][state][action]
                            receiver_payoff_ifknew = max(r_payoffs[1][state])

                            actions_ifknew = [act for act, payoff in enumerate(r_payoffs[1][state])
                                                if payoff == receiver_payoff_ifknew]
                        else:
                            receiver_payoff_actual = r_payoffs[state][action]
                            receiver_payoff_ifknew = max(r_payoffs[state])

                            actions_ifknew = [act for act, payoff in enumerate(r_payoffs[state])
                                                if payoff == receiver_payoff_ifknew]

                        sender_payoff_actual = s_payoffs[state][action]
                        sender_payoffs_ifknew = [s_payoffs[state][act] for act in actions_ifknew]

                        if any(sender_payoff_actual > spk for spk in sender_payoffs_ifknew):
                            percent_interactions_halfdeceptive += sender_prob * receiver_prob * state_probs[state]
                            sender_benefit = True
                            if not s_decept:
                                times_sender_hdeception += 1
                                s_decept = True
                            if not s_hdeceptive:
                                percent_senders_halfdeceptive += sender_prob
                                s_hdeceptive = True
                            if not sri_hdeceptive:
                                percent_interactions_halfdeceptive_potential += sender_prob * receiver_prob
                                sri_hdeceptive = True
                        else:
                            sender_benefit = False

                        if receiver_payoff_actual < receiver_payoff_ifknew:
                            receiver_detriment = True
                            if not r_decept:
                                times_receiver_hdeception += 1
                                r_decept = True
                        else:
                            receiver_detriment = False

                        if sender_benefit and receiver_detriment:
                            percent_interactions_deceptive += sender_prob * receiver_prob * state_probs[state]
                            if not f_decept:
                                times_full_deception += 1
                                f_decept = True
                            if not s_fdeceptive:
                                percent_senders_deceptive += sender_prob
                                s_fdeceptive = True
                            if not sri_fdeceptive:
                                percent_interactions_deceptive_potential += sender_prob * receiver_prob
                                sri_fdeceptive = True

        pcts_senders_deceptive.append(percent_senders_deceptive)
        pcts_senders_hdeceptive.append(percent_senders_halfdeceptive)
        pcts_interactions_hdeceptive_potential.append(percent_interactions_halfdeceptive_potential)
        pcts_interactions_deceptive_potential.append(percent_interactions_deceptive_potential)
        pcts_interactions_hdeceptive.append(percent_interactions_halfdeceptive)
        pcts_interactions_deceptive.append(percent_interactions_deceptive)

    print >> out, "Number of duplications with misinformation: {0}".format(times_misinformation)
    print >> out, "Number of duplications with senders benefitting: {0}".format(times_sender_hdeception)
    print >> out, "Number of duplications with receivers losing out: {0}".format(times_receiver_hdeception)
    print >> out, "Number of duplications with deception: {0}".format(times_full_deception)
    print >> out

    if options.gphx is not None:
        fname_s_hdecept = options.gphx.format("s_hdecept")
        fname_s_decept = options.gphx.format("s_decept")
        fname_i_hdecept_pos = options.gphx.format("i_hdecept_pos")
        fname_i_decept_pos = options.gphx.format("i_decept_pos")
        fname_i_hdecept = options.gphx.format("i_hdecept")
        fname_i_decept = options.gphx.format("i_decept")
    else:
        fname_s_hdecept = None
        fname_s_decept = None
        fname_i_hdecept_pos = None
        fname_i_decept_pos = None
        fname_i_hdecept = None
        fname_i_decept = None

    print >> out, "Percent of Senders Benefitting:"
    print >> out, format_stats(pcts_senders_hdeceptive, "\t", fname_s_hdecept)
    print >> out, "Percent of Senders Deceptive:"
    print >> out, format_stats(pcts_senders_deceptive, "\t", fname_s_decept)
    print >> out, "Percent of Interactions with Potential Sender Benefit:"
    print >> out, format_stats(pcts_interactions_hdeceptive_potential, "\t", fname_i_hdecept_pos)
    print >> out, "Percent of Interactions with Potential Deception:"
    print >> out, format_stats(pcts_interactions_deceptive_potential, "\t", fname_i_decept_pos)
    print >> out, "Percent of Interactions with Sender Benefit:"
    print >> out, format_stats(pcts_interactions_hdeceptive, "\t", fname_i_hdecept)
    print >> out, "Percent of Interactions with Deception:"
    print >> out, format_stats(pcts_interactions_deceptive, "\t", fname_i_decept)
    print >> out, "=" * 72


def format_stats(data, prefix=None, gphx_file=None):
    ret = ""

    if prefix is None:
        prefix = ""

    if _use_stats:
        series = TimeSeries(data)

        ret += "{0}Mean: {1:.4%}\n".format(prefix, series.mean())
        ret += "{0}StdDev: {1:.4%}\n".format(prefix, series.standard_deviation())
        ret += "{0}Histogram: {1}\n".format(prefix, series.histogram(bins=10))

        if gphx_file is not None:
            series.plot_histogram(filename=gphx_file, figsize=6)
    else:
        ret = "[Warning] stats software is not available"

    return ret


@listener('oparser set up', set_options)
@listener('result', handle_result)
@listener('result options', handle_options)
@listener('done', when_done)
class StatsParser(StatsParserBase):

    def __init__(self, *args, **kwdargs):
        super(StatsParser, self).__init__(*args, **kwdargs)
        self._effective_zero = 1e-10


@listener('oparser set up', set_options)
@listener('result', handle_result)
@listener('result options', handle_options)
@listener('done', when_done_comparative)
class ComparativeStatsParser(StatsParserBase):

    def __init__(self, *args, **kwdargs):
        super(ComparativeStatsParser, self).__init__(*args, **kwdargs)
        self._effective_zero = 1e-10
