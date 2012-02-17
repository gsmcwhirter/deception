from simulation import ComparativeRunner

if __name__ == '__main__':
    runner = ComparativeRunner()
    print "Yaba daba doo..."

    runner.go()

#    import simulation
#    data = {}
#    data['s_payoffs'] = simulation.sender_2
#    data['r_payoffs_c'] = simulation.receiver_2
#    data['r_payoffs_nc'] = simulation.lambda_payoffs(0.1)
#
#    sim = simulation.ComparativeSignallingGame(data, 1, None)
#    sim.run()
