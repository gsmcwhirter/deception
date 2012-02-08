from simulation import Runner, SingleSignallingGame

if __name__ == '__main__':
    runner = Runner(SingleSignallingGame, pp_deps=(SingleSignallingGame,))
    print "Yaba daba doo..."

    runner.go()
