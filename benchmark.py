import utils



def plot(config, alg, stream):
    if config.plot_2d:
        utils.display_result(alg.means, alg.covars, stream[:])
    if config.plot_err:
        utils.display_err(alg.hist)

