import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse


import compute
import score
import utils
import loader

# Default values

binSize = 30
binNumber = 20
n = 20
max_worker = 8
metric=[1, 1]

# Hyperparameters

stepSize = 50

# Functions

try:
    matplotlib.use('tkagg')
except ImportError:
    pass

def plotWaveforms(spk, normalization):
    spk = spk - spk.mean(axis=0)
    spk = spk/normalization
    spk = spk[:,::-1]

    for i in range(spk.shape[1]):
        plt.plot(spk[:, i] + i)

def getScore(lags, crosscorr, corr1, corr2, waveforms1, waveforms2):
    score1 = score.lagScore(lags, crosscorr)
    score2 = score.waveformsScore(waveforms1, waveforms2)
    return score1, score2
    

def plotSingle(args):
    print("Loading data")
    clu_data = loader.loadClu(args)
    clu_data = clu_data[1:]
    relevent = np.logical_or(clu_data == args.ref, clu_data == args.target)
    res_data = loader.loadRes(args)
    xml_data = loader.loadXml(args)
    spk_data = loader.loadSpikes(clu_data, [args.ref, args.target], xml_data, args)

    clu_data = clu_data[relevent]
    res_data = res_data[relevent]

    units = utils.getUnits(clu_data)
    lags, crosscorr1 = compute.getSingleCrossCorr(res_data, clu_data, args.ref, args.target, args.binSize, args.binNumber)
    lags, corr1 = compute.getSingleCrossCorr(res_data, clu_data, args.ref, args.ref, args.binSize, args.binNumber)
    lags, corr2 = compute.getSingleCrossCorr(res_data, clu_data, args.target, args.target, args.binSize, args.binNumber)
    
    waveforms = compute.computeWaveforms(clu_data, spk_data, xml_data, args.session)
    waveforms1 = waveforms[np.where(units == args.ref)[0][0]]
    waveforms2 = waveforms[np.where(units == args.target)[0][0]]

    score1, score2 = getScore(lags, crosscorr1, corr1, corr2, waveforms1, waveforms2)
    score = 100*score1**args.metric[0]*score2**args.metric[1]
    print(f"Refractory: {score1}\nWaveforms: {score2}\nScore: {score:.1f}")

    spks = compute.computeWaveforms(clu_data, spk_data, xml_data, args.session)

    units = utils.getUnits(clu_data)
    indexUnitref = np.where(units == args.ref)[0][0]
    indexUnittarget = np.where(units == args.target)[0][0]

    spk1 = spks[indexUnitref]
    spk2 = spks[indexUnittarget]
    normalisation = max(np.max(spk1 - spk1.mean(axis=0)), np.max(spk2 - spk2.mean(axis=0)))

    pdf = crosscorr1/np.sum(crosscorr1)

    plt.figure()
    plt.subplot(221)
    plotWaveforms(spk1, normalisation)
    plt.subplot(222)
    plotWaveforms(spk2, normalisation)
    plt.subplot(223)
    plt.bar(lags, pdf)
    plt.subplot(224)
    y = np.correlate(pdf, pdf, mode='full')
    y = y[len(pdf)-1:]
    x = np.arange(0, len(pdf))

    coeffs = np.polyfit(x, y, deg=1)  # Returns [slope, intercept]
    y_pred = np.polyval(coeffs, x)

    plt.bar(x, y)
    plt.plot(x, y_pred, color='red')
    plt.show()

def computeScore(args):
    flag = True
    lastUnits = np.array([])
    lastCorr = np.array([])
    lastCorr = lastCorr.reshape((0, 0, 2*args.binNumber))
    memory = []
    print("Loading data")
    res_data, clu_data, spk_data, xml_data = loader.load(args)
    input("Press any key to load the .clu file and start the computation")
    while flag:
        clu_data = loader.loadClu(args)
        clu_data = clu_data[1:]
        units = utils.getUnits(clu_data)
        n = len(units)
        sameIndex = np.isin(units, lastUnits)
        reusedIndex = np.isin(lastUnits, units)
        restrict = units[~sameIndex]
        lastUnits = units
        
        lags, newCorrs = compute.computeAllCrossCorr(res_data, clu_data, args.binSize, args.binNumber, args.max_workers, restrict)
        waveforms = compute.computeWaveforms(clu_data, spk_data, xml_data, args.session)

        corrs = np.zeros((n, n, 2*args.binNumber))

        old = np.outer(sameIndex, sameIndex)
        reused = np.outer(reusedIndex, reusedIndex)

        corrs[old, :] = lastCorr[reused, :]
        corrs[~old, :] = newCorrs[~old, :]

        lastCorr = corrs

        likelihood = score.getLikelihood(lags, corrs, waveforms)
        for i in range(len(args.metric)):
            if args.metric[i] == 0:
                likelihood[:, :, i] = 1
        likelihood = np.prod(likelihood, axis=2)

        if args.trust:
            for pair in memory:
                if pair[0] in units and pair[1] in units:
                    x = np.where(units == pair[0])[0][0]
                    y = np.where(units == pair[1])[0][0]
                    likelihood[x, y] = 0
                    likelihood[y, x] = 0

        largest_values, rows, cols = score.sortScore(likelihood, args.n)

        nonZeros = largest_values > 0
        largest_values = largest_values[nonZeros]
        rows = rows[nonZeros]
        cols = cols[nonZeros]

        if (~nonZeros).all():
            print("No more pairs")
        else:

            graph, linkScore = utils.computeGraph(largest_values, rows, cols)
            groups, groupsScore = utils.getConnectedComponents(graph, linkScore)
            
            sorted_indices = np.argsort(groupsScore)[::-1]
            groupsScore = [groupsScore[i] for i in sorted_indices]
            groups = [groups[i] for i in sorted_indices]

            pairs = utils.getPairsFromGroups(units, groups)
            for pair in pairs:
                memory.append(pair)

            for groupScore, group in zip(groupsScore, groups):
                grouped_units = tuple(int(u) for u in units[list(group)])
                grouped_units = sorted(grouped_units)
                print(f"Units: {grouped_units} | Score: {100*groupScore:.1f}")


        if args.plot:
            plt.figure()
            plt.imshow(likelihood, origin='lower', aspect='auto')
            plt.show()
        
        if args.persistent:
            key = None
            escape = False
            print("<Enter> to refresh | <r> to recompute | <q> to quit | :<arg>=<value> to set parameters")
            while not escape:
                key = input()
                key = key.strip()
                if key == "":
                    flag = True
                    escape = True
                elif key == "r":
                    flag = True
                    escape = True
                    lastUnits = np.array([])
                    lastCorr = np.array([])
                    lastCorr = lastCorr.reshape((0, 0, 2*args.binNumber))
                    memory = []
                elif key == "q":
                    flag = False
                    escape = True
                elif key == "forget":
                    memory = []
                elif key[0] == ":":
                    key = key[1:]
                    command = key.split("=")
                    if len(command) != 2:
                        print("Invalid command")
                        continue
                    arg = command[0]
                    value = command[1]
                    arg = arg.strip()
                    value = value.strip()
                    if arg not in dir(args) or not value.replace(" ", "").isnumeric():
                        print("Invalid command")
                        continue
                    value = value.split(" ")
                    value = [int(v) for v in value]
                    if len(value) == 1:
                        value = value[0]
                    setattr(args, arg, value)
                else:
                    print("Invalid command")

        else:
            flag = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Klusters")
    
    subparsers = parser.add_subparsers(dest="command")

    single_parser = subparsers.add_parser("single", help="Compute cross-correlation")
    single_parser.add_argument("path", type=str, help="Path to the data")
    single_parser.add_argument("session", type=int, help="Session number")
    single_parser.add_argument("ref", type=int, help="Reference unit")
    single_parser.add_argument("target", type=int, help="Target unit")
    single_parser.add_argument("--metric", type=int, choices=[0, 1], nargs=len(metric), default=metric, help="Metric to use")
    single_parser.add_argument("--binSize", type=int, default=binSize, help="Bin size")
    single_parser.add_argument("--binNumber", type=int, default=binNumber, help="Bin number")
    single_parser.set_defaults(func=plotSingle)

    score_parser = subparsers.add_parser("score", help="Compute score")
    score_parser.add_argument("path", type=str, help="Path to the data")
    score_parser.add_argument("session", type=int, help="Session number")
    score_parser.add_argument("--binSize", type=int, default=binSize, help="Bin size")
    score_parser.add_argument("--binNumber", type=int, default=binNumber, help="Bin number")
    score_parser.add_argument("--n", type=int, default=n, help="Number of similar units")
    score_parser.add_argument("--persistent", action="store_true", help="Use persistent homology")
    score_parser.add_argument("--plot", action="store_true", help="Plot the likelihood matrix")
    score_parser.add_argument("--max_workers", type=int, default=max_worker, help="Number of workers")
    score_parser.add_argument("--metric", type=int, choices=[0, 1], nargs=len(metric), default=metric, help="Metric to use")
    score_parser.add_argument("--trust", action="store_true", help="Do not show same pairs if rejected")
    score_parser.set_defaults(func=computeScore)
    

    args = parser.parse_args()
    args.func(args)
    
