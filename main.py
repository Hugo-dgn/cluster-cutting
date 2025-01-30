import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
from tqdm.auto import tqdm


import compute
import score
import utils
import loader
import clustering

# Default values

binSize = 30
binNumber = 20
n = 20
max_worker = 8
metric=[1, 1, 1]

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

    channel = utils.getChannels(spk[None, :])[0]

    for i in range(spk.shape[1]):
        if i == channel:
            plt.plot(spk[:, i] + i, linewidth=4)
        else:
            plt.plot(spk[:, i] + i)

def getScore(lags, crosscorr, corr1, corr2, waveforms1, waveforms2):
    score1 = score.CrosscorrScore(corr1, corr2, crosscorr)
    score2 = score.similaritySocre(corr1, corr2)
    score3 = score.waveformsScore(waveforms1, waveforms2)
    return score1, score2, score3

def plotSingle(args):
    #only plot two units : waveforms and correlograms
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
    indexUnitref = np.where(units == args.ref)[0][0]
    indexUnittarget = np.where(units == args.target)[0][0]
    
    lags, crosscorr1 = compute.getSingleCrossCorr(res_data, clu_data, args.ref, args.target, args.binSize, args.binNumber)
    lags, corr1 = compute.getSingleCrossCorr(res_data, clu_data, args.ref, args.ref, args.binSize, args.binNumber)
    lags, corr2 = compute.getSingleCrossCorr(res_data, clu_data, args.target, args.target, args.binSize, args.binNumber)
    
    waveforms = compute.computeWaveforms(clu_data, spk_data, xml_data, args.session)
    
    waveforms1 = waveforms[indexUnitref]
    waveforms2 = waveforms[indexUnittarget]

    scores = getScore(lags, crosscorr1, corr1, corr2, waveforms1, waveforms2)
    score = 1
    for i, _score in enumerate(scores):
        score *= _score**args.metric[i]
    
    score1, score2, score3 = scores
    score = 100*score1**args.metric[0]*score2**args.metric[1]*score3**args.metric[2]
    print(f"Refractory: {score1}\nSimilarity: {score2}\nWaveforms: {score3}\nScore: {score:.1f}")

    normalisation = max(np.max(waveforms1 - waveforms1.mean(axis=0)), np.max(waveforms2 - waveforms2.mean(axis=0)))

    pdf = crosscorr1/np.sum(crosscorr1)

    plt.figure()
    plt.subplot(221)
    plotWaveforms(waveforms1, normalisation)
    plt.subplot(222)
    plotWaveforms(waveforms2, normalisation)
    plt.subplot(223)
    plt.bar(lags, pdf)
    plt.subplot(224)
    y = np.correlate(crosscorr1, crosscorr1, mode='full')
    y = y/max(y)
    y = y[len(crosscorr1):]
    corrrection = np.arange(1, len(crosscorr1))[::-1]
    y = y/corrrection
    y = y/np.mean(y)
    x = np.arange(1, len(crosscorr1))

    plt.bar(x, y)
    plt.show()

def computeClusters(args):
    #use clusterisation method on the waveforms to find potential cluster merge
    print("Loading data")
    _, clu_data, spk_data, xml_data = loader.load(args)
    n_units = clu_data[0]
    max_number_custer = n_units - 2
    clu_data = clu_data[1:]
    waveforms = compute.computeWaveforms(clu_data, spk_data, xml_data, args.session)

    list_labels = []
    inertias = []
    k_values = range(2, max_number_custer)

    X, reduced_ratio, principal_components = clustering.waveforms_pca(waveforms, args.var)

    print(f"Reduced ratio: {reduced_ratio}")
    for k in tqdm(k_values, desc="Clustering"):
        labels, inertia = clustering.cluster(X, k)
        list_labels.append(labels)
        inertias.append(inertia)
    
    print(f"Number of units: {n_units}")

    plt.figure()
    plt.plot(k_values, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.show()

    k = int(input("Choose the number of clusters: "))

    labels = list_labels[int(k)-2]
    units = utils.getUnits(clu_data)
    for i in range(k):
        same = units[labels == i]
        if len(same) > 1:
            same = np.sort(same)
            print(f"Cluster {i}: {same}")

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
        print("Loading .clu file")
        clu_data = loader.loadClu(args)
        print("Computing")
        clu_data = clu_data[1:]
        units = utils.getUnits(clu_data)
        n = len(units)

        #find which units did not change from last iteration (for the --trust flag)
        sameIndex = np.isin(units, lastUnits)
        reusedIndex = np.isin(lastUnits, units)
        restrict = units[~sameIndex]
        lastUnits = units
        
        #compute cross-corelofram
        lags, newCorrs = compute.computeAllCrossCorr(res_data, clu_data, args.binSize, args.binNumber, args.max_workers, restrict)
        waveforms = compute.computeWaveforms(clu_data, spk_data, xml_data, args.session)

        corrs = np.zeros((n, n, 2*args.binNumber))

        old = np.outer(sameIndex, sameIndex) #same units as last iteration
        reused = np.outer(reusedIndex, reusedIndex) #units for which the cross-corelogram was already computed

        corrs[old, :] = lastCorr[reused, :]
        corrs[~old, :] = newCorrs[~old, :]

        lastCorr = corrs

        likelihood = score.getLikelihood(lags, corrs, waveforms)
        for i in range(len(args.metric)):
            likelihood[:, :, i] = likelihood[:, :, i]**args.metric[i]

        likelihood = np.prod(likelihood, axis=2)


        if args.trust:
            #remove pairs that were rejected in the last iteration
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

            pairs = utils.getPairsFromGroups(units, groups) #get all the recommended pairs
            for pair in pairs:
                memory.append(pair) #store them so they are not recomputed if --trust is used

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

def notes(args):
    graph = {}
    linkScore = {}
    flag = True
    while flag:
        command = input("candidate merge:")

        if command == "q":
            flag = False
        elif command == "": #if enter is hit
            groups, _ = utils.getConnectedComponents(graph, linkScore)
            for group in groups:
                print(sorted(group))
            graph = {}
            linkScore = {}
        else:
            #there is two ways to input a pair : "i j" or "i-j"
            pair1 = command.split(" ")
            pair2 = command.split("-")
            if len(pair1) == 2:
                pair = pair1
            elif len(pair2) == 2:
                pair = pair2
            else:
                print("Invalid command")
                continue

            #adds the pair to the graph
            i, j = [int(p) for p in pair]
            if i not in graph:
                graph[i] = set()
            if j not in graph:
                graph[j] = set()
            graph[i].update([j])
            graph[j].update([i])
            linkScore[(i, j)] = 1
            linkScore[(j, i)] = 1

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

    cluster_parser = subparsers.add_parser("cluster", help="Cluster waveforms")
    cluster_parser.add_argument("path", type=str, help="Path to the data")
    cluster_parser.add_argument("session", type=int, help="Session number")
    cluster_parser.add_argument("--var", type=float, default=0.99, help="Variance")
    cluster_parser.set_defaults(func=computeClusters)

    notes_parser = subparsers.add_parser("notes", help="Notes")
    notes_parser.set_defaults(func=notes)
    

    args = parser.parse_args()
    args.func(args)
    
