import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
import xml.etree.ElementTree as ET

import compute
import score
import utils

def getPath(args):
    path = args.path
    session = args.session

    clu = rf'.*clu\.{session}$'
    res = rf'.*res\.{session}$'
    spk = rf'.*spk\.{session}$'
    xml = rf'Rat[0-9]*_[0-9]*\.xml$'

    clu_files = [f for f in os.listdir(path) if re.match(clu, f)][0]
    res_files = [f for f in os.listdir(path) if re.match(res, f)][0]
    spk_files = [f for f in os.listdir(path) if re.match(spk, f)][0]
    xml_files = [f for f in os.listdir(path) if re.match(xml, f)][0]

    clu_path = os.path.join(path, clu_files)
    res_path = os.path.join(path, res_files)
    spk_path = os.path.join(path, spk_files)
    xml_path = os.path.join(path, xml_files)

    return res_path, clu_path, spk_path, xml_path

def load(args):
    res_path, clu_path, spk_path, xml_path = getPath(args)

    res_data = np.loadtxt(res_path, dtype=int)
    clu_data = np.loadtxt(clu_path, dtype=int)
    spk_data = np.fromfile(spk_path, dtype=np.int16)
    xml_data = ET.parse(xml_path)

    return res_data, clu_data, spk_data, xml_data

def loadClu(args):
    _, clu_path, _, _ = getPath(args)

    clu_data = np.loadtxt(clu_path, dtype=int)

    return clu_data

def plotWaveforms(spk, normalization):
    spk = spk - spk.mean(axis=0)
    spk = spk/normalization
    spk = spk[:,::-1]

    for i in range(spk.shape[1]):
        plt.plot(spk[:, i] + i)

def getScore(lags, crosscorr1, corr1, corr2, waveforms1, waveforms2):
    score1 = score.lagScore(lags, crosscorr1)
    score2 = score.simetryScore(crosscorr1)
    score3 = score.similarityScore(corr1, corr2)
    score4 = score.waveformsScore(waveforms1, waveforms2)
    return score1, score2, score3, score4
    


def plotSingle(args):
    res_data, clu_data, spk_data, xml_data = load(args)
    clu_data = clu_data[1:]
    units = utils.getUnits(clu_data)
    lags, crosscorr1 = compute.getSingleCrossCorr(res_data, clu_data, args.ref, args.target, args.binSize, args.binNumber)
    lags, corr1 = compute.getSingleCrossCorr(res_data, clu_data, args.ref, args.ref, args.binSize, args.binNumber)
    lags, corr2 = compute.getSingleCrossCorr(res_data, clu_data, args.target, args.target, args.binSize, args.binNumber)
    
    waveforms = compute.computeWaveforms(clu_data, spk_data, xml_data, args.session)
    waveforms1 = waveforms[np.where(units == args.ref)[0][0]]
    waveforms2 = waveforms[np.where(units == args.target)[0][0]]

    score1, score2, score3, score4 = getScore(lags, crosscorr1, corr1, corr2, waveforms1, waveforms2)
    print(f"Score 1: {score1}\nScore 2: {score2}\nScore 3: {score3}\nScore 4: {score4}\nscore: {score1*score2*score3*score4}")

    spks = compute.computeWaveforms(clu_data, spk_data, xml_data, args.session)

    units = utils.getUnits(clu_data)
    indexUnitref = np.where(units == args.ref)[0][0]
    indexUnittarget = np.where(units == args.target)[0][0]

    spk1 = spks[indexUnitref]
    spk2 = spks[indexUnittarget]
    normalisation = max(np.max(spk1 - spk1.mean(axis=0)), np.max(spk2 - spk2.mean(axis=0)))
    plt.figure()
    plt.subplot(221)
    plotWaveforms(spk1, normalisation)
    plt.subplot(222)
    plotWaveforms(spk2, normalisation)
    plt.subplot(223)
    plt.bar(lags, crosscorr1)
    plt.show()

def computeScore(args):
    flag = True
    lastUnits = np.array([])
    lastCorr = np.array([])
    lastCorr = lastCorr.reshape((0, 0, 2*args.binNumber))
    res_data, clu_data, spk_data, xml_data = load(args)
    while flag:
        clu_data = loadClu(args)
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

        largest_values, rows, cols = score.sortScore(likelihood, args.n)

        graph, linkScore = utils.computeGraph(largest_values, rows, cols)
        
        
        groups, groupsScore = utils.getConnectedComponents(graph, linkScore)
        
        sorted_indices = np.argsort(groupsScore)[::-1]
        groupsScore = [groupsScore[i] for i in sorted_indices]
        groups = [groups[i] for i in sorted_indices]

        for groupScore, group in zip(groupsScore, groups):
            grouped_units = tuple(int(u) for u in units[list(group)])
            grouped_units = sorted(grouped_units)
            print(f"Units: {grouped_units} | Score: {groupScore:.1f}")


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
                elif key == "q":
                    flag = False
                    escape = True
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
    single_parser.add_argument("--binSize", type=int, default=10, help="Bin size")
    single_parser.add_argument("--binNumber", type=int, default=20, help="Bin number")
    single_parser.set_defaults(func=plotSingle)

    score_parser = subparsers.add_parser("score", help="Compute score")
    score_parser.add_argument("path", type=str, help="Path to the data")
    score_parser.add_argument("session", type=int, help="Session number")
    score_parser.add_argument("--binSize", type=int, default=10, help="Bin size")
    score_parser.add_argument("--binNumber", type=int, default=20, help="Bin number")
    score_parser.add_argument("--n", type=int, default=10, help="Number of similar units")
    score_parser.add_argument("--persistent", action="store_true", help="Use persistent homology")
    score_parser.add_argument("--plot", action="store_true", help="Plot the likelihood matrix")
    score_parser.add_argument("--max_workers", type=int, default=16, help="Number of workers")
    score_parser.add_argument("--metric", type=int, choices=[0, 1], nargs=4, default=[1, 1, 1, 1], help="Metric to use")
    score_parser.set_defaults(func=computeScore)
    

    args = parser.parse_args()
    args.func(args)
    
