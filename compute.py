import numpy as np
from tqdm.auto import tqdm
import concurrent.futures

import utils

def getBins(time, binSize, binNumber):
    edge = np.arange(-binNumber, binNumber+1)*binSize
    edges = time + edge[:, np.newaxis]
    lags = np.arange(-binNumber, binNumber)
    return lags, edges

def computeCrossCorr(timeTarget, edges, auto):

    corr = np.searchsorted(timeTarget, edges)
    corr = np.diff(corr, axis=0)
    corr = np.sum(corr, axis=1)

    if auto:
        corr[len(edges)//2] -= len(timeTarget)

    return corr

def getSingleCrossCorr(res, clu, ref, target, binSize, binNumber):
    spikesRef = clu == ref
    spikesTarget = clu == target
    timeRef = res[spikesRef]
    timeTarget = res[spikesTarget]
    
    lags, edges = getBins(timeRef, binSize, binNumber)
    corr = computeCrossCorr(timeTarget,edges, ref == target)
    return lags, corr

def computeWaveforms(clu_data, spk_data, xml_data, session):
    units = utils.getUnits(clu_data)
    nSamples, nChannels = utils.getSampleParameters(xml_data, session)
    spk = spk_data.reshape(-1, nSamples, nChannels)
    waveforms = np.zeros((len(units), nSamples, nChannels))
    for i, unit in enumerate(units):
        waveforms[i] = spk[clu_data == unit].mean(axis=0)
    return waveforms

def computeCrossCorrWrapper(timeTarget, edges, auto, index):
    corr = computeCrossCorr(timeTarget, edges, auto)
    return index, corr

def computeAllCrossCorr(res, clu, binSize, binNumber, max_workers, restrict = None):
    units = utils.getUnits(clu)
    if restrict is None:
        restrict = units
    n = len(units)
    spikes = []
    edges = [None for i in range(n)]

    for unit in tqdm(units, desc="sorting spikes"):
        spike = clu == unit
        time = res[spike]
        spikes.append(time)
    
    k = n*(n+1)//2
    futures = []

    
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=k, desc="Sending jobs") as pbar:
            for i in range(n):
                time1 = spikes[i]
                edge = edges[i]
                for j in range(i+1):
                    if units[i] in restrict or units[j] in restrict:
                        time2 = spikes[j]
                        if len(time1) <= len(time2):
                            edge = edges[i]
                            if edge is None:
                                lags, edge = getBins(time1, binSize, binNumber)
                                edges[i] = edge
                            futures.append(executor.submit(computeCrossCorrWrapper, time2, edge, i==j, (i,j)))
                        else:
                            edge = edges[j]
                            if edge is None:
                                lags, edge = getBins(time1, binSize, binNumber)
                                edges[j] = edge
                            futures.append(executor.submit(computeCrossCorrWrapper, time1, edge, i==j, (i,j)))
                    pbar.update(1)


        corrs = np.zeros((n,n,2*binNumber))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
            (i,j), result = future.result()

            corrs[i,j] = result

    lags = np.arange(-binNumber, binNumber)
    return lags, corrs
            