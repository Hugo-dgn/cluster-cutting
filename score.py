import numpy as np

def lagScore(lags, corr):
    norm = np.linalg.norm(corr)
    if norm == 0:
        return 0
    
    corr = corr/norm
    
    scoreAbs = np.sum(lags**2*corr/np.sum(corr))
    return scoreAbs

def similarityScore(corr1, corr2):
    norm1 = np.linalg.norm(corr1)
    norm2 = np.linalg.norm(corr2)
    if norm1 == 0 or norm2 == 0:
        return 0
    
    corr1 = corr1/norm1
    corr2 = corr2/norm2
    
    diffNorm = np.linalg.norm(corr1 - corr2)
    diffNormAdjusted = 2*diffNorm/(np.linalg.norm(corr1) + np.linalg.norm(corr2))
    score = np.exp(-diffNormAdjusted)
    return score

def symmetryScore(corr):
    norm = np.linalg.norm(corr)
    if norm == 0:
        return 0
    
    corr = corr/norm

    mirror = np.flip(corr)
    mirrorNorm = np.linalg.norm(corr - mirror)
    mirrorNormAdjusted = mirrorNorm/np.linalg.norm(corr)
    score = np.exp(-mirrorNormAdjusted)
    return score

def waveformsScore(waveforms1, waveforms2):

    
    normalisation = np.mean(waveforms1**2, axis=0) + np.mean(waveforms2**2, axis=0)
    third_quartile = np.percentile(normalisation, 75)
    significant = normalisation > third_quartile

    nmse = np.mean((waveforms1 - waveforms2)**2, axis=0)/normalisation
    nmse = np.mean(nmse[significant])

    score = np.exp(-nmse)
    return score

def spikeChannelDistanceScore(waveforms1, waveforms2):
    CenteredWaveforms1 = waveforms1 - np.mean(waveforms1, axis=0)
    CenteredWaveforms2 = waveforms2 - np.mean(waveforms2, axis=0)
    norm1 = np.linalg.norm(CenteredWaveforms1, axis=0)
    norm2 = np.linalg.norm(CenteredWaveforms2, axis=0)
    largestPeakDistance = np.abs(np.argmax(norm1) - np.argmax(norm2))
    score = np.exp(-largestPeakDistance)
    return score


def getLikelihood(lags, corrs, waveforms):
    n = corrs.shape[0]
    likelihood = np.zeros((n,n, 5))

    for i in range(n):
        for j in range(i):
            score1 = lagScore(lags, corrs[i,j, :])
            score2 = symmetryScore(corrs[i,j, :])
            score3 = similarityScore(corrs[i,i, :], corrs[j,j, :])
            score4 = waveformsScore(waveforms[i], waveforms[j])
            score5 = spikeChannelDistanceScore(waveforms[i], waveforms[j])
            likelihood[i,j, :] = score1, score2, score3, score4, score5

    return likelihood

def sortScore(likelihood, n):
    k = likelihood.shape[0]
    n = np.clip(n, 0, k*(k+1)//2)
    flat_indices = np.argpartition(likelihood.flatten(), -n)[-n:]

    sorted_indices = flat_indices[np.argsort(-likelihood.flatten()[flat_indices])]
    rows, cols = np.unravel_index(sorted_indices, likelihood.shape)

    largest_values = likelihood[rows, cols]

    return largest_values, rows, cols