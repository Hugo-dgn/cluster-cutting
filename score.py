import numpy as np

def lagScore(lags, corr):
    S = np.sum(corr)
    if S == 0:
        return 0
    L = np.max(lags)
    elags = np.sum(np.abs(lags)*corr)/S
    score = np.exp(-abs(elags)/L)
    return score

def similarityScore(corr1, corr2):
    norm1 = np.sum(corr1)
    norm2 = np.sum(corr2)
    if norm1 == 0 or norm2 == 0:
        return 0
    
    pdf1 = corr1/norm1
    pdf2 = corr2/norm2
    
    helinger = (0.5*np.sum((pdf1**0.5 - pdf2**0.5)**2))**0.5
    score = np.exp(-helinger)
    return score

def symmetryScore(lags, corr):
    norm = np.linalg.norm(corr)
    if norm == 0:
        return 0
    Elag = np.sum(lags*corr/np.sum(corr))
    score = np.exp(-abs(Elag)/np.max(lags))
    return score

def waveformsScore(waveforms1, waveforms2):

    catWaveform1 = waveforms1.flatten()
    catWaveform2 = waveforms2.flatten()

    cosineSimilarity = np.sum(catWaveform1*catWaveform2)/np.linalg.norm(catWaveform1)/np.linalg.norm(catWaveform2)
    theta = np.arccos(cosineSimilarity)
    score = np.exp(-abs(theta))
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
            score2 = symmetryScore(lags, corrs[i,j, :])
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