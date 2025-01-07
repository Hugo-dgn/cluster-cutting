import numpy as np

def CrosscorrScore(corr1, corr2, crosscorr):
    n1 = np.linalg.norm(corr1)
    n2 = np.linalg.norm(corr2)
    n = np.linalg.norm(crosscorr)
    if n1*n2*n == 0:
        return 0
    cosineSimilarity1 = np.sum(corr1*crosscorr)/n1/n
    cosineSimilarity2 = np.sum(corr2*crosscorr)/n2/n

    score = max(cosineSimilarity1, cosineSimilarity2)

    return score

def similaritySocre(corr1, corr2):
    cosineSimilarity = np.sum(corr1*corr2)/np.linalg.norm(corr1)/np.linalg.norm(corr2)
    score = cosineSimilarity
    return score

def waveformsScore(waveforms1, waveforms2):

    wave1 = np.diff(waveforms1)
    wave2 = np.diff(waveforms2)

    catWaveform1 = wave1.flatten()
    catWaveform2 = wave2.flatten()

    xx = catWaveform1 - np.mean(catWaveform1)
    yy = catWaveform2 - np.mean(catWaveform2)

    corr = np.sum(xx*yy)/np.linalg.norm(xx)/np.linalg.norm(yy)
    
    score = corr
    return score


def getLikelihood(lags, corrs, waveforms):
    n = corrs.shape[0]
    likelihood = np.zeros((n,n, 3))

    for i in range(n):
        for j in range(i):
            score1 = CrosscorrScore(corrs[i, i], corrs[j, j], corrs[i,j])
            score2 = similaritySocre(corrs[i, i], corrs[j, j])
            score3 = waveformsScore(waveforms[i], waveforms[j])
            likelihood[i,j, :] = score1, score2, score3

    return likelihood

def sortScore(likelihood, n):
    k = likelihood.shape[0]
    n = np.clip(n, 0, k*(k+1)//2)
    flat_indices = np.argpartition(likelihood.flatten(), -n)[-n:]

    sorted_indices = flat_indices[np.argsort(-likelihood.flatten()[flat_indices])]
    rows, cols = np.unravel_index(sorted_indices, likelihood.shape)

    largest_values = likelihood[rows, cols]
    return largest_values, rows, cols