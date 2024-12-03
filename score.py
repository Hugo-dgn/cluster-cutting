import numpy as np

def lagScore(lags, crosscorr):

    pdf = crosscorr/np.sum(crosscorr)
    y = np.correlate(pdf, pdf, mode='full')
    y = y[len(pdf)-1:]
    x = np.arange(0, len(pdf))

    coeffs = np.polyfit(x, y, deg=1)  # Returns [slope, intercept]
    y_pred = np.polyval(coeffs, x)

    ss_total = np.sum((y - np.mean(y))**2)  # Total sum of squares
    ss_residual = np.sum((y - y_pred)**2)   # Residual sum of squares
    r_squared = 1 - (ss_residual / ss_total)

    score = 10*(1 - r_squared)

    return score

def waveformsScore(waveforms1, waveforms2):

    catWaveform1 = waveforms1.flatten()
    catWaveform2 = waveforms2.flatten()

    cosineSimilarity = np.sum(catWaveform1*catWaveform2)/np.linalg.norm(catWaveform1)/np.linalg.norm(catWaveform2)
    theta = np.arccos(cosineSimilarity)
    score = np.exp(-abs(theta))
    return score


def getLikelihood(lags, corrs, waveforms):
    n = corrs.shape[0]
    likelihood = np.zeros((n,n, 5))

    for i in range(n):
        for j in range(i):
            score1 = lagScore(lags, corrs[i,j, :])
            score2 = waveformsScore(waveforms[i], waveforms[j])
            likelihood[i,j, :] = score1, score2

    return likelihood

def sortScore(likelihood, n):
    k = likelihood.shape[0]
    n = np.clip(n, 0, k*(k+1)//2)
    flat_indices = np.argpartition(likelihood.flatten(), -n)[-n:]

    sorted_indices = flat_indices[np.argsort(-likelihood.flatten()[flat_indices])]
    rows, cols = np.unravel_index(sorted_indices, likelihood.shape)

    largest_values = likelihood[rows, cols]

    return largest_values, rows, cols