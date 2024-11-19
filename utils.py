import numpy as np

def smooth(x, kernel_size, sigma):
    kernel = np.exp(-np.arange(-kernel_size, kernel_size+1)**2/(2*sigma**2))
    kernel = kernel/np.sum(kernel)
    return np.convolve(x, kernel, mode='same')
    

def getUnits(clu):
    units = np.unique(clu)
    units = np.sort(units)
    units = units[units > 1]
    return units

def getSampleParameters(xml_data, session):
    root = xml_data.getroot()
    spikeDetection = root.find('spikeDetection')
    channelGroups = spikeDetection[0]
    group = channelGroups[session-1]
    nSamples = int(group.find('nSamples').text)
    nChannels = len(group.find('channels'))
    return nSamples, nChannels

def computeGraph(largest_values, rows, cols):
    graph = {}
    linkScore = {}

    for value, (i,j) in zip(largest_values, zip(rows, cols)):
        i, j = int(i), int(j)
        if i not in graph:
            graph[i] = set()
        if j not in graph:
            graph[j] = set()
        graph[i].update([j])
        graph[j].update([i])
        linkScore[(i, j)] = value
        linkScore[(j, i)] = value
    
    return graph, linkScore

def getConnectedComponents(graph, linkScore):
    visited = {i : False for i in graph}
    next = [i for i in graph]
    groups = []
    groupsScore = []
    while len(next) > 0:
        start = next.pop(0)
        if visited[start]:
            continue
        queue = [start]
        visited[start] = True
        group = []
        groupScore = 0
        while len(queue) > 0:
            node = queue.pop(0)
            group.append(node)
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    groupScore += linkScore[(node, neighbor)]
                    visited[neighbor] = True
                    queue.append(neighbor)
        groups.append(group)
        groupsScore.append(groupScore/(len(group)-1))
    
    return groups, groupsScore

def getPairsFromGroups(units, groups):
    pairs = []
    for group in groups:
        n = len(group)
        for i in range(n):
            for j in range(i):
                a = units[group[i]]
                b = units[group[j]]
                x = min(a, b)
                y = max(a, b)
                pairs.append((x, y))
    return pairs