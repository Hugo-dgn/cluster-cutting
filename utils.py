import numpy as np

def getUnits(clu):
    units = np.unique(clu)
    units = np.sort(units)
    units = units[units > 1]
    return units

def getSampleParameters(xml_data, session):
    root = xml_data.getroot()
    spikeDetection = root.find('spikeDetection')
    channelGroups = spikeDetection[0]
    group = channelGroups[session]
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
        groupsScore.append(groupScore/len(group))
    
    return groups, groupsScore