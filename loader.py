import os
import re
import numpy as np
import xml.etree.ElementTree as ET
from tqdm.auto import tqdm

import utils

def checkInitClu(args):
    path = args.path
    clu = rf'.*clu\.[0-9]$'
    init_clu_path = os.path.join(path, 'init_clu')
    if not os.path.exists(init_clu_path):
        print("Warning: init_clu not found")
    else:
        initclu_files = [f for f in os.listdir(init_clu_path) if re.match(clu, f)]
        clu_files = [f for f in os.listdir(args.path) if re.match(clu, f)]
        if len(initclu_files) != len(clu_files):
            print("Warning: init_clu and clu files do not match")

def getPath(args):
    checkInitClu(args)
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

def loadRes(args):
    res_path, _, _, _ = getPath(args)

    res_data = np.loadtxt(res_path, dtype=int)
    return res_data

def loadXml(args):
    _, _, _, xml_path = getPath(args)

    xml_data = ET.parse(xml_path)
    return xml_data

def loadSpikes(clu, spkIds, xml_data, args):
    _, _, spk_path, _ = getPath(args)
    nSamples, nChannels = utils.getSampleParameters(xml_data, args.session)
    indices = np.where(np.isin(clu, spkIds))[0]
    spks = np.zeros((len(indices)*nSamples*nChannels))
    delta = nSamples*nChannels
    for n, i in tqdm(enumerate(indices), total=len(indices)):
        startLine = i*nSamples*nChannels
        endLine = (i+1)*nSamples*nChannels
        offset = startLine*2
        count = endLine - startLine
        spk = np.fromfile(spk_path, dtype=np.int16, count=count, offset=offset)
        spks[n*delta:(n+1)*delta] = spk
    return spks


def loadClu(args):
    _, clu_path, _, _ = getPath(args)

    clu_data = np.loadtxt(clu_path, dtype=int)

    return clu_data