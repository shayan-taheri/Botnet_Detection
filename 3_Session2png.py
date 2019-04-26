# -*- coding: utf-8 -*-
# Wei Wang (ww8137@mail.ustc.edu.cn)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file, You
# can obtain one at http://mozilla.org/MPL/2.0/.
# ==============================================================================

from PIL import Image
import binascii
import errno
import os
import numpy
import matplotlib.pyplot as plt
PNG_SIZE = 54

def getMatrixfrom_pcap(filename,width):
    with open(filename, 'rb') as f:
        content = f.read()
    hexst = binascii.hexlify(content)
    print(len(hexst))
    fh = numpy.array([int(hexst[i:i+2], 16) for i in range(0, len(hexst), 2)])
    rn = len(fh) / width
    fh = numpy.reshape(fh[:round(rn)*width], (-1, width))
    fh = numpy.uint8(fh)
    return fh

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

paths = [r'D:\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\3_ProcessedSession\TrimmedSession\Train',
         r'D:\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\4_Png\Train',
         r'D:\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\3_ProcessedSession\TrimmedSession\Test',
         r'D:\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\4_Png\Test']

my_list = []

for p in range(len(paths)):
    for i, d in enumerate(os.listdir(paths[0])):
        dir_full = os.path.join(paths[1], str(i))
        mkdir_p(dir_full)
        for f in os.listdir(os.path.join(paths[0], d)):
            bin_full = os.path.join(paths[0], d, f)

            with open(bin_full, 'rb') as f:
                content = f.read()

            hexst = binascii.hexlify(content)
            my_list.append(len(hexst))
            np_list = numpy.array(my_list)
            print(len(hexst))

print(len(np_list))

numpy.savetxt(r'D:\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\file.txt',np_list)
