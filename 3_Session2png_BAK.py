# -*- coding: utf-8 -*-
# Wei Wang (ww8137@mail.ustc.edu.cn)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file, You
# can obtain one at http://mozilla.org/MPL/2.0/.
# ==============================================================================

import numpy
from PIL import Image
import binascii
import errno
import os

PNG_SIZE = 128

def getMatrixfrom_pcap(hexst,width):
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

paths = [r'D:\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\3_ProcessedSession\FilteredSession\Train',
         r'D:\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\4_Png\Train',
         r'D:\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\3_ProcessedSession\FilteredSession\Test',
         r'D:\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\4_Png\Test']



for p in range(len(paths)):
    for i, d in enumerate(os.listdir(paths[0])):
        dir_full = os.path.join(paths[1], str(i))
        mkdir_p(dir_full)
        for f in os.listdir(os.path.join(paths[0], d)):
            X1 = dir_full
            X2 = os.path.splitext(f)[0]+'.png'
            bin_full = os.path.join(paths[0], d, f)
            if os.path.getsize(bin_full) < 1e6:
                with open(bin_full, 'rb') as f:
                    content = f.read()
                content = bytearray(content)
                if len(content) <= 16384:
                    for j in range(16384-len(content)):
                        content.append(255)
                    hexst_var = binascii.hexlify(content)
                    im = Image.fromarray(getMatrixfrom_pcap(hexst_var, PNG_SIZE))
                    png_full = X1 + '\\' + X2
                    im.save(png_full)


for p in range(len(paths)):
    for i, d in enumerate(os.listdir(paths[2])):
        dir_full = os.path.join(paths[3], str(i))
        mkdir_p(dir_full)
        for f in os.listdir(os.path.join(paths[2], d)):
            X1 = dir_full
            X2 = os.path.splitext(f)[0]+'.png'
            bin_full = os.path.join(paths[2], d, f)
            if os.path.getsize(bin_full) < 1e6:
                with open(bin_full, 'rb') as f:
                    content = f.read()
                content = bytearray(content)
                if len(content) <= 16384:
                    for j in range(16384-len(content)):
                        content.append(255)
                    hexst_var = binascii.hexlify(content)
                    im = Image.fromarray(getMatrixfrom_pcap(hexst_var, PNG_SIZE))
                    png_full = X1 + '\\' + X2
                    im.save(png_full)