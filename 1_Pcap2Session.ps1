# Wei Wang (ww8137@mail.ustc.edu.cn)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2_0. If a copy of the MPL was not distributed with this file, You
# can obtain one at http://mozilla.org/MPL/2_0/.
# ==============================================================================

foreach($f in gci D:\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\1_Pcap *.pcap)
{
    D:\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\0_Tool\SplitCap_2-1\SplitCap -s Flow -r $f.FullName -o D:\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\2_Session\AllLayers\$($f.BaseName)-ALL
    gci D:\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\2_Session\AllLayers\$($f.BaseName)-ALL | ?{$_.Length -eq 0} | del

    D:\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\0_Tool\SplitCap_2-1\SplitCap -s Flow -r $f.FullName -o D:\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\2_Session\L7\$($f.BaseName)-L7 -y L7
    gci D:\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\2_Session\L7\$($f.BaseName)-L7 | ?{$_.Length -eq 0} | del
}

D:\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\0_Tool\finddupe -del D:\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\2_Session\AllLayers
D:\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\0_Tool\finddupe -del D:\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\2_Session\L7