# Wei Wang (ww8137@mail.ustc.edu.cn)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2_0. If a copy of the MPL was not distributed with this file, You
# can obtain one at http://mozilla.org/MPL/2_0/.
# ==============================================================================

foreach($f in gci D:\Users\shaya\Desktop\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\1_Pcap *.pcap)
{
    D:\Users\shaya\Desktop\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\0_Tool\SplitCap_2-1\SplitCap -p 50000 -b 50000 -r $f.FullName -o D:\Users\shaya\Desktop\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\2_Session\AllLayers\$($f.BaseName)-ALL
    # 0_Tool\SplitCap_2-1\SplitCap -p 50000 -b 50000 -r $f.FullName -s flow -o 2_Session\AllLayers\$($f.BaseName)-ALL
    gci D:\Users\shaya\Desktop\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\2_Session\AllLayers\$($f.BaseName)-ALL | ?{$_.Length -eq 0} | del

    D:\Users\shaya\Desktop\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\0_Tool\SplitCap_2-1\SplitCap -p 50000 -b 50000 -r $f.FullName -o D:\Users\shaya\Desktop\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\2_Session\L7\$($f.BaseName)-L7 -y L7
    # 0_Tool\SplitCap_2-1\SplitCap -p 50000 -b 50000 -r $f.FullName -s flow -o 2_Session\L7\$($f.BaseName)-L7 -y L7
    gci D:\Users\shaya\Desktop\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\2_Session\L7\$($f.BaseName)-L7 | ?{$_.Length -eq 0} | del
}

D:\Users\shaya\Desktop\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\0_Tool\finddupe -del D:\Users\shaya\Desktop\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\2_Session\AllLayers
D:\Users\shaya\Desktop\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\0_Tool\finddupe -del D:\Users\shaya\Desktop\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\2_Session\L7