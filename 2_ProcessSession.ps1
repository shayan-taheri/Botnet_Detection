# Wei Wang (ww8137@mail.ustc.edu.cn)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2_0. If a copy of the MPL was not distributed with this file, You
# can obtain one at http://mozilla.org/MPL/2_0/.
# ==============================================================================

$SESSIONS_COUNT_LIMIT_MIN = 0
$SESSIONS_COUNT_LIMIT_MAX = 60000
$TRIMED_FILE_LEN = 16384
#$SOURCE_SESSION_DIR = "Flow\L7"

$SOURCE_SESSION_DIR = "D:\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\2_Session\AllLayers"

echo "If Sessions more than $SESSIONS_COUNT_LIMIT_MAX we only select the largest $SESSIONS_COUNT_LIMIT_MAX."
echo "Finally Selected Sessions:"

$dirs = gci $SOURCE_SESSION_DIR -Directory
foreach($d in $dirs)
{
    $files = gci $d.FullName
    $count = $files.count
    if($count -gt $SESSIONS_COUNT_LIMIT_MIN)
    {             
        echo "$($d.Name) $count"        
        if($count -gt $SESSIONS_COUNT_LIMIT_MAX)
        {
            $files = $files | sort Length -Descending | select -First $SESSIONS_COUNT_LIMIT_MAX
            $count = $SESSIONS_COUNT_LIMIT_MAX
        }

        $files = $files | resolve-path
        $test  = $files | get-random -count ([int]($count/10))
        $train = $files | ?{$_ -notin $test}     

        $path_test  = "D:\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\3_ProcessedSession\FilteredSession\Test\$($d.Name)"
        $path_train = "D:\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\3_ProcessedSession\FilteredSession\Train\$($d.Name)"
        ni -Path $path_test -ItemType Directory -Force
        ni -Path $path_train -ItemType Directory -Force    

        cp $test -destination $path_test        
        cp $train -destination $path_train
    }
}

echo "All files will be trimed to $TRIMED_FILE_LEN length and if it's even shorter we'll fill the end with 0x00..."

$paths = @(('D:\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\3_ProcessedSession\FilteredSession\Train', 'D:\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\3_ProcessedSession\TrimmedSession\Train'), ('D:\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\3_ProcessedSession\FilteredSession\Test', 'D:\DeepTraffic-master\DeepTraffic-master\1_malware_traffic_classification\2_PreprocessedTools\3_ProcessedSession\TrimmedSession\Test'))
foreach($p in $paths)
{
    foreach ($d in gci $p[0] -Directory) 
    {
        ni -Path "$($p[1])\$($d.Name)" -ItemType Directory -Force
        foreach($f in gci $d.fullname)
        {
            if($f.length -le 16384) # 128 X 128
	        {
               $content = [System.IO.File]::ReadAllBytes($f.FullName)
               $len = $f.length - $TRIMED_FILE_LEN
               if($len -gt 0)
               {        
                   $content = $content[0..($TRIMED_FILE_LEN-1)]        
               }
               elseif($len -lt 0)
               {        
                   # $padding = [Byte[]] (,0x00 * ([math]::abs($len)))
                   $padding = [Byte[]] (,0xff * ([math]::abs($len)))
                   $content = $content += $padding
               }
               Set-Content -value $content -encoding byte -path "$($p[1])\$($d.Name)\$($f.Name)"
            }
        }        
    }
}