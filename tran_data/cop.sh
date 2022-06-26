#! /bin/bash

subs=(Cli Codec Collections Compress  Csv JacksonCore JacksonXml Jsoup JxPath)

for sub in ${subs[@]}
do
cp /data/zqh/${sub}cover.pkl .
done

