#!/bin/sh

#set j= 'echo "$1" | sed "s/\(^[A-Za-z0-9]\+\(\.[A-Za-z0-9-]\+\)*\)\.[a-zA-Z]\+$/\1 /" '
#echo $j
nucmer -maxmatch -c $3 -p nucmer.$2 $1 $2

echo " -maxmatch -c $3 -p nucmer.$2 $1 $2"

show-coords -r -c -l nucmer.$2.delta 1>nucmer.$2.coords

echo " -r -c -l nucmer.$2.delta > nucmer.$2.coords"

mapview -n 1 -f pdf -p mapview.nucmer.$2  nucmer.$2.coords

echo " -n 1 -f pdf -p mapview.nucmer.$2  nucmer.$2.coords"

mummerplot -postscript -p mummerplot.nucmer.$2 nucmer.$2.delta

echo " -postscript -p mummerplot.nucmer.$2 nucmer.$2.delta"

sed -e "s/w lp/w lines/g" mummerplot.nucmer.$2.gp 1> tmpfile && mv tmpfile mummerplot.nucmer.$2.gp

gnuplot mummerplot.nucmer.$2.gp

