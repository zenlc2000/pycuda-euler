#!/bin/sh


# $1 Reference
# $2 Query
# $3 match  length 100,50, 10 etc


#set j= 'echo "$1" | sed "s/\(^[A-Za-z0-9]\+\(\.[A-Za-z0-9-]\+\)*\)\.[a-zA-Z]\+$/\1 /" '
#echo $j
nucmer -maxmatch -f -c $3 -p nucmer.f.$1 $1 $2

echo " -maxmatch -f -c $3 -p nucmer.f.$1 $1 $2"

show-coords -r -c -l nucmer.f.$1.delta 1>nucmer.f.$1.coords

echo " -r -c -l nucmer.f.$1.delta > nucmer.f.$1.coords"

mapview -n 1 -f pdf -p mapview.nucmer.f.$1  nucmer.f.$1.coords

echo " -n 1 -f pdf -p mapview.nucmer.f.$1  nucmer.f.$1.coords"

mummerplot -postscript -p mummerplot.nucmer.f.$1 nucmer.f.$1.delta

echo " -postscript -p mummerplot.nucmer.f.$1 nucmer.f.$1.delta"

sed -e "s/w lp/w lines/g" mummerplot.nucmer.f.$1.gp 1> tmpfile && mv tmpfile mummerplot.nucmer.f.$1.gp

gnuplot mummerplot.nucmer.f.$1.gp


nucmer -maxmatch -r -c $3 -p nucmer.r.$1 $1 $2

echo " -maxmatch -r -c $3 -p nucmer.r.$1 $1 $2"

show-coords -r -c -l nucmer.r.$1.delta 1>nucmer.r.$1.coords

echo " -r -c -l nucmer.r.$1.delta > nucmer.r.$1.coords"

mapview -n 1 -f pdf -p mapview.nucmer.r.$1  nucmer.r.$1.coords

echo " -n 1 -f pdf -p mapview.nucmer.r.$1  nucmer.r.$1.coords"

mummerplot -postscript -p mummerplot.nucmer.r.$1 nucmer.r.$1.delta

echo " -postscript -p mummerplot.nucmer.r.$1 nucmer.r.$1.delta"

sed -e "s/w lp/w lines/g" mummerplot.nucmer.r.$1.gp 1> tmpfile && mv tmpfile mummerplot.nucmer.r.$1.gp

gnuplot mummerplot.nucmer.r.$1.gp


