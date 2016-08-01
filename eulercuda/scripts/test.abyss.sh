#   ABYSS test script
#
#
#
#	$1 in folder
#	$2 orig
#	$3

timeStamp=`date +%Y%m%dT%H%M%S`
appPrefix='abyss'
dirPrefix=$appPrefix.$timeStamp
outputDir=./$dirPrefix
abyssApp=/home/smahmood/local/bin/ABySS/bin/ABYSS
inputDir=$1
origDir=$2
seqK=`seq 13 1 32`
infoFile="$outputDir/README"
logFile="$outputDir/output.abyss.log"
statFile="$outputDir/output.abyss.$timeStamp.xml"

#echo $dirPrefix > $infoFile
#seqK="10 20 30"


mkdir $outputDir -p					#create output directory


echo "OutputDir	: $dirPrefix ">> $infoFile	
echo "TimeStamp : $timeStamp" >>$infoFile
infoK=`echo $seqK | tr \n\r \t` 
echo "k	: $infoK" >>$infoFile
echo "file	:" >> $infoFile

cp $0 $outputDir


echo "<Application name='ABySS'>" >>$statFile
for f in $inputDir/*.fas*					# iterate over each input
do
	fileBaseName=`basename $f`					#get fileName
	echo "	$fileBaseName" >>$infoFile
	cp $f $outputDir						#copy to output folder
	cp $origDir/orig.$fileBaseName $outputDir
	
	for kmer in $seqK 
	do
	#	echo "$k $fileBaseName"
		
		echo "<Result id='$timeStamp.$kmer.$fileBaseName' method='ABySS' datasetName='$fileBaseName' params='l=$kmer, file=$fileBaseName' l='$kmer'>" >>$statFile
		outputFile="$outputDir/output.abyss.k$kmer.$fileBaseName"
		outputFileBaseName=`basename $outputFile`
			
		T1=`date +%s.%N`
		$abyssApp  -k$kmer -o $outputFile  $f &>> $logFile		# Run ABySS
		T2=`date +%s.%N`
		
		totalTime=`echo "scale=3;( $T2 - $T1)/1"|bc -l`			#this division is wierd :-/
		echo "<item name='Total Time' value='$totalTime' />" >>$statFile
		
		perl n50.pl $outputFile 1>>$statFile

		for matchLength in 10 50 100
		do
			nucmerOutputFile=$outputDir/nucmer.$matchLength.$outputFileBaseName
			nucmer -maxmatch -c $matchLength -p $nucmerOutputFile   $origDir/orig.$fileBaseName   $outputFile &>> $logFile
			show-coords -r -c -l $nucmerOutputFile.delta 1> $nucmerOutputFile.coords

			accTotal=`awk ' NR>5 { if(a[$19]<$16) {a[$19]=$16}} END {{j=0}for (i in a) { j+=a[i]  } {printf "%f\n", j} } ' $nucmerOutputFile.coords`
			totalContig=`awk '/^>.*/ {i++} END { print i}' $outputFile`
			weightedAccTotal=`awk ' NR>5 { if(a[$19]<$16*$13) {a[$19]=$16*$13}} END {{j=0}for (i in a) { j+=a[i]  } {printf "%f\n", j} } ' $nucmerOutputFile.coords`
			weightedTotalContig=`awk '/^[^>].*/ {i+=length($0)} END { print i}' $outputFile`

		#	accTotal=`awk`
		#	echo "$accTotal   $totalContig"
			if [ "${accTotal+x}" = x ] && [ -z "$accTotal" ] ;then
                        	accTotal=0
                         fi
	
			accuracy=`echo "scale=3; $accTotal/$totalContig" | bc -l`
			weightedAccuracy=`echo "scale=3; $weightedAccTotal/$weightedTotalContig" | bc -l`

			echo "<item name='Accuracy$matchLength' value='$accuracy'/>" >>$statFile 
			echo "<item name='WeightedAccuracy$matchLength' value='$weightedAccuracy'/>" >>$statFile 
			
		done
		echo "</Result>" >> $statFile
	done 
	
done

echo "</Application>" >>$statFile


