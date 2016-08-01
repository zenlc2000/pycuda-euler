#   CUDA test script
#
#
#
#	$1 in folder
#	$2 orig
#	$3

timeStamp=`date +%Y%m%dT%H%M%S`
appPrefix='EulerCuda'
dirPrefix=$appPrefix.$timeStamp
outputDir=./$dirPrefix
eulerCudaApp=/home/smahmood/test/pruned-3-dataset-hashed-k64/eulercuda/bin/linux/release/eulercuda
inputDir=$1
origDir=$2
seqK=`seq 14 2 32`
seqBlock="1 2 4 8 16 32 64 128 256 512"

infoFile="$outputDir/README"
logFile="$outputDir/output.eulercuda.log"
statFile="$outputDir/output.eulercuda.$timeStamp.xml"

#echo $dirPrefix > $infoFile
#seqK="10 20 30"
#seqBlock="512"


mkdir $outputDir -p					#create output directory


echo "OutputDir	: $dirPrefix ">> $infoFile	
echo "TimeStamp : $timeStamp" >>$infoFile
infoK=`echo $seqK | tr \n\r \t` 
echo "k	: $infoK" >>$infoFile
echo "block	: $seqBlock" >>$infoFile
echo "file	:" >> $infoFile

cp $0 $outputDir


echo "<Application name='EulerCUDA'>" >>$statFile
for f in $inputDir/nm*.fas*					# iterate over each input
do
	fileBaseName=`basename $f`					#get fileName
	echo "	$fileBaseName" >>$infoFile
	cp $f $outputDir						#copy to output folder
	cp $origDir/orig.$fileBaseName $outputDir
	
	for kmer in $seqK 
	do
	#	echo "$k $fileBaseName"
		for block in $seqBlock
		do
		
			echo "<Result id='$timeStamp.$kmer.$block.$fileBaseName' method='EulerCUDA' datasetName='$fileBaseName' params='l=$kmer, file=$fileBaseName ,block=$block' l='$kmer' block='$block'>" >>$statFile
			outputFile="$outputDir/output.eulercuda.b$block.k$kmer.$fileBaseName"
			outputFileBaseName=`basename $outputFile`
				
			T1=`date +%s.%N`
			$eulerCudaApp $f  $outputFile  $kmer $block  1>>$statFile  2>> $logFile		# Run CUDA
			T2=`date +%s.%N`
			
			totalTime=`echo "scale=3;( $T2 - $T1)/1"|bc -l`			#this division is wierd :-/
			echo "<item name='Total Time' value='$totalTime' />" >>$statFile
			
			perl n50.pl $outputFile 1>>$statFile

	#		for matchLength in 10 50 100
	#		do
	#			nucmerOutputFile=$outputDir/nucmer.$matchLength.$outputFileBaseName
	#			nucmer -maxmatch -c $matchLength -p $nucmerOutputFile   $origDir/orig.$fileBaseName   $outputFile &>> $logFile
	#			show-coords -r -c -l $nucmerOutputFile.delta 1> $nucmerOutputFile.coords
        #
	#			accTotal=`awk ' NR>5 { if(a[$19]<$16) {a[$19]=$16}} END {{j=0}for (i in a) { j+=a[i]  } {printf "%f\n", j} } ' $nucmerOutputFile.coords`
	#			totalContig=`awk '/^>.*/ {i++} END { print i}' $outputFile`
	#			weightedAccTotal=`awk ' NR>5 { if(a[$19]<$16*$13) {a[$19]=$16*$13}} END {{j=0}for (i in a) { j+=a[i]  } {printf "%f\n", j} } ' $nucmerOutputFile.coords`
	#			weightedTotalContig=`awk '/^[^>].*/ {i+=length($0)} END { print i}' $outputFile`
	#		#	accTotal=`awk`
	#		#	echo "$accTotal   $totalContig"
	#		#	if [ "${accTotal+x}" = x ] && [ -z "$accTotal" ] ;then
	#		#			accTotal=0
	#		#	fi  
	#			accuracy=`echo "scale=3; $accTotal/$totalContig" | bc -l`
	#			weightedAccuracy=`echo "scale=3; $weightedAccTotal/$weightedTotalContig" | bc -l`
	#			echo "<item name='Accuracy$matchLength' value='$accuracy'/>" >>$statFile 
	#			echo "<item name='WeightedAccuracy$matchLength' value='$weightedAccuracy'/>" >>$statFile 
	#			
	#		done
			echo "</Result>" >> $statFile
		done
	done 
	
done

echo "</Application>" >>$statFile


