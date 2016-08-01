
echo "<Application name='EulerCuda'>"
for m in c1.fas  c2.fas c3.fas
do 
	for j in  13 14  15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
	do 
		#for i  in  1 2 4 8 16 32 64 128  256 512
		for i  in  512
		do
			#echo "block=$i k=$j file=$m"
			echo "<Result id='$i.$j.$m' method='EulerCuda' datasetName='$m' params='block=$i,k=$j,file=$m' l='$j'>"
			#echo "./bin/linux/release/eulercuda ./$m output.b$i.k$j.$m $j $i"
			SECONDS=0
			./bin/linux/release/eulercuda ./$m output.b$i.k$j.$m $j $i >./output.cuda.log
			#	echo "N50 Score"
			perl n50.pl output.b$i.k$j.$m 
			#./nucscript.sh orig.$m output.b$i.k$j.$m 100
			echo "<item name='Total Time' value='$SECONDS'/>"
			echo "</Result>"
		done
	done
done
echo "</Application>"

