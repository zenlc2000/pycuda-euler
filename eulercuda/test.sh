echo "<Application name='EulerCuda'>"

for m in c1.fas  c2.fas c3.fas
do 
	#for j in  13 14  15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
	for j in   18 19  26 
	do 
		#for i  in  1 2 4 8 16 32 64 128  256 512
		for i  in  512
		do
			#echo "block=$i k=$j file=$m"
			for t in  1 2 3 4 5 6 7 8 9 10 11 12 13 14 15  
			do
				echo "<Result id='$t.$i.$j.$m' method='EulerCuda' datasetName='$m' params='t=$t,block=$i,k=$j,file=$m' l='$j' block='$i' iteration='$t'>"
				#echo "./bin/linux/release/eulercuda ./$m output.b$i.k$j.$m $j $i"
				./bin/linux/release/eulercuda ./$m output.$t.b$i.k$j.$m $j $i >/dev/null
				#	echo "N50 Score"
				perl n50.pl output.$t.b$i.k$j.$m 
				#./nucscript.sh orig.$m output.b$i.k$j.$m 100
				echo "</Result>"
			done
		done
	done
done

echo "</Application>"




