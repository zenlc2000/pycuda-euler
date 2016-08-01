for f in $1/output.*nm.50bp*.contig
do 
echo $f
perl n50.pl $f
done


