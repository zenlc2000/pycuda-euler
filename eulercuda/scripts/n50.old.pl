#!/bin/perl

open(CONTIGFILE,$ARGV[0]) || die "could not open the contig file\n";

@contig_len = ();
$sum = 0;
$i=0;
$totalContigs;
$avgLen;
$n50;

while ($line = <CONTIGFILE>){
   chomp $line;
   if ($line =~/>/){

#        print $line, "\n";

   }elsif( length $line >0){

       $contig_len[$i++] = length $line;
       $sum += $contig_len[$i-1];
   }
}

#print $i," ",$sum/$i," ",$contig_len[$i/2], "\n";
$totalContigs=$i;
$avgLen=$sum/$i;

sort @contig_len;

$runnng_sum = 0;
$flag == 0;
for ($j=$i;$j>=0&&$flag==0;$j--){

   $running_sum += $contig_len[$j];
   if ($running_sum >= $sum/2){
    #   print "N50= ", $contig_len[$j], "\n";
	$n50=$contig_len[$j];
       $flag++;
   }

}

$avgStr= sprintf("%.3f",$avgLen);
print "<item name='N50' value='$n50'/>\n";
print "<item name='Avg. Len.' value='$avgStr'/>\n";
print "<item name='Contigs' value='$totalContigs'/>\n";

