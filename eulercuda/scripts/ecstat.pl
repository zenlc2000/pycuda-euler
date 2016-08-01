#!/bin/perl
use strict;
use warnings;

open(CORRECTED,$ARGV[0]) || die "could not open the fasta file\n";
open(ALNFILE,$ARGV[1]) || die "could not open alg file\n";
open(FQFILE,$ARGV[2]) || die "could not open fq file\n";


@aln =();
@reads=();
@fq = ();

# read corrected reads
 while( $line =<CORRECTED>){
	chomp $line;
	if($line =~/>/){

	}elsif( length $line >0){
			
	}
}

@contig_len = ();
$sum = 0;
$i=0;
$totalContigs;
$avgLen;
$n50;

$len_sum = 0;
$cntr = 0;
while ($line = <CONTIGFILE>){
   chomp $line;
   if ($line =~/>/){

#        print $line, "\n";
    if($cntr !=0){

       	$contig_len[$cntr -1] = $sum;
	$len_sum += $sum;
     }
     $sum = 0;
     $cntr ++;

   }elsif( length $line >0){

       $line_length = length $line;
       $sum += $line_length;
	
   }
}

$contig_len[$cntr-1] = $sum;
$len_sum += $sum;


#print $i," ",$sum/$i," ",$contig_len[$i/2], "\n";
$totalContigs=$cntr;
$avgLen=$len_sum/$cntr;

@contig_len=sort{$a <=> $b} @contig_len;

$sum = $len_sum;
$runnng_sum = 0;
$flag == 0;
for ($j=$cntr;$j>=0&&$flag==0;$j--){

   $running_sum += $contig_len[$j];
   if ($running_sum >= $sum/2){
    #   print "N50= ", $contig_len[$j], "\n";
	$n50=$contig_len[$j];
       $flag++;
   }

}

$filter=100;
$contig_len_f=();
$len_sum_f=0;
$cntr_f=0;

for($i=0;$i<$cntr;$i++){
	if($contig_len[$i]>=$filter)
		{
		$contig_len_f[$cntr_f]=$contig_len[$i];
		$len_sum_f+=$contig_len_f[$cntr_f];
		$cntr_f++;
		}
}

$totalContigs_f=$cntr_f;
$avgLen_f=$len_sum_f/$totalContigs_f;
$sum = $len_sum_f;
$running_sum = 0;
$flag = 0;
$n50_f=0;
for ($j=$cntr_f;$j>=0&&$flag==0;$j--){

   $running_sum += $contig_len_f[$j];
   if ($running_sum >= $sum/2){
    #   print "N50= ", $contig_len[$j], "\n";
	$n50_f=$contig_len_f[$j];
       $flag++;
   }

}

#for($k=0; $k<$cntr_f;$k++) {
#print "$contig_len_f[$k] \n";}
$avgStr= sprintf("%.3f",$avgLen);
$avgStr_f= sprintf("%.3f",$avgLen_f);
print "<item name='N50' value='$n50'/>\n";
print "<item name='N50Filter' value='$n50_f'/>\n";
print "<item name='Avg. Len.' value='$avgStr'/>\n";
print "<item name='Avg. Len.Filter' value='$avgStr_f'/>\n";
print "<item name='Contigs' value='$totalContigs'/>\n";
print "<item name='ContigsFilter' value='$totalContigs_f'/>\n";
print "<item name='TotalBases' value='$len_sum'/>\n";
print "<item name='TotalBasesFilter' value='$len_sum_f'/>\n";
print "<item name='LongestContig' value='$contig_len[$cntr-1]'/>\n";
print "<item name='LongestContigFilter' value='$contig_len_f[$cntr_f-1]'/>\n";


