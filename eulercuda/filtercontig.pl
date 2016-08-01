#!/bin/perl

open(CONTIGFILE,$ARGV[0]) || die "could not open the contig file\n";

@contig =();
$header="";
$i=0;$j=0;
$len =0;
while( $line = <CONTIGFILE>){
	chomp $line;
	if($line =~/>/){
		if($i>0 && $len >= $ARGV[1] ){
			print "$header\n"; 		
			$j=0;
			while($j<$i){
				print "$contig[$j++]\n";
			}	
		}
		$header=$line;
		$i=0; $len=0;
		@contig=();
	
	}elsif((length $line)>0) {
		$contig[$i++]=$line;
		$len+= length $line;	
	}
}
if($i>0 && $len >= $ARGV[1] ){
	print "$header\n"; 		
	$j=0;
	while($j<$i){
		print "$contig[$j++]\n";
	}	
}

