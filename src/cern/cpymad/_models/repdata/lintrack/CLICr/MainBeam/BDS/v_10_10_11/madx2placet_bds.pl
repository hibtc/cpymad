#!/usr/local/bin/perl

if (@ARGV < 1) { die "usage:\tmadx2placet.pl FILE 1 (:take apertures from madX (0 apertures 1 m)) 1 (: for wakefield studies) 1 unsplitted elements \n"; }

open(FILE, $ARGV[0]) or die "Could not open file $ARGV[0]\n";

@names = ();
@line = ();
@elemnames = ();
@elemleng = ();
@key = ();
@a_x = (); 
@a_y = (); 
@til = ();
@k0 = ();
@k1 = ();
@k2 = ();
@k3 = ();
@k4 = ();
@k5 = ();
@k6 = ();
@ang = ();
@e1 = ();
@e2 = ();
@type = ();
@skip_lines = ();

sub search_index  # $i = search_index("ENERGY"); $energy = $line[$i];
{
    my $name = $_[0];
    my $i;
    
    for ($i = 0; $i < @names; $i++)
    {
	if ($names[$i] eq $name)
	{
	    last;
	}
    }
    return $i;
}

sub search_value  # $energy = search_value("ENERGY");
{
    return $line[search_index($_[0])];
}

sub check_name  
{
    my $name = $_[0];
    my $jj = $_[1];
    my $chk_nm = 1;

    for ($j =$jj-2; $j<$jj; $j++)
    {
	if(@elemnames[$j] eq $name)
	{   
	   $chk_nm = 0; 
	   last;
	}
    }
    return $chk_nm;
}

sub check_leng  
{
    my $name = $_[0];
    my $j = $_[1];
    my $chk_lg = 1;
    $elemleng = @elemleng;
    if($j > 1) {
	if((@elemleng[$j-1] eq 0 && @elemnames[$j-2] eq $name) || (@elemleng[$j-1] eq 0 && @elemleng[$j-2] eq 0 && @elemnames[$j-3] eq $name))
	{   
	    $chk_lg = 0;
	}
    }
    return $chk_lg;
}

sub check_same_line
{
    my $name = $_[0];
    my $jj = $_[1];
    my $kk = $_[2];
    my $nl = 1;
    for ($k=$jj+1;$k<$jj+5; $k++)
    {
	if(@elemnames[$k] eq $name && @elemleng[$k] eq @elemleng[$jj] && ($kk eq $k0[$jj] || $kk eq $k1[$jj] || $kk eq $k2[$jj] || $kk eq $k3[$jj] || $kk eq $k4[$jj]) )
	{
	  $nl++;
	  @skip_lines[$k] = 1;
	}
	elsif (@elemleng[$k] != 0) 
	{
	    last;
	}
    }
    return $nl;
}

sub check_merror
{
    my $name = $_[0];
    my $jj = $_[1];
    my $ki = 0;
    for ($k=$jj+1;$k<$jj+3; $k++)
    {
	if( $key[$k]=~ /MULTIPOLE/ && @elemleng[$k] == 0 && ($k0[$k] ne 0 || $k1[$k] ne 0 || $k2[$k] ne 0 || $k3[$k] ne 0 || $k4[$k] ne 0 || $k5[$k] ne 0 || $k6[$k] ne 0) )
	{
	    @skip_lines[$k] = 1;
	    $ki = $k;
	} 
    }
    return $ki;
}

my $aper = $ARGV[1];
my $coll = $ARGV[2];
my $split = $ARGV[3];

# ==========================>>>>>>>>>>>read file

my $count = 0;

while ($lines = <FILE>)
{

    if ($lines =~ /^\*/) {
	
	@line = split(" ", $lines);
	
	for ($i=1;$i<@line;$i++)
	{
	    push(@names, $line[$i]);
	}
	
    } elsif ($lines !~ /^[@\*\$]/) {
	
	@line = split(" ", $lines);
	
	my $keyword = search_value("KEYWORD");
	my $length = search_value("L");
	my $name = search_value("NAME");

	my $apx = 1.0; # beware, this is in meters
	my $apy = 1.0; # beware, this is in meters	    
	if($aper == 1) {
	     $apx = search_value("APER_1"); # beware, this is in meters
	     $apy = search_value("APER_2"); # beware, this is in meters
        }

	my $tilt = search_value("TILT");
	my $k0l = search_value("K0L");
	my $k1l = search_value("K1L");
	my $k2l = search_value("K2L");
	my $k3l = search_value("K3L");
	my $k4l = search_value("K4L");
	my $k5l = search_value("K5L");
	my $k6l = search_value("K6L");
	my $angle = search_value("ANGLE");

	push(@key,$keyword);
	push(@elemleng,$length);
	push(@elemnames,"$name");
	push(@a_x,$apx);
	push(@a_y,$apy);
	push(@til,$tilt);
	push(@k0,$k0l);
	push(@k1,$k1l);
	push(@k2,$k2l);
	push(@k3,$k3l);
	push(@k4,$k4l);
	push(@k5,$k5l);
	push(@k6,$k6l);
	push(@ang,$angle);
	push(@skip_lines,0);
	$count++;
    }
}

close(FILE);

# ==================================>>>>>>>> write file


for ($i=0;$i<$count; $i++)
{

    if (@skip_lines[$i] == 1) { next; } 
    
    my $name = $elemnames[$i];
    my $keyword= $key[$i];
    my $apx = $a_x[$i];
    my $apy = $a_y[$i];
    my $tilt = $til[$i];
    my $k0l = $k0[$i];
    my $k1l = $k1[$i];
    my $k2l = $k2[$i];
    my $k3l = $k3[$i];
    my $k4l = $k4[$i];
    my $k5l = $k5[$i];
    my $k6l = $k6[$i];
    my $angle = $ang[$i];
    my $length = $elemleng[$i];

    my $chk = check_name($name,$jj);
    my $chl = check_leng($name,$i);

    if ($keyword =~ /DRIFT/)
    {
# This part is commented because in some lattice after a drift a series of multipoles may come 09/03/2010
#	if( $key[$i+1] =~ /MULTIPOLE/ && $elemleng[$i+1] == 0 && ( $k0[$i+1] != 0 || $k1[$i+1] != 0 || $k2[$i+1] != 0 || $k3[$i+1] != 0 || $k4[$i+1] != 0 || $k5[$i+1] != 0 || $k6[$i+1] != 0) ){
#	if( $key[$i+1] =~ /MULTIPOLE/ && $elemleng[$i+1] == 0 ){
#	  
#	    $elemleng[$i+1] = $length;
#	}
#	else {
#	    if($chl) {
            if ($length != 0) {
	       print "Girder\n";
	    }
#	    }
	    print "Drift -name $name -length $length";
	    if($apx !=0 && $apy !=0){
		print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apy\n";
	    }
	    elsif($apx !=0 && $apy ==0){
		print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apx\n";	  
	    }
	    else {
		print " -aperture_shape elliptic -aperture_x 0.008 -aperture_y 0.008\n";
	    }
#	}
    }
    elsif ($keyword =~ /QUADRUPOLE/)
    {
	if($split) {
	    if($chl) {
		if ($length != 0) {
		    print "Girder\n"; 
		}
	    }
	    my $chk_nl = check_same_line($name,$i,$k1l);
	    my $chk_me = check_merror($name,$i);
	    my $streng = $chk_nl*$k1l;
	    my $lengt = $chk_nl*$length;
	    print "Quadrupole -name $name -synrad \$quad_synrad -length $lengt -strength \[expr $streng*\$e0\]";
	}
	else {
	    if ($length != 0) {
		print "Girder\n";
	    }
	    print "Quadrupole -name $name -synrad \$quad_synrad -length $length -strength \[expr $k1l*\$e0\]";
	}
#	if( $chk_nl >1 && $chk_me )
#	{
#	    if ($k1[$chk_me] ne 0){
#		my $str = $k1[$chk_me];
#		print " -type 2 -Kn \[expr $str*\$e0\]";
#	    } elsif ($k2[$chk_me] ne 0 && $k2[$chk_me] != 0){
#		my $str = $k2[$chk_me];
#		print " -type 3 -Kn \[expr $str*\$e0\]";
#	    } elsif ($k3[$chk_me] ne 0 && $k3[$chk_me] != 0){
#		my $str = $k3[$chk_me];
#		print " -type 4 -Kn \[expr -1.0*$str*\$e0\]";		
#	    } elsif ($k4[$chk_me] ne 0 && $k4[$chk_me] != 0){
#		my $str = $k4[$chk_me];
#		print " -type 5 -Kn \[expr $str*\$e0\]";	
#	    } elsif ($k5[$chk_me] ne 0 && $k5[$chk_me] != 0 ){
#		my $str = $k5[$chk_me];
#		print " -type 6 -Kn \[expr -1.0*$str*\$e0\]";	
#	    } elsif ($k6[$chk_me] ne 0 && $k6[$chk_me] != 0 ){
#		my $str = $k6[$chk_me];
#		print " -type 7 -Kn \[expr $str*\$e0\]";	
#	    } 
#	}

	if ($tilt != 0)
	{
	    print " -tilt $tilt";
	}
	if($apx !=0 && $apy !=0){
	    print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apy\n";
	}
	elsif($apx !=0 && $apy ==0){
	    print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apx\n";	  
	}
	else {
	    print " -aperture_shape elliptic -aperture_x 0.008 -aperture_y 0.008\n";
	}
    }
    elsif ($keyword =~ /SEXTUPOLE/)
    {
	my $tilt = -1.0 * $tilt;
	if($split){
	    if($chk && $chl) {
		if ($length != 0) {
		    print "Girder\n";
		}
	    }
	    my $chk_nl = check_same_line($name,$i,$k2l);
	    my $streng = $chk_nl*$k2l;
	    my $lengt = $chk_nl*$length;
            my $stp = $chk_nl*5;
	    print "Multipole -name $name -synrad \$mult_synrad -type 3 -length $lengt -strength \[expr $streng*\$e0\] -steps $stp";
	}
	else {
	    if ($length != 0) {
		print "Girder\n";
	    }
	    print "Multipole -name $name -synrad \$mult_synrad -type 3 -length $length -strength \[expr $k2l*\$e0\]";
	}
	if ($tilt != 0)
	{
	    print " -tilt $tilt";
	}      
	if($apx !=0 && $apy !=0){
	    print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apy\n";
	}
	elsif($apx !=0 && $apy ==0){
	    print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apx\n";	  	  
	}
	else {
	    print " -aperture_shape elliptic -aperture_x 0.008 -aperture_y 0.008\n";	  		
	}	    
    }
    elsif ($keyword =~ /OCTUPOLE/)
    {
	my $tilt = -1.0 * $tilt;
	if($split) {
	    if($chk && $chl) {
		if ($length != 0) {
		    print "Girder\n";
		}
	    }
	    my $chk_nl = check_same_line($name,$i,$k3l);
	    my $streng = $chk_nl*$k3l;
	    my $lengt = $chk_nl*$length;
            my $stp = $chk_nl*5;
	    print "Multipole -name $name -synrad \$mult_synrad -type 4 -length $lengt -strength \[expr -1.0*$streng*\$e0\] -steps $stp";
	}
	else {
	    if ($length != 0) {
		print "Girder\n";	
	    }	    
	    print "Multipole -name $name -synrad \$mult_synrad -type 4 -length $length -strength \[expr -1.0*$k3l*\$e0\]";
	}
	if ($tilt != 0)
	{
	    print " -tilt $tilt";
	}
	if($apx !=0 && $apy !=0){
	    print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apy\n";
	}
	elsif($apx !=0 && $apy ==0){
		print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apx\n"; 	  
	    }
	    else {
		print " -aperture_shape elliptic -aperture_x 0.008 -aperture_y 0.008\n";	  		
	    }
	}
	elsif ($keyword =~ /MULTIPOLE/)
	{
	    my $tilt = -1.0 * $tilt;
	    if ($k0l != 0) {
		if($split) {
		    if($chk && $chl) {
			if ($length != 0) {
			    print "Girder\n";
			}
		    }
		    my $chk_nl = check_same_line($name,$i,$k0l);
		    my $streng = $chk_nl*$k0l;
		    my $lengt = $chk_nl*$length;
		    my $stp = $chk_nl*5;
		    print "Dipole -name $name -synrad 0 -length $lengt -strength \[expr -1.0*$streng*\$e0\] -steps $stp";
		}
		else {
		    if ($length != 0) {
			print "Girder\n";
		    }
		    print "Dipole -name $name -synrad 0 -length $length -strength \[expr -1.0*$k0l*\$e0\]";
		}
		if ($tilt != 0)
		{
		    print " -tilt $tilt";
		}
		if($apx !=0 && $apy !=0){
		    print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apy\n";
		}
		elsif($apx !=0 && $apy ==0){
		    print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apx\n"; 	    
		}
		else {
		    print " -aperture_shape elliptic -aperture_x 0.008 -aperture_y 0.008\n";	  		
		}		
	    } elsif ($k1l != 0) {
		if($split) {
		    if($chk && $chl) {
			if ($length != 0) {
			    print "Girder\n";
			}
		    }
		    my $chk_nl = check_same_line($name,$i,$k1l);
		    my $streng = $chk_nl*$k1l;
		    my $lengt = $chk_nl*$length;
		    my $stp = $chk_nl*5;
		    print "Quadrupole -name $name -synrad \$quad_synrad -length $lengt -strength \[expr -1.0*$streng*\$e0\] -steps $stp"; # to be check
		}
		else {
		    if ($length != 0) {
			print "Girder\n";
		    }
		    print "Quadrupole -name $name -synrad \$quad_synrad -length $length -strength \[expr -1.0*$k1l*\$e0\]";
		}
		if ($tilt != 0)
		{
		    print " -tilt $tilt";
		}
		if($apx !=0 && $apy !=0){
		    print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apy\n";
		}
		elsif($apx !=0 && $apy ==0){
		    print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apx\n"; 	    
		}
		else {
		    print " -aperture_shape elliptic -aperture_x 0.008 -aperture_y 0.008\n";	  		
		}
	    } elsif ($k2l != 0) {
		if($split) {
		    if($chk && $chl) {
			if ($length != 0) {
			    print "Girder\n";
			}
		    }
		    my $chk_nl = check_same_line($name,$i,$k2l);
		    my $streng = $chk_nl*$k2l;
		    my $lengt = $chk_nl*$length;
		    my $stp = $chk_nl*5;
		    print "Multipole -name $name -synrad \$mult_synrad -type 3 -length $lengt -strength \[expr $streng*\$e0\] -steps $stp";
		}
		else {
		    if ($length != 0) {
			print "Girder\n";
		    }
		    print "Multipole -name $name -synrad \$mult_synrad -type 3 -length $length -strength \[expr $k2l*\$e0\]";
		}
		if ($tilt != 0)
		{
		    print " -tilt $tilt";
		}
		if($apx !=0 && $apy !=0){	  
		    print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apy\n";
		}
		elsif($apx !=0 && $apy ==0){
		    print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apx\n";	      
		}
		else {
		    print " -aperture_shape elliptic -aperture_x 0.008 -aperture_y 0.008\n";	  		
		}
	    } elsif ($k3l != 0) {
		if($split) {
		    if($chk && $chl) {
			if ($length != 0) {
			    print "Girder\n";
			}
		    }
		    my $chk_nl = check_same_line($name,$i,$k3l);
		    my $streng = $chk_nl*$k3l;
		    my $lengt = $chk_nl*$length;
		    my $stp = $chk_nl*5;
		    print "Multipole -name $name -synrad \$mult_synrad -type 4 -length $lengt -strength \[expr -1.0*$streng*\$e0\] -steps $stp";
		}
		else {
		    if ($length != 0) {
			print "Girder\n";
		    }
		    print "Multipole -name $name -synrad \$mult_synrad -type 4 -length $length -strength \[expr -1.0*$k3l*\$e0\]";    
		}
		if ($tilt != 0)
		{
		    print " -tilt $tilt";
		}
		if($apx !=0 && $apy !=0){
		    print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apy\n";
		}
		elsif($apx !=0 && $apy ==0){
		    print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apx\n";	      
		}
		else {
		    print " -aperture_shape elliptic -aperture_x 0.008 -aperture_y 0.008\n";	  		
		}
	    } elsif ($k4l != 0) {
		if($split) {
		    if($chk && $chl) {
			if ($length != 0) {
			    print "Girder\n";
			}
		    }
		    my $chk_nl = check_same_line($name,$i,$k4l);
		    my $streng = $chk_nl*$k4l;
		    my $lengt = $chk_nl*$length;
		    my $stp = $chk_nl*5;
		    print "Multipole -name $name -synrad \$mult_synrad -type 5 -length $lengt -strength \[expr $streng*\$e0\] -steps $stp";
		}
		else {
		    if ($length != 0) {
			print "Girder\n";
		    }
		    print "Multipole -name $name -synrad \$mult_synrad -type 5 -length $length -strength \[expr $k4l*\$e0\]";
		}
		if ($tilt != 0)
		{
		    print " -tilt $tilt";
		}
		if($apx !=0 && $apy !=0){
		    print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apy\n";
		}
		elsif($apx !=0 && $apy ==0){
		    print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apx\n";	     
		}
		else {
		    print " -aperture_shape elliptic -aperture_x 0.008 -aperture_y 0.008\n";	  		
		}
	    } elsif ($k5l != 0) {
		if($split) {
		    if($chk && $chl) {
			if ($length != 0) {
			    print "Girder\n";
			}
		    }
		    my $chk_nl = check_same_line($name,$i,$k5l);
		    my $streng = $chk_nl*$k5l;
		    my $lengt = $chk_nl*$length;
		    my $stp = $chk_nl*5;
		    print "Multipole -name $name -synrad \$mult_synrad -type 6 -length $lengt -strength \[expr -1.0*$streng*\$e0\] -steps $stp";
		}
		else {
		    if ($length != 0) {
			print "Girder\n";
		    }
		    print "Multipole -name $name -synrad \$mult_synrad -type 6 -length $length -strength \[expr -1.0*$k5l*\$e0\]";
		}
		if ($tilt != 0)
		{
		    print " -tilt $tilt";
		}
		if($apx !=0 && $apy !=0){
		    print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apy\n";
		}
		elsif($apx !=0 && $apy ==0){
		    print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apx\n";	     
		}
		else {
		    print " -aperture_shape elliptic -aperture_x 0.008 -aperture_y 0.008\n";	  		
		}
	    } else {
		print "# WARNING: Multipole options not defined. Multipole type 0 with 0.";
		print "\n";
		if($chk && $chl) {
		    if ($length != 0) {
			print "Girder\n";
		    }
		}
		print "#Multipole -name $name -synrad \$mult_synrad -type 0 -length $length -strength \[expr 0.0*\$e0\]";
		if ($tilt != 0)
		{
		    print " -tilt $tilt";
		}
		if($apx !=0 && $apy !=0){
		    print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apy\n";
		}
		elsif($apx !=0 && $apy ==0){
		    print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apx\n";	     
		}
		else {
		    print " -aperture_shape elliptic -aperture_x 0.008 -aperture_y 0.008\n";	  		
		}
	    } 
	}
	elsif ($keyword =~ /RBEND/)
	{
	    my $half_angle = $angle * 0.5;
	    my $radius = $length / sin($half_angle) / 2;
	    my $arc_length = $angle * $radius;
	    print "# WARNING: putting a Sbend instead of a Rbend. Arc's length is : angle * L / sin(angle/2) / 2\n";
	    print "# WARNING: original length was $length\n";
#	    if($chk) {  # I comment it for the moment because the bend always have the same names
		if ($length != 0) {
		    print "Girder\n";
		}
#	    }
	    print "Sbend -name $name -synrad \$sbend_synrad -length $arc_length -angle $angle -e0 \$e0 -E1 $half_angle -E2 $half_angle";
	    if ($k1l != 0.0) {
		print " -K \[expr $k1l*\$e0\]";
	    }
	    if($apx !=0 && $apy !=0){
		print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apy\n";
	    }
	    elsif($apx !=0 && $apy ==0){
		print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apx\n";	     
	    }
	    else {
		print " -aperture_shape elliptic -aperture_x 0.008 -aperture_y 0.008\n";	  		
	    }
	}
	elsif ($keyword =~ /SBEND/)
	{
	    my $half_angle = $angle * 0.5;
#	    if($chk) {  # I comment it for the moment because the bend always have the same names
	    if ($length != 0) {
		print "Girder\n";
	    }
#	    }
	    print "Sbend -name $name -synrad \$sbend_synrad -length $length -angle $angle -e0 \$e0 -E1 $half_angle -E2 $half_angle";
	    if ($k1l != 0.0) {
		print " -K \[expr $k1l*\$e0\]";
	    }
	    if($apx !=0 && $apy !=0){
		print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apy\n";
	    }
	    elsif($apx !=0 && $apy ==0){
		print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apx\n";	     
	    }
	    else {
		print " -aperture_shape elliptic -aperture_x 0.008 -aperture_y 0.008\n";	  		
	    }
	    print "set e0 \[expr \$e0-14.1e-6*$angle*$angle/$length*\$e0*\$e0*\$e0*\$e0*\$sbend_synrad\]\n";
	}
	elsif ($keyword =~ /MATRIX/)
	{
	}
	elsif ($keyword =~ /LCAVITY/)
	{
	    if($chk && $chl) {
		if ($length != 0) {
		    print "Girder\n";
		}
	    }
	    print "Cavity -name $name -length $length\n";
	}
	elsif ($keyword =~ /TWCAVITY/)
	{
	    if($chk && $chl) {
		if ($length != 0) {
		    print "Girder\n";
		}
	    }
#	    print "Cavity -name $name -length $length\n";
	    print "Drift -name $name -length $length\n";
	}
	elsif ($keyword =~ /RCOLLIMATOR/)
	{
	    if($chk && $chl) {
		if ($length != 0) {
		    print "Girder\n";
		}
	    }
	    if($coll!=0){
		print "# WARNING: for the collimator wakefield studies you need to add more info ==> Adding a Collimator commented \n";
		print "# Collimator -name $name -length $length \n";
	    }
	    else {
		print "Drift -name $name -length $length";
		if($apx !=0 && $apy !=0){
		    print " -aperture_shape rectangular -aperture_x $apx -aperture_y $apy\n";
		}
		elsif($apx !=0 && $apy ==0){
		    print " -aperture_shape rectangular -aperture_x $apx -aperture_y $apx\n";	     
		}
		else {
		    print " -aperture_shape rectangular -aperture_x 0.008 -aperture_y 0.008\n";	  		
		}
	    }
	}
	elsif ($keyword =~ /ECOLLIMATOR/)
	{
	    if($chk && $chl) {
		if ($length != 0) {
		    print "Girder\n";
		}
	    }
	    if($coll!=0){
		print "# WARNING: for the collimator wakefield studies you need to add more info ==> Adding a Collimator commented \n";
		print "# Collimator -name $name -length $length \n";
	    }
	    else {
		print "Drift -name $name -length $length";
		if($apx !=0 && $apy !=0){
		    print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apy\n";
		}
		elsif($apx !=0 && $apy ==0){
		    print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apx\n";	     
		}
		else {
		    print " -aperture_shape elliptic -aperture_x 0.008 -aperture_y 0.008\n";	  		
		}
	    }
	}
	elsif ($keyword =~ /HKICKER/)
	{
#	    print "Girder\n";
	    print "# HCORRECTOR -name $name -length $length\n";
	    if($chk && $chl) {
		if ($length != 0) {
		    print "Girder\n";    
		    print "Drift -name $name -length $length";
		    if($apx !=0 && $apy !=0){
			print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apy\n";
		    }
		    elsif($apx !=0 && $apy ==0){
			print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apx\n";	     
		    }
		    else {
			print " -aperture_shape elliptic -aperture_x 0.008 -aperture_y 0.008\n";	  		
		    }
		}
	    }
	}
	elsif ($keyword =~ /VKICKER/)
	{
#	    print "Girder\n";
	    print "# VCORRECTOR -name $name -length $length\n";
	    if($chk && $chl) {
		if ($length != 0) {
		    print "Girder\n";
		    print "Drift -name $name -length $length";
		    if($apx !=0 && $apy !=0){
			print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apy\n";
		    }
		    elsif($apx !=0 && $apy ==0){
			print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apx\n";	     
		    }
		    else {
			print " -aperture_shape elliptic -aperture_x 0.008 -aperture_y 0.008\n";	  		
		    }
		}
	    }
	}
	elsif ($keyword =~ /MARKER/)
	{
	    if($chk && $chl) {
		if ($length != 0) {
		    print "#Girder\n";
		}
	    }
	    print "#Drift -name $name -length $length";
	    if($apx !=0 && $apy !=0){
		print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apy\n";
	    }
	    elsif($apx !=0 && $apy ==0){
		print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apx\n";	     
	    }
	    else {
		print " -aperture_shape elliptic -aperture_x 0.008 -aperture_y 0.008\n";	  		
	    }	
	}
	elsif ($keyword =~ /MONITOR/)
	{
	    if($chk && $chl) {
		if ($length != 0) {
		    print "Girder\n";
		}
	    }
	    print "Bpm -name $name -length $length";
	    if($apx !=0 && $apy !=0){
		print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apy\n";
	    }
	    elsif($apx !=0 && $apy ==0){
		print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apx\n";	     
	    }
	    else {
		print " -aperture_shape elliptic -aperture_x 0.008 -aperture_y 0.008\n";	  		
	    }
	}
	else 
	{
	    print "# UNKNOWN: @line\n";
	    if($chk && $chl) {
		if ($length != 0) {
		    print "Girder\n";
		}
	    }
	    print "Drift -name $name -length $length"; 
	    if($apx !=0 && $apy !=0){
		print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apy\n";
	    }
	    elsif($apx !=0 && $apy ==0){
		print " -aperture_shape elliptic -aperture_x $apx -aperture_y $apx\n";	     
	    }
	    else {
		print " -aperture_shape elliptic -aperture_x 0.008 -aperture_y 0.008\n";	  		
	    }
	}
}


