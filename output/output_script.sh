#for i in {100..105}
#do
#    mkdir H$i
#done
#for i in {100..105}
#do
#    mv H$i* ./H$i
#done
#
#for i in {95..99}
#do
#    mkdir H0$i
#done
#for i in {95..99}
#do
#    mv H0$i* ./H0$i
#done
for i in {117..117}
do
    cp 'raw/H'$i'/H'$i'_eigenbills_pf.png' ./eigenbills
    cp 'raw/H'$i'/H'$i'_eigenbills_polarized.png' ./eigenbills
    cp 'raw/H'$i'/H'$i'_eigenmembers.png' ./eigenmembers
    cp 'raw/H'$i'/H'$i'_evdistribution.png' ./evdistribution
    cp 'raw/H'$i'/H'$i'_eigenbills_squared_pf.png' ./eigenbills_squared
    cp 'raw/H'$i'/H'$i'_eigenbills_squared_polarized.png' ./eigenbills_squared
    cp 'raw/H'$i'/H'$i'_eigenbills_squared_comparison.png' ./eigenbills_squared
    cp 'raw/H'$i'/H'$i'_eigenmembers_squared.png' ./eigenmembers_squared
    mkdir 'raw/H'$i'/interactive_output'
done

#for i in {95..99}
#do
#    cp 'raw/H0'$i'/H0'$i'_eigenbills_pf.png' ./eigenbills
#    cp 'raw/H0'$i'/H0'$i'_eigenbills_polarized.png' ./eigenbills
#    cp 'raw/H0'$i'/H0'$i'_eigenmembers.png' ./eigenmembers
#    cp 'raw/H0'$i'/H0'$i'_evdistribution.png' ./evdistribution
#    cp 'raw/H0'$i'/H0'$i'_eigenbills_squared_pf.png' ./eigenbills_squared
#    cp 'raw/H0'$i'/H0'$i'_eigenmembers_squared.png' ./eigenmembers_squared
#    cp 'raw/H0'$i'/H0'$i'_eigenbills_squared_comparison.png' ./eigenbills_squared
#    mkdir 'raw/H0'$i'/interactive_output'
#done
