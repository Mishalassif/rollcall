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

for i in {100..105}
do
    cd HS$i
    rm 'HS' + $i + '_eigenbills.png'
    cd ..
done
