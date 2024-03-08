for i in {100..117}
do   
    #python rollcall_analysis.py HS$i
    #python eigenbill_analysis.py H$i
    python3.6 spectral_analysis.py H$i
    #python display_voting_pattern.py H$i
    #python tsne_bills.py H$i
    #python2.7 eigenbill_density.py H$i
done
for i in {95..99}
do   
    #python rollcall_analysis.py HS0$i
    #python eigenbill_analysis.py H0$i
    python3.6 spectral_analysis.py H0$i
    #python display_voting_pattern.py H0$i
    #python tsne_bills.py H0$i
    #python2.7 eigenbill_density.py H0$i
done
#for i in {95..99}
#do   
    #python rollcall_analysis.py H0$i
    #python eigenbill_analysis.py H0$i
#done
