import numpy as np

n_p=105;
n_o=95;
n=n_p+n_o;
m=200;

for i in range(6):
    for j in range(6):
        for k in range(6):
            for p in range(6):
                for q in range(1):
                    
                    output_file = 'output/pseudo_congress/PH'+str(i*10000+j*1000+k*100+p*10+q)

                    pr_nonpart=.05+i*0.07;
                    pr_pol=.05+j*1.5;
                    pr_opp=.05+k*1.5;
                    pr_troll=1-pr_pol-pr_opp;
                    pr_rand = 0.7+p*0.04
                    a=np.zeros([n,m]);

                    def bill(pr_p,pr_po,pr_op,pr_tr,b_p,b_o,p_sl,pr_rand):
                        b=np.zeros([b_p+b_o]);
                        if np.random.rand(1)<pr_nonpart:
                            for k in range(b_p+b_o):
                                b[k]=2*np.random.randint(0,2) -1;
                        else:
                            if np.random.rand(1)<pr_po:
                                for k in np.arange(b_p):
                                    b[k]=2*int(np.random.binomial(1,p_sl))-1;
                                for k in np.arange(b_p+1,b_p+b_o):
                                    b[k]=2*int(np.random.binomial(1,1-p_sl))-1;
                            elif np.random.rand(1)<pr_op:
                                for k in np.arange(b_p):
                                    b[k]=2*int(np.random.binomial(1,.7))-1;
                                for k in np.arange(b_p+1,b_p+b_o):
                                    b[k]=2*int(np.random.binomial(1,.7))-1;
                            else:
                                for k in np.arange(b_p):
                                    b[k]=2*int(np.random.binomial(1,1-p_sl))-1;
                                for k in np.arange(b_p+1,b_p+b_o):
                                    b[k]=2*int(np.random.binomial(1,p_sl))-1;
                        return(b);

                    for l in range(m):
                        a[:,l]=bill(pr_nonpart,pr_pol,pr_opp,pr_troll,n_p,n_o,.95+q*0.05,pr_rand);

                    [u,s,v]=np.linalg.svd(a);

                    import matplotlib.pyplot as plt

                    plt.scatter(v[0,:], v[1,:])
                    plt.savefig(output_file)
                    plt.clf()
