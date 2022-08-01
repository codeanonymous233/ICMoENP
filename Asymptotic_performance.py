import matplotlib.pyplot as plt 



'''
Collected Results for CIFAR10 are as follows:
NP-->
ordered case: number of context points-->avg mse(std)
10-->0.1087(0.0385)
100-->0.0860(0.0284)
200-->0.0676(0.0215)
500-->0.0427(0.0117)
800-->0.0297(0.0082)
1000-->0.0230(0.0056)
random case: number of context points-->avg mse(std)
10-->0.0464(0.0140)
100-->0.0255(0.0064)
200-->0.0240(0.0059)
500-->0.0231(0.0056)
800-->0.0229(0.0055)
1000-->0.0228(0.0055)


MoE-NP-->
ordered case: number of context points-->avg mse(std)
10-->0.0519(0.0097)
100-->0.0460(0.0085)
200-->0.0431(0.0110)
500-->0.0320(0.0062)
800-->0.0173(0.0046)
1000-->0.0093(0.0017)
random cases: number of context points-->avg mse(std)
10-->0.0366(0.0076)
100-->0.0165(0.0030)
200-->0.0130(0.0032)
500-->0.0102(0.0018)
800-->0.0094(0.0024)
1000-->0.0092(0.0017)


CNP-->
ordered case: number of context points-->avg mse(std)
10-->0.0529(0.0092)
100-->0.0546(0.0092)
200-->0.0523(0.0130)
500-->0.0515(0.0109)
800-->0.0395(0.0162)
1000-->0.0154(0.0076)
random cases: number of context points-->avg mse(std)
10-->0.0381(0.0077)
100-->0.0194(0.0035)
200-->0.0166(0.0041)
500-->0.0145(0.0026)
800-->0.0140(0.0035)
1000-->0.0138(0.0031)


FCRL-->
ordered case: number of context points-->avg mse(std)
10-->0.0616(0.0115)
100-->0.0598(0.0112)
200-->0.0583(0.0157)
500-->0.0567(0.0123)
800-->0.0579(0.0232)
1000-->0.0402(0.0201)
random cases: number of context points-->avg mse(std)
10-->0.0618(0.0115)
100--> 0.055(0.0115)
200-->0.0618(0.0163)
500-->0.0318(0.0115)
800-->0.0618(0.0164)
1000-->0.0220(0.0123)



CAVIA-->
ordered case: number of context points-->avg mse(std)
10-->0.028230771409114824(0.015061210494230193)
100-->0.029752083493489772(0.015636334287782543)
500-->0.0349328690765542(0.01740207307570431)
1000-->0.0512602776512284(0.03540407379612199)
random cases: number of context points-->avg mse(std)
10-->0.02818048977639992(0.015031260423973786)
100-->0.028182706681010312(0.015034368625251326)
500-->0.028172830146783963(0.015036833423238409)
1000-->0.027217366476450115(0.015545864292846826)


MAML-->
ordered case: number of context points-->avg mse(std)
10-->0.05787685379944742(0.028158938549841944)
100-->0.05651883692368865(0.02711200984744531)
500-->0.051317142689973116(0.029452787638018077)
1000-->0.0501850948082516(0.04901430964734363)
random cases: number of context points-->avg mse(std)
10-->0.05984744825735688(0.028693955915044034)
100-->0.05769303792081773(0.029060893980404574)
500-->0.05929628235809505(0.0297505621209166)
1000-->0.05877915835045278(0.03311903503668367)


'''



def asymptotic_vis():
    x = [10, 100, 200, 500, 800, 1000]
    
    # random cases
    y1 = [0.0464, 0.0255, 0.0240, 0.0231, 0.0229, 0.0228] # NP
    y2 = [0.0381, 0.0194, 0.0166, 0.0145, 0.0140, 0.0138] # CNP
    y3 = [0.0618, 0.055, 0.0618, 0.0318, 0.0618, 0.0220] # FCRL
    y4 = [0.05984, 0.0576, 0.0623, 0.0592, 0.0592, 0.0587] # MAML
    y5 = [0.02818, 0.0281, 0.0281, 0.02817, 0.02814, 0.0272] # CAVIA
    y6 = [0.0377, 0.0175, 0.0142, 0.0117, 0.0110, 0.0107] # MoE-CondNP
    
    # ordered cases
    y7 = [0.1087, 0.0860, 0.0676, 0.0427, 0.0297, 0.0230] # NP
    y8 = [0.0529, 0.0546, 0.0523, 0.0515, 0.0395, 0.0154] # CNP
    y9 = [0.0616, 0.0598, 0.0583, 0.0567, 0.0579, 0.0402] # FCRL
    y10 = [0.0578, 0.056, 0.0534, 0.0513, 0.0513, 0.0501] # MAML
    y11 = [0.0282, 0.0297, 0.0316, 0.0349, 0.0505, 0.0512] # CAVIA
    y12 = [0.0534, 0.0482, 0.0458, 0.0338, 0.0212, 0.0110] # MoE-CondNP
    
    
    plt.subplot(1, 2, 2)
    
    plt.plot(x, y7, '--', marker='*', label='NP')
    
    plt.plot(x, y8, '--', marker='*', label='CNP')
    
    plt.plot(x, y9, '--', marker='*', label='FCRL')
    
    plt.plot(x, y10, '--', marker='*', label='MAML')
    
    plt.plot(x, y11, '--', marker='*', label='CAVIA')
    
    plt.plot(x, y12, '--', marker='*', label='MoE-NP')
    
    plt.title("Testing in CIFAR10 Completion (Ordered)",fontsize=14)
    plt.xticks(fontsize=10)
    plt.xlabel("Number of Context Pixels",fontsize=12)
    plt.yticks(fontsize=10)
    plt.ylabel("MSE",fontsize=12)
        
    plt.grid(color='c',linestyle='--',linewidth=1.0,alpha=0.3)
    plt.legend(loc='upper right')
    
    
    plt.subplot(1, 2, 1)
    
    plt.plot(x, y1, '--', marker='*', label='NP')
    
    plt.plot(x, y2, '--', marker='*', label='CNP')
    
    plt.plot(x, y3, '--', marker='*', label='FCRL')
    
    plt.plot(x, y4, '--', marker='*', label='MAML')
    
    plt.plot(x, y5, '--', marker='*', label='CAVIA')
    
    plt.plot(x, y6, '--', marker='*', label='MoE-NP')
    
    plt.title("Testing in CIFAR10 Completion (Random)",fontsize=14)
    plt.xticks(fontsize=10)
    plt.xlabel("Number of Context Pixels",fontsize=12)
    plt.yticks(fontsize=10)
    plt.ylabel("MSE",fontsize=12)
        
    plt.grid(color='c',linestyle='--',linewidth=1.0,alpha=0.3)
    plt.legend(loc='upper right')
    
    plt.show() 
    
asymptotic_vis()