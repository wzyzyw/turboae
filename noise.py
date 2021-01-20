import numpy as np
from scipy.special import lambertw
import matplotlib.pyplot as plt
class Cnoise():
    def __init__(self):
        pass
    def addnoise(self,x,snr):
        pass
        return x
    def gaussion(self,x,snr):
        row,col=x.shape
        snr = 10 ** (snr / 10.0)
        xpower = np.sum(x ** 2) / len(x)
        npower = xpower / snr
        noise_real = np.random.randn(row,col) * np.sqrt(npower)
        noise_imag = np.random.randn(row,col) * np.sqrt(npower)
        noise=noise_real+1j*noise_imag
        return noise
class gaussionnoise(Cnoise):
    def __init__(self):
        super().__init__()
        print("gaussion noise generate")
    def addnoise(self,x,snr):
        noise=self.gaussion(x,snr)
        return x+noise
class itsnoise(Cnoise):
    def __init__(self,config,hnum):
        super().__init__()
        print("its noise generate")
        self._samples=hnum
        self._timeseq=np.arange(0,hnum*config["sample_interval"],config["sample_interval"]).reshape((-1,1))
        self._noiseconfig={
            "carrierfreq":13.86e6,
            "lpbw":config["bandwidth"],
            "theta_zg":2,
            "Ni":40,
            "theta_mg":1.2,
            
            "Nj":200,
            "mg_max":2.0e-5,
            "mg_section":(450e-6,550e-6),
            "mg_interval":4e-6,
            "theta_dz":4,
            "z1_tf":57.43,
            "z2_tf":32.23,
            "z3_tf":12.68,
            "z1_aj":18.62,
            "z2_aj":16.62,
            "z3_aj":1.49,
            "sample_interval":config["sample_interval"],
        }
        self._count=0
    def hall1(self,li,theta,gamma):
        res=gamma*np.sqrt(np.power(1-li,1/(1-theta))-1)
        return res
    def hall2(self,li,theta,gamma,mgmax):
        res=gamma*np.sqrt(np.power((li*(np.power(mgmax**2/gamma**2+1,(1-theta)/2)-1)+1),2/(1-theta))-1)
        return res
    def hall3(self,li,z1,z2,z3):
        res=-1*z1/(z2*z3)-1/z3*np.log(1-li)+1/z2*lambertw(z1/z3*np.exp((z1+z2*np.log(1-li))/z3))
        return res
    def zhaidaiganrao(self,npower):
        gamma_zg=np.sqrt(0.5*(-npower+np.sqrt(npower**2+4*(self._noiseconfig["theta_zg"]-1)/(self._noiseconfig["theta_zg"]+1))))
        zg=np.zeros((self._samples,1))+1j*np.zeros((self._samples,1))
        for index in range(self._samples):
            for i in range(0,self._noiseconfig["Ni"]):
                distribution=np.random.uniform(0,1)
                zg_maganitude=self.hall1(distribution,self._noiseconfig["theta_zg"],gamma_zg)
                omega=np.random.uniform(-1*self._noiseconfig["lpbw"],self._noiseconfig["lpbw"])
                fai=np.random.uniform(0,np.pi*2)
                zg[index,0]+=zg_maganitude*np.exp(-1j*(omega*self._timeseq[index]+fai))
            if np.abs(np.real(zg[index,0]))>50 or np.abs(np.imag(zg[index,0]))>50:
                zg[index,0]=0
        return zg
    
    
    def maichongganrao(self,npower):
        mg=np.zeros((self._samples,1))+1j*np.zeros((self._samples,1))
        gamma_mg=np.sqrt(0.5*(-npower+np.sqrt(npower**2+4*(self._noiseconfig["theta_mg"]-1)/(self._noiseconfig["theta_mg"]+1))))
        mg_tj=[0]
        while len(mg_tj)<=self._noiseconfig["Nj"]:
            t=np.random.uniform(self._noiseconfig["mg_section"][0],self._noiseconfig["mg_section"][1])
            t+=mg_tj[-1]
            mg_tj.append(np.random.uniform(t,t+self._noiseconfig["mg_interval"]))
        mg_tj.pop(0)
        for index in range(self._samples):
            for i in range(self._noiseconfig["Nj"]):
                distribution=np.random.uniform(0,1)
                mg_maganitude=self.hall2(distribution,self._noiseconfig["theta_mg"],gamma_mg,self._noiseconfig["mg_max"])
                mg[index,0]+=mg_maganitude*np.sin(2*np.pi*self._noiseconfig["lpbw"]*(self._timeseq[index]-mg_tj[i]))/(self._timeseq[index]-mg_tj[i])*np.exp(1j*self._noiseconfig["carrierfreq"]*mg_tj[i])
        return mg
    
    def daqizaosheng(self,npower):
        gamma_dz=np.sqrt(0.5*(-npower+np.sqrt(npower**2+4*(self._noiseconfig["theta_dz"]-1)/(self._noiseconfig["theta_dz"]+1))))
        distribution=np.random.uniform(0,1)
        T_tf=self.hall3(distribution,self._noiseconfig["z1_tf"],self._noiseconfig["z2_tf"],self._noiseconfig["z3_tf"])
        distribution=np.random.uniform(0,1)
        T_aj=self.hall3(distribution,self._noiseconfig["z1_aj"],self._noiseconfig["z2_aj"],self._noiseconfig["z3_aj"])
        num_tf=int(T_tf/self._noiseconfig["sample_interval"])
        num_aj=int(T_aj/self._noiseconfig["sample_interval"])
        pshreshold=gamma_dz**2*(np.power(1+T_aj/T_tf,2/(self._noiseconfig["theta_dz"]-1))-1)
        print("shreshold:",pshreshold)
        dz_maganitude=[]
        def genedz(state):
            distribution=np.random.uniform(0,1)
            res=self.hall1(distribution,self._noiseconfig["theta_dz"],gamma_dz)
            print("res:",res)
            while (state=="tf" and res<=pshreshold) or (state=="aj" and res>=pshreshold):
                distribution=np.random.uniform(0,1)
                res=self.hall1(distribution,self._noiseconfig["theta_dz"],gamma_dz)
            return res

        while len(dz_maganitude)<self._samples:
            count=0
            while len(dz_maganitude)<self._samples and count<num_tf:
                count+=1
                dz_maganitude.append(genedz("tf"))
            count=0
            while len(dz_maganitude)<self._samples and count<num_aj:
                count+=1
                dz_maganitude.append(genedz("aj"))
        
        dz=np.zeros((self._samples,1))+1j*np.zeros((self._samples,1))
        for index in range(self._samples):
            fai=np.random.uniform(0,1)
            dz[index,0]=dz_maganitude[index]*np.exp(1j*fai)
        return dz

    def addnoise(self,npower):
        zg=self.zhaidaiganrao(npower)
        mg=self.maichongganrao(npower)
        dz=self.daqizaosheng(npower)
        # plt.figure(1)
        # plt.plot(self._timeseq,np.imag(zg))
        # plt.figure(2)
        # plt.plot(self._timeseq,np.imag(mg))
        # plt.figure(3)
        # plt.plot(self._timeseq,np.imag(dz))
        # plt.show()
        return np.real(zg+mg+dz)
    
if __name__ == "__main__":
    sps=1
    bandwidth=250e3
    roll_off=0.25
    baud_rate=bandwidth/(1+roll_off)
    sample_rate=baud_rate/sps
    sample_interval=1/sample_rate
    params={
        "sps":sps,
        "bandwidth":bandwidth,
        "roll_off":roll_off,
        "baud_rate":baud_rate,
        "sample_rate":sample_rate,
        "sample_interval":sample_interval
    }
    hnum=10200
    noise=itsnoise(params,hnum)
    x=np.ones((100,1))
    snr=10
    noise.addnoise(x,snr)


            




    
