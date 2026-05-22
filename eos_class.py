from scipy import interpolate
import scipy.integrate
import scipy.special
import numpy as np
from unitconvert import toMevfm,toMev4,toLength,toMass
from scipy.constants import c,G,e
import copy
dlnx_cs2=1e-10

def derivative(func, x, dx=1e-6):
    return (func(x + dx) - func(x - dx)) / (2 * dx)

def quality_control_none(eos_array):
    logic_faithful_range=np.logical_and(eos_array[0]<3,eos_array[2]<3000)
    logic_positive=(eos_array>0).min(axis=0)
    eos_array=eos_array[:,np.logical_and(logic_faithful_range,logic_positive)]
    logic_mono=np.concatenate((np.array([True]),(eos_array[:,1:]-eos_array[:,:-1]>0).min(axis=0)))
    eos_array=eos_array[:,logic_mono]
    eos_array_low=eos_array[:,eos_array[2]<200]
    success_causality=((eos_array_low[2,1:]-eos_array_low[2,:-1])<(eos_array_low[1,1:]-eos_array_low[1,:-1])).min()
    success_stiff_enough=eos_array_low[0,-1]<1.6
    success_baryondensity_range=eos_array[0,-1]>0.3
    success_pressure_range=eos_array[2,0]<3e-8 and eos_array[2,-1]>200
    success_eos_array=np.array([success_causality, success_stiff_enough, success_baryondensity_range, success_pressure_range])
    #success_eos=success_eos_array.min()
    #print(eos_array[2,0], eos_array[2,-1])
    #print(success_causality , success_stiff_enough , success_baryondensity_range , success_pressure_range)
    logic_mono=(eos_array[:,1:]<eos_array[:,:-1]).max(axis=0)
    index_max=np.where(logic_mono)
    p_max=eos_array[2,index_max][0]
    if(len(p_max)==0):
        p_max=eos_array[2,-1]
    else:
        p_max=p_max[0]
        eos_array=eos_array[:,:index_max[0][0]]
    return True,eos_array,p_max,success_eos_array

def quality_control_basic(eos_array):
    logic_faithful_range=np.logical_and(eos_array[0]<3,eos_array[2]<3000)
    logic_positive=(eos_array>0).min(axis=0)
    eos_array=eos_array[:,np.logical_and(logic_faithful_range,logic_positive)]
    logic_mono=np.concatenate((np.array([True]),(eos_array[:,1:]-eos_array[:,:-1]>0).min(axis=0)))
    eos_array=eos_array[:,logic_mono]
    eos_array_low=eos_array[:,eos_array[2]<200]
    success_causality=((eos_array_low[2,1:]-eos_array_low[2,:-1])<(eos_array_low[1,1:]-eos_array_low[1,:-1])).min()
    success_stiff_enough=eos_array_low[0,-1]<1.6
    success_baryondensity_range=eos_array[0,-1]>0.3
    success_pressure_range=eos_array[2,0]<3e-8 and eos_array[2,-1]>200
    success_eos_array=np.array([success_causality, success_stiff_enough, success_baryondensity_range, success_pressure_range])
    success_eos=success_eos_array.min()
    #print(eos_array[2,0], eos_array[2,-1])
    #print(success_causality , success_stiff_enough , success_baryondensity_range , success_pressure_range)
    logic_mono=(eos_array[:,1:]<eos_array[:,:-1]).max(axis=0)
    logic_causal=(eos_array[2,1:]-eos_array[2,:-1])>(eos_array[1,1:]-eos_array[1,:-1])
    index_max=np.where(np.logical_or(logic_mono,logic_causal))
    p_max=eos_array[2,index_max][0]
    if(len(p_max)==0):
        p_max=eos_array[2,-1]
    else:
        p_max=p_max[0]
        eos_array=eos_array[:,:index_max[0][0]]
    return success_eos,eos_array,p_max,success_eos_array

class EOS_interpolation(object):
    def __init__(self,baryon_density_s,eos_array,quality_control_f=quality_control_basic,s_k=[0,2]): #defalt s=0,k=3 equal quadratic 1d intepolation
        self.s,self.k=s_k
        if(len(eos_array)==2):
            n_array,energy_array=eos_array
            eosDensity_frombaryon = interpolate.UnivariateSpline(n_array,energy_array, k=self.k,s=self.s)
            pressure_array=derivative(eosDensity_frombaryon,n_array,dx=n_array*dlnx_cs2**0.5)*n_array-energy_array
            eos_array=np.array([n_array,energy_array,pressure_array])
        else:
            pass
        self.success_eos,self.eos_array,self.p_max,self.success_eos_array=quality_control_f(eos_array)
        n_array,energy_array,pressure_array=self.eos_array
        if(self.success_eos):
            self.eosPressure_frombaryon = interpolate.UnivariateSpline(n_array,pressure_array, k=self.k,s=self.s)
            self.eosDensity  = interpolate.UnivariateSpline(pressure_array,energy_array, k=self.k,s=self.s)
            self.eosBaryonDensity = interpolate.UnivariateSpline(pressure_array,n_array, k=self.k,s=self.s)
            self.chempo_surface=(pressure_array[0]+energy_array[0])/n_array[0]
            self.baryon_density_s=baryon_density_s
            self.pressure_s=self.eosPressure_frombaryon(self.baryon_density_s)
            self.density_s=self.eosDensity(self.pressure_s)
            self.unit_mass=c**4/(G**3*self.density_s*1e51*e)**0.5
            self.unit_radius=c**2/(G*self.density_s*1e51*e)**0.5
            self.unit_N=self.unit_radius**3*self.baryon_density_s*1e45
        else:
            tmp=self.success_eos_array
            for item_name in [*self.__dict__]:  #this two lines is added on 8/15/2019
                del self.__dict__[item_name]    #to try free some memories.
            self.success_eos=False
            self.success_eos_array=tmp
            self.eosPressure_frombaryon,self.eosDensity,self.eosBaryonDensity = ['undefined']*3
    def __getstate__(self):
        state = self.__dict__.copy()
        for dict_intepolation in ['eosPressure_frombaryon','eosDensity','eosBaryonDensity']:
            del state[dict_intepolation]
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)
        if(self.success_eos):
            n_array,energy_array,pressure_array=self.eos_array
            self.eosPressure_frombaryon = interpolate.UnivariateSpline(n_array,pressure_array, k=self.k,s=self.s)
            self.eosDensity  = interpolate.UnivariateSpline(pressure_array,energy_array, k=self.k,s=self.s)
            self.eosBaryonDensity = interpolate.UnivariateSpline(pressure_array,n_array, k=self.k,s=self.s)
        else:
            self.eosPressure_frombaryon,self.eosDensity,self.eosBaryonDensity = ['undefined']*3
    def eosChempo(self,pressure):
        return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
    def eosCs2(self,pressure):
        return 1.0/derivative(self.eosDensity,pressure,dx=self.eosDensity(pressure)*dlnx_cs2)
    def setMaxmass(self,result_maxmaxmass):
        self.pc_max,self.mass_max,self.cs2_max=result_maxmaxmass
        self.maxmass_success=self.cs2_max<1 and 2<self.mass_max<5 and self.pc_max<self.p_max
        self.eos_success_all=self.maxmass_success and self.success_eos
        return self.eos_success_all
    def setProperity(self,Properity_one,Properity_onepointfour):
        self.properity_one,self.properity_onepointfour=Properity_one,Properity_onepointfour

# =============================================================================
# class EOS_interpolation(object):
#     def __init__(self,baryon_density_s,eos_array,quality_control_f=quality_control_basic,s_k=[0,2],eos_array_adiabatic=[],eos_array_Ylep=[]): #defalt s=0,k=3 equal quadratic 1d intepolation
#         self.s,self.k=s_k
#         if(len(eos_array)==2):
#             n_array,energy_array=eos_array
#             eosDensity_frombaryon = interpolate.UnivariateSpline(n_array,energy_array, k=self.k,s=self.s)
#             pressure_array=derivative(eosDensity_frombaryon,n_array,dx=n_array*dlnx_cs2**0.5)*n_array-energy_array
#             eos_array=np.array([n_array,energy_array,pressure_array])
#         else:
#             pass
#         self.success_eos,self.eos_array,self.p_max,self.success_eos_array=quality_control_f(eos_array)
#         n_array,energy_array,pressure_array=self.eos_array
#         if(self.success_eos):
#             self.eosPressure_frombaryon = interpolate.UnivariateSpline(n_array,pressure_array, k=self.k,s=self.s)
#             self.eosDensity  = interpolate.UnivariateSpline(pressure_array,energy_array, k=self.k,s=self.s)
#             self.eosBaryonDensity = interpolate.UnivariateSpline(pressure_array,n_array, k=self.k,s=self.s)
#             if(len(eos_array_adiabatic)==4):
#                 self.eos_array_adiabatic=eos_array_adiabatic
#                 self.eosCs2_adiabatic_int = interpolate.UnivariateSpline(self.eos_array_adiabatic[2],self.eos_array_adiabatic[3], k=max(self.k-1,1),s=self.s)
#                 self.has_cs2_adiabatic=True
#             else:
#                 self.has_cs2_adiabatic=False
#             if(len(eos_array_Ylep)==4):
#                 self.eos_array_Ylep=eos_array_Ylep
#                 self.eosYlep_int = interpolate.UnivariateSpline(self.eos_array_Ylep[2],self.eos_array_Ylep[3], k=max(self.k-1,1),s=self.s)
#                 self.eosYqak_int = lambda x:0
#                 self.has_Ylep=True
#                 self.has_Yqak=False
#             elif(len(eos_array_Ylep)==5):
#                 self.eos_array_Ylep=eos_array_Ylep
#                 self.eosYlep_int = interpolate.UnivariateSpline(self.eos_array_Ylep[2],self.eos_array_Ylep[3], k=max(self.k-1,1),s=self.s)
#                 self.eosYqak_int = interpolate.UnivariateSpline(self.eos_array_Ylep[2],self.eos_array_Ylep[4], k=max(self.k-1,1),s=self.s)
#                 self.has_Ylep=True
#                 self.has_Yqak=True
#             else:
#                 self.has_Ylep=False
#                 self.has_Yqak=False
#             self.chempo_surface=(pressure_array[0]+energy_array[0])/n_array[0]
#             self.baryon_density_s=baryon_density_s
#             self.pressure_s=self.eosPressure_frombaryon(self.baryon_density_s)
#             self.density_s=self.eosDensity(self.pressure_s)
#             self.unit_mass=c**4/(G**3*self.density_s*1e51*e)**0.5
#             self.unit_radius=c**2/(G*self.density_s*1e51*e)**0.5
#             self.unit_N=self.unit_radius**3*self.baryon_density_s*1e45
#         else:
#             tmp=self.success_eos_array
#             for item_name in [*self.__dict__]:  #this two lines is added on 8/15/2019
#                 del self.__dict__[item_name]    #to try free some memories.
#             self.success_eos=False
#             self.success_eos_array=tmp
#             self.eosPressure_frombaryon,self.eosDensity,self.eosBaryonDensity = ['undefined']*3
#     def __getstate__(self):
#         state = self.__dict__.copy()
#         for dict_intepolation in ['eosPressure_frombaryon','eosDensity','eosBaryonDensity']:
#             del state[dict_intepolation]
#         if(self.has_cs2_adiabatic):
#             del state['eosCs2_adiabatic_int']
#         if(self.has_Ylep):
#             del state['eosYlep_int']
#         if(self.has_Yqak):
#             del state['eosYqak_int']
#         return state
#     def __setstate__(self, state):
#         self.__dict__.update(state)
#         if(self.success_eos):
#             n_array,energy_array,pressure_array=self.eos_array
#             self.eosPressure_frombaryon = interpolate.UnivariateSpline(n_array,pressure_array, k=self.k,s=self.s)
#             self.eosDensity  = interpolate.UnivariateSpline(pressure_array,energy_array, k=self.k,s=self.s)
#             self.eosBaryonDensity = interpolate.UnivariateSpline(pressure_array,n_array, k=self.k,s=self.s)
#             if(self.has_cs2_adiabatic):
#                 self.eosCs2_adiabatic_int = interpolate.UnivariateSpline(self.eos_array_adiabatic[2],self.eos_array_adiabatic[3], k=max(self.k-1,1),s=self.s)
#             if(self.has_Ylep):
#                 self.eosYlep_int = interpolate.UnivariateSpline(self.eos_array_Ylep[2],self.eos_array_Ylep[3], k=max(self.k-1,1),s=self.s)
#             if(self.has_Yqak):
#                 self.eosYqak_int = interpolate.UnivariateSpline(self.eos_array_Ylep[2],self.eos_array_Ylep[4], k=max(self.k-1,1),s=self.s)
#         else:
#             self.eosPressure_frombaryon,self.eosDensity,self.eosBaryonDensity = ['undefined']*3
#     def eosChempo(self,pressure):
#         return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
#     def eosCs2(self,pressure):
#         return 1.0/derivative(self.eosDensity,pressure,dx=self.eosDensity(pressure)*dlnx_cs2)
#     def eosCs2_adiabatic(self,pressure):
#         return np.where(np.logical_and(self.eos_array_adiabatic[2].min()<=pressure,pressure<=self.eos_array_adiabatic[2].max()),self.eosCs2_adiabatic_int(pressure),self.eosCs2(pressure))
#     def eosYlep(self,pressure):
#         return np.where(np.logical_and(self.eos_array_Ylep[2].min()<=pressure,pressure<=self.eos_array_Ylep[2].max()),self.eosYlep_int(pressure),0.5)
#     def eosYqak(self,pressure):
#         return np.where(np.logical_and(self.eos_array_Ylep[2].min()<=pressure,pressure<=self.eos_array_Ylep[2].max()),self.eosYqak_int(pressure),0.5)
#     def setMaxmass(self,result_maxmaxmass):
#         self.pc_max,self.mass_max,self.cs2_max=result_maxmaxmass
#         self.maxmass_success=self.cs2_max<1 and 2<self.mass_max<5 and self.pc_max<self.p_max
#         self.eos_success_all=self.maxmass_success and self.success_eos
#         return self.eos_success_all
#     def setProperity(self,Properity_one,Properity_onepointfour):
#         self.properity_one,self.properity_onepointfour=Properity_one,Properity_onepointfour
# =============================================================================

class EOS_attach(EOS_interpolation):
    def __init__(self,baryon_density_s,eos_low,eos_high,quality_control_f=quality_control_basic,s_k=[0,2],nb_range=[0.01,0.1],match_method=1):
        nb_in_range_low=np.logical_and(eos_low.eos_array[0]>nb_range[0],eos_low.eos_array[0]<nb_range[1])
        nb_in_range_high=np.logical_and(eos_high.eos_array[0]>nb_range[0],eos_high.eos_array[0]<nb_range[1])
        get_match_index=((1-eos_low.eos_array[:,:,np.newaxis]/eos_high.eos_array[:,np.newaxis,:])**2).sum(axis=0)
        get_match_index=np.where(get_match_index==get_match_index[nb_in_range_low][:,nb_in_range_high].min())
        if(match_method==1): #find the closest point i,j of two EOSs and use eos_low to i-1, use eos_high above j+1, and include the matching point corresponds to average of i in eos_low and j in eos_high
            match_point=0.5*(eos_low.eos_array[:,get_match_index[0]]+eos_high.eos_array[:,get_match_index[1]])
            eos_array=np.concatenate((eos_low.eos_array[:,:get_match_index[0][0]],match_point,eos_high.eos_array[:,get_match_index[1][0]+1:]),axis=1)
        EOS_interpolation.__init__(self,baryon_density_s,eos_array,s_k=s_k,quality_control_f=quality_control_f)

class EOS_Incompressible(object):
    baryon_density_s=0.16
    def __init__(self,args):
        self.eps,self.chempo_surface=args
        self.baryondensity=self.eps/self.chempo_surface
        self.pressure_s=self.eps
        self.density_s=self.eps
        self.unit_mass=c**4/(G**3*self.density_s*1e51*e)**0.5
        self.unit_radius=c**2/(G*self.density_s*1e51*e)**0.5
        self.unit_N=self.unit_radius**3*self.baryon_density_s*1e45
    def eosDensity(self,pressure):
        return self.eps
    def eosBaryonDensity(self,pressure):
        return self.baryondensity
    def eosChempo(self,pressure):
        return (pressure+self.eps)/self.baryondensity
    def eosCs2(self,pressure):
        return np.infty
    def eosMRbeta(self,pc):
        beta = (9-(2/(1+3*pc/self.eps)+1)**2)/18
        R=toLength((3*beta/(4*np.pi*self.eps))**0.5,'mev-1fm3_sqr')
        M=toMass(R*beta,'radius_cm')
        return np.array([M,R,beta])

class EOS_SLY4(EOS_interpolation):
    def __init__(self,s_k=[0,1]):
        eos_array=np.loadtxt('./EOS_Tables_Ozel/sly.dat',skiprows=0)
        eos_array[:,0]=toMevfm(eos_array[:,0]/1.66*1e24,'baryondensity')
        eos_array[:,1]=toMevfm(eos_array[:,1],'density')
        eos_array[:,2]=toMevfm(eos_array[:,2],'density')
        eos_array[:,1:]=eos_array[:,[2,1]]
        eos_array=eos_array.transpose()
        EOS_interpolation.__init__(self,0.159,eos_array,s_k=s_k)

class EOS_hybrid1(object):
    baryon_density_s=0.16
    def __init__(self,p_trans,eos_low,eos_high,allow_unstable=False): #defalt s=0,k=3 equal quadratic 1d intepolation
        self.p_trans=p_trans
        self.eos_low,self.eos_high=copy.deepcopy([eos_low,eos_high]) #!!!this'not deep copy here!!!can over write!!
        if(eos_low.success_eos and eos_high.success_eos):
            self.det_density=eos_high.eosDensity(p_trans)-eos_low.eosDensity(p_trans)
            self.success_eos=self.det_density>=0 or allow_unstable
            self.n_trans_low = eos_low.eosBaryonDensity(p_trans)
            self.n_trans_high=eos_high.eosBaryonDensity(p_trans)
            self.p_max=eos_high.p_max
            #self.eos_array=np.concatenate((eos_low.eos_array[:,eos_low.eos_array[2]<p_trans],eos_high.eos_array[:,eos_high.eos_array[2]<p_trans]),axis=1)
        else:
            self.success_eos=False
        if(self.success_eos):
            self.chempo_surface=eos_low.chempo_surface
            self.eos_low.baryon_density_s=self.baryon_density_s
            self.eos_high.baryon_density_s=self.baryon_density_s
            self.pressure_s=self.eosPressure_frombaryon(self.baryon_density_s)
            self.eos_low.pressure_s=self.pressure_s
            self.eos_high.pressure_s=self.pressure_s
            self.density_s=self.eosDensity(self.pressure_s)
            self.eos_low.density_s=self.density_s
            self.eos_high.density_s=self.density_s
            self.unit_mass=c**4/(G**3*self.density_s*1e51*e)**0.5
            self.eos_low.unit_mass=self.unit_mass
            self.eos_high.unit_mass=self.unit_mass
            self.unit_radius=c**2/(G*self.density_s*1e51*e)**0.5
            self.eos_low.unit_radius=self.unit_radius
            self.eos_high.unit_radius=self.unit_radius
            self.unit_N=self.unit_radius**3*self.baryon_density_s*1e45
            self.eos_low.unit_N=self.unit_N
            self.eos_high.unit_N=self.unit_N
            self.has_cs2_adiabatic=False
        else:
            self.eosPressure_frombaryon,self.eosDensity,self.eosBaryonDensity,self.eosChempo,self.eosCs2 = ['undefined hybrid EoS']*5
    def eosPressure_frombaryon(self,baryondensity):
        return np.where(baryondensity<self.n_trans_low,self.eos_low.eosPressure_frombaryon(baryondensity),np.where(baryondensity<self.n_trans_high,self.p_trans,self.eos_high.eosPressure_frombaryon(baryondensity)))
    def eosDensity(self,pressure):
        return np.where(pressure<self.p_trans,self.eos_low.eosDensity(pressure),self.eos_high.eosDensity(pressure))
    def eosBaryonDensity(self,pressure):
        return np.where(pressure<self.p_trans,self.eos_low.eosBaryonDensity(pressure),self.eos_high.eosBaryonDensity(pressure))
    def eosChempo(self,pressure):
        return np.where(pressure<self.p_trans,self.eos_low.eosChempo(pressure),self.eos_high.eosChempo(pressure))
    def eosCs2(self,pressure):
        return np.where(pressure<self.p_trans,self.eos_low.eosCs2(pressure),self.eos_high.eosCs2(pressure))
    def setMaxmass(self,result_maxmaxmass):
        self.pc_max,self.mass_max,self.cs2_max=result_maxmaxmass
        self.maxmass_success=self.cs2_max<1 and 2<self.mass_max<5 and self.pc_max<self.p_max
        self.eos_success_all=self.maxmass_success and self.success_eos
        return self.eos_success_all
    def setProperity(self,Properity_one,Properity_onepointfour):
        self.properity_one,self.properity_onepointfour=Properity_one,Properity_onepointfour
    def setGmode(self,omega_g_cen):
        self.omega_g_cen_real=omega_g_cen.real
        self.omega_g_cen_imag=omega_g_cen.imag