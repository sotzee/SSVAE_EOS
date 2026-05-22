#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 17:31:33 2018

@author: sotzee
"""
import numpy as np
from tov_f import f,f_lnR,f_R2,f_baryon_number,f_lepton_number,f_quark_number,f_tidal,f_MRI,f_Newton,f_phi,f_HKWX,f_HKWX_lnR,f_complete,lsoda_ode,lsoda_ode_array
from astropy.constants import M_sun
from scipy.constants import m_n,c,pi

def Radius_correction_ratio(pc,Preset_Pressure_final,beta,eos):
    X=(eos.eosChempo(pc*Preset_Pressure_final)/(eos.chempo_surface))**2-1
    return beta/(beta+beta*X-0.5*X)
def Radius_correction_ratio_OFF(pc,Preset_Pressure_final,beta,eos):
    return 1

def MassRadius(pressure_center,Preset_Pressure_final,Preset_rtol,MRorMRBIT,eos,Radius_correction_ratio=Radius_correction_ratio,l=2):
    x0 = -np.log(pressure_center/eos.density_s)
    xf = x0-np.log(Preset_Pressure_final)
    if(MRorMRBIT=='M'):
        r = lsoda_ode(f,Preset_rtol,[0.,0.],x0,xf,eos)
        M=r.y[0]*eos.unit_mass/M_sun.value
        return M
    elif(MRorMRBIT=='B'):
        r = lsoda_ode(f_baryon_number,Preset_rtol,[0.,0.,0.,],x0,xf,eos)
        M_binding=r.y[2]*eos.unit_N*m_n/M_sun.value
        return M_binding
    elif(MRorMRBIT=='NbNp'):
        r = lsoda_ode(f_lepton_number,Preset_rtol,[0.,0.,0.,0.],x0,xf,eos)
        Nb=r.y[2]*eos.unit_N
        Np=r.y[3]*eos.unit_N
        return [Nb,Np]
    elif(MRorMRBIT=='NbNq'):
        r = lsoda_ode(f_quark_number,Preset_rtol,[0.,0.,0.,0.],x0,xf,eos)
        Nb=r.y[2]*eos.unit_N
        Nq=r.y[3]*eos.unit_N
        return [Nb,Nq]
    elif(MRorMRBIT=='MR'):
        r = lsoda_ode(f,Preset_rtol,[0.,0.],x0,xf,eos)
        M=r.y[0]*eos.unit_mass/M_sun.value
        R=r.y[1]**0.5*eos.unit_radius
        beta=r.y[0]/R*eos.unit_radius
        R=R*Radius_correction_ratio(pressure_center,Preset_Pressure_final,beta,eos)
        return [M,R]
    elif(MRorMRBIT=='MRT'):
        r = lsoda_ode(f_tidal,Preset_rtol,[0.,0.,2.],x0,xf,eos)
        M=r.y[0]*eos.unit_mass/M_sun.value
        R=r.y[1]**0.5*eos.unit_radius
        beta=r.y[0]/R*eos.unit_radius
        radius_correction_ratio=Radius_correction_ratio(pressure_center,Preset_Pressure_final,beta,eos)
        R=R*radius_correction_ratio
        det_k2_trans=eos.eosDensity(pressure_center*Preset_Pressure_final)/(eos.density_s*(r.y[0]/(4*np.pi*r.y[1]**1.5)+np.exp(-xf)))
        yR=r.y[2]-det_k2_trans
        tidal_R=6*beta*(2-yR+beta*(5*yR-8))+4*beta**3*(13-11*yR+beta*(3*yR-2)+2*beta**2*(1+yR))+3*(1-2*beta)**2*(2-yR+2*beta*(yR-1))*np.log(1-2*beta)
        k2=8.0/5.0*beta**5*(1-2*beta)**2*(2-yR+2*beta*(yR-1))/tidal_R
        tidal=2.0/3.0*(k2/beta**5)
        beta=beta/radius_correction_ratio
        return [M,R,beta,k2,tidal]
    elif(MRorMRBIT=='MRf'):
        r = lsoda_ode(f_Newton,Preset_rtol,[0.,0.,0.,0.],x0,xf,[eos,l])
        beta=r.y[0]
        R=r.y[1]**0.5*eos.unit_radius
        M=R*beta*eos.unit_mass/(M_sun.value*eos.unit_radius)
        radius_correction_ratio=Radius_correction_ratio(pressure_center,Preset_Pressure_final,beta,eos)
        Omega=(2*l*(l-1)*r.y[2]*r.y[1]*r.y[0]**2/((2*l+1)*r.y[3]))**0.5
        print((2*l*(l-1)*r.y[2]*r.y[1]/((2*l+1)*r.y[3]*r.y[0])))
        R=R*radius_correction_ratio
        beta=beta/radius_correction_ratio
        return [M,R,beta,Omega]
    elif(MRorMRBIT=='MRphi'):
        r = lsoda_ode(f_phi,Preset_rtol,[0.,0.,0.,0.],x0,xf,eos)
        M=r.y[0]*eos.unit_mass/M_sun.value
        R=r.y[1]**0.5*eos.unit_radius
        beta=r.y[0]/R*eos.unit_radius
        Phi0=-r.y[2]+0.5*np.log(1-2*beta)
        radius_correction_ratio=Radius_correction_ratio(pressure_center,Preset_Pressure_final,beta,eos)
        R=R*radius_correction_ratio
        beta=beta/radius_correction_ratio
        a=r.y[3]*eos.unit_N
        return [M,R,beta,Phi0,a]
    elif(MRorMRBIT=='MRBIT'):
        r = lsoda_ode(f_complete,Preset_rtol,[0.,0.,0.,0.,2.],x0,xf,eos)
        M=r.y[0]*eos.unit_mass/M_sun.value
        R=r.y[1]**0.5*eos.unit_radius
        beta=r.y[0]/R*eos.unit_radius
        radius_correction_ratio=Radius_correction_ratio(pressure_center,Preset_Pressure_final,beta,eos)
        R=R*radius_correction_ratio
        N=r.y[2]*eos.unit_N
        M_binding=N*m_n/M_sun.value
        momentofinertia=r.y[3]/(6.0+2.0*r.y[3])/beta**3
        det_k2_trans=eos.eosDensity(pressure_center*Preset_Pressure_final)/(eos.density_s*(r.y[0]/(4*np.pi*r.y[1]**1.5)+np.exp(-xf)))
        yR=r.y[4]-det_k2_trans
        tidal_R=6*beta*(2-yR+beta*(5*yR-8))+4*beta**3*(13-11*yR+beta*(3*yR-2)+2*beta**2*(1+yR))+3*(1-2*beta)**2*(2-yR+2*beta*(yR-1))*np.log(1-2*beta)
        k2=8.0/5.0*beta**5*(1-2*beta)**2*(2-yR+2*beta*(yR-1))/tidal_R
        tidal=2.0/3.0*(k2/beta**5)
        beta=beta/radius_correction_ratio
        return [M,R,beta,M_binding,momentofinertia,k2,tidal]

def MassRadiusPhi_lnR(pressure_center,Preset_Radius_final,Preset_rtol,N,eos):
    p0=pressure_center/eos.density_s
    x0=np.log(1/eos.unit_radius)
    xf=np.log(Preset_Radius_final/eos.unit_radius)
    xf_array = np.linspace(x0,xf,N)
    y_array = lsoda_ode_array(f_lnR,Preset_rtol,[0.,p0,0.],x0,xf_array,eos)
    # runtime warning due to zeros at star center
    M_array=y_array[:,0]*np.exp(xf_array)*(eos.unit_mass/M_sun.value)
    r_array=np.exp(xf_array)*eos.unit_radius
    beta_array=y_array[:,0]
    Phi_array=(y_array[:,2]-y_array[-1,2])+0.5*np.log(1-2*beta_array[-1])
    return  [y_array,xf_array,M_array,r_array,beta_array,Phi_array]

def MassRadiusPhi_R2(pressure_center,Preset_Radius_final,Preset_rtol,N,eos):
    p0=pressure_center/eos.density_s
    r2f=(Preset_Radius_final/eos.unit_radius)**2
    r2_array = np.linspace(0,r2f,N)
    y_array = lsoda_ode_array(f_R2,Preset_rtol,[0.,p0,0.],0,r2_array,eos)
    # runtime warning due to zeros at star center
    M_array=y_array[:,0]*r2_array**0.5*(eos.unit_mass/M_sun.value)
    r_array=r2_array**0.5*eos.unit_radius
    beta_array=y_array[:,0]
    Phi_array=(y_array[:,2]-y_array[-1,2])+0.5*np.log(1-2*beta_array[-1])
    return  [y_array,r2_array,M_array,r_array,beta_array,Phi_array]

def MassRadiusPhi_profile(pressure_center,Preset_Pressure_final,Preset_rtol,N,eos):
    x0 = -np.log(pressure_center/eos.density_s)
    xf = x0-np.log(Preset_Pressure_final)
    xf_array = np.linspace(x0,xf,N)
    y_array = lsoda_ode_array(f_phi,Preset_rtol,[0.,0.,0.,0.],x0,xf_array,eos)
    # runtime warning due to zeros at star center
    M_array=y_array[:,0]*eos.unit_mass/M_sun.value
    r_array=y_array[:,1]**0.5*eos.unit_radius
    beta_array=np.concatenate(([0],y_array[1:,0]/(r_array[1:])*eos.unit_radius))
    Phi_array=(y_array[:,2]-y_array[-1,2])+0.5*np.log(1-2*beta_array[-1])
    a_array=y_array[:,3]*eos.unit_N#/y_array[-1,3]
    return  [y_array,xf_array,M_array,r_array,beta_array,Phi_array,a_array]

def init_HKWX(exp_nu_sqr,pressure_center,r2_cen,K_cen_i,W_cen,eos_omega_l):
    eos,omega,l=eos_omega_l
    density_cen=eos.eosDensity(pressure_center)/eos.density_s
    pressure_cen=pressure_center/eos.density_s
    beta_cen=4*pi*r2_cen*density_cen/3
    phi_cen=np.log(exp_nu_sqr)+2*pi*(density_cen/3+pressure_cen)*r2_cen
    K_cen=(density_cen+pressure_cen)*K_cen_i
    X_cen=(density_cen+pressure_cen)*exp_nu_sqr*((4*pi*(density_cen/3+pressure_cen)-omega**2/(l*exp_nu_sqr**2))*W_cen+0.5*K_cen)
    H_cen=(2*l*K_cen+16*pi*(density_cen+pressure_cen)*W_cen)/(l*(l+1))
    #print(m_cen,r2_cen,phi_cen)
    return beta_cen,r2_cen,phi_cen,H_cen,K_cen,W_cen,X_cen

def f_Zerilli_q(r,y,M_omega_l):
    K_hat,R_hat=y
    M,omega,l=M_omega_l
    beta=M/r
    n=0.5*(l+2)*(l-1) #n_tmp in some other codes of mine
    Vz=2*(1-2*beta)*(n**2*(n+1)+3*n**2*beta+9*n*beta**2+9*beta**3)/(r*(n+3*beta))**2
    dK_hat_dr=R_hat/(1-2*beta)
    dR_hat_dr=(Vz-omega**2)*K_hat/(1-2*beta)
    return np.array([dK_hat_dr,dR_hat_dr])

def Zerilli_WKB(r2,M_omega_l): #checked 01/16/2021
    M,omega,l=M_omega_l
    n=0.5*(l+2)*(l-1)
    beta=M/r2**0.5
    Vz_numeritor=(n**2*(n+1)+3*n**2*beta+9*n*beta**2+9*beta**3)
    Vzdbeta_numeritor=3*n**2+18*n*beta+27*beta**2
    Vz_denominator=((n+3*beta)**2)
    Vzdbeta_denominator=6*(n+3*beta)
    Vz=2*(1-2*beta)*Vz_numeritor/(Vz_denominator*r2)
    dVzdr=Vz*(-2/r2**0.5-M/r2*(-2/(1-2*beta)+Vzdbeta_numeritor/Vz_numeritor-Vzdbeta_denominator/Vz_denominator))
    U=(omega**2-Vz+(2*beta-3*beta**2)/r2)/(1-2*beta)**2
    dUdr=-4*M*U/(r2*(1-2*beta))+(-dVzdr-6*M/r2**2+12*M**2/r2**2.5)/(1-2*beta)**2
    return [U,dUdr]

def Zerilli_map(y_surface,Preset_rtol,eos_omega_l): #Eq5-Eq16 from Edward D. Fackerell 1971
    beta,r2,phi,H,K,W,X=y_surface
    #print('==================')
    #print('H,K,W,X=',H,K,W,X)
    eos,omega,l=eos_omega_l
    n_tmp=0.5*(l+2)*(l-1)
    beta=np.real(beta)
    r2=np.real(r2)
    beta2=beta**2
    r=r2**0.5
    M=beta*r
    g_time_r=(n_tmp*(n_tmp+1)+3*n_tmp*beta+6*beta2)/(n_tmp+3*beta)
    h=1j*(-n_tmp+3*n_tmp*beta+3*beta2)/((1-2*beta)*(n_tmp+3*beta))
    k_over_r=-1j/(1-2*beta)
    K_time_r2=r2*K
    R_time_romega=r*H/omega
    matrix=np.array([[g_time_r,1],[h,k_over_r]])  #[K_time_r2,R_time_romega]=np.dot(matrix,[K_hat_time_r,R_hat_time_r2])
    K_hat_time_r,R_hat_time_r2=np.dot(np.linalg.inv(matrix),[K_time_r2,R_time_romega])
    #K_hat=Z, and R_hat=dZ/dr* as in many literature!!!
    #print(matrix)
    #print([K,H])
    #print([K_time_r2,R_time_romega])
    init=[K_hat_time_r/r,R_hat_time_r2/r2]
    M_omega_l=[beta*r,omega,l]
    U,dUdr=Zerilli_WKB(r2,M_omega_l)
    q=U**0.5
    dqdr_over2q=dUdr/(2*q)**2
    A_out=init[0]*((1-2*beta)*(q*1j-dqdr_over2q)-M/r2)-init[1]
    A_in =init[0]*((1-2*beta)*(q*1j-dqdr_over2q)+M/r2)+init[1]
    return A_in/np.abs(A_out)
    
# =============================================================================
#     rf=25/np.abs(omega)
#     r_array=np.linspace(r,rf,100)
#     #print(np.abs(omega))
#     #print(25/np.abs(omega),r_array)
#     result_array=lsoda_ode_array(f_Zerilli_q,Preset_rtol,init,r,r_array,M_omega_l,method='zvode')
#     result_array=np.concatenate((r_array[:,np.newaxis],result_array),axis=1)
#     
#     alpha1=-1j*(n_tmp+1)/omega
#     alpha2=(-n_tmp*(n_tmp+1)+1j*M*omega*(1.5+3/n_tmp))/(2*omega**2)
#     K_out=(1+alpha1/rf+alpha2/rf**2)
#     K_in =np.conjugate(K_out)
#     R_out=-1j*omega*(1+alpha1/rf+(alpha2+1j*alpha1/omega*(1-2*M/rf))/rf**2)
#     R_in =np.conjugate(R_out)
#     KR_matrix=np.array([[K_out,K_in],[R_out,R_in]])
#     A_out,A_in=np.dot(np.linalg.inv(KR_matrix),result_array[-1,1:])
# # =============================================================================
# #     print('=============')
# #     print(alpha1,alpha2)
# #     print(K_out,K_in)
# #     print(R_out,R_in)
# #     print(A_out,A_in)
# #     print(np.abs(A_in/A_out))
# # =============================================================================
#     #return np.abs(A_in/A_out)
#     return A_in/np.abs(A_out)
# =============================================================================
    
    
def MassRadius_HKWX_profile(pressure_center,Preset_Pressure_final,Preset_rtol,N,eos,omega_dimentionful=2000+5j,Phi_cen=0,l=2,r2_cen=1e-10):
    if(Phi_cen==0):
        mr_result=MassRadiusPhi_profile(pressure_center,Preset_Pressure_final,Preset_rtol,100,eos)
        Phi_cen=mr_result[5][0]
    if(np.isreal(omega_dimentionful)):
        method='lsoda'
    else:
        method='zvode'
    x0 = -np.log(pressure_center/eos.density_s)
    xf = x0-np.log(Preset_Pressure_final)
    xf_array = np.linspace(x0,xf,N)
    exp_nu_sqr=np.exp(Phi_cen)
    omega=omega_dimentionful*eos.unit_radius/c
    eos_omega_l=[eos,omega,l]
    y_array_list=[]
    W_cen=1
    for K_cen_i in [-1,1]:
        init=init_HKWX(exp_nu_sqr,pressure_center,r2_cen,K_cen_i,W_cen,eos_omega_l)
        print('hello!!', omega,init,f_HKWX(x0,init,eos_omega_l))
        #print(f_phi(x0,init[:3],eos))
        y_array_list.append(lsoda_ode_array(f_HKWX,Preset_rtol,init,x0,xf_array,eos_omega_l,method=method))
    y_array=y_array_list[0]*y_array_list[1][-1,-1]-y_array_list[1]*y_array_list[0][-1,-1]
    y_array[:,:3]=0.5*(y_array_list[0][:,:3]+y_array_list[1][:,:3])
    y_surface=y_array[-1]
    if(np.abs(np.exp(2*y_surface[2])-(1-2*y_surface[0]))>1e-6):
        print('phi0 ERROR at surface. y_surface=',y_surface)
    Zerilli_result=Zerilli_map(y_surface,Preset_rtol,eos_omega_l)
# =============================================================================
#     # runtime warning due to zeros at star center
#     M_array=y_array[:,0]*eos.unit_mass/M_sun.value
#     r_array=y_array[:,1]**0.5*eos.unit_radius
#     beta_array=np.concatenate(([0],y_array[1:,0]/(r_array[1:])*eos.unit_radius))
#     Phi_array=y_array[:,2]
# =============================================================================
    return  y_array,Zerilli_result

# =============================================================================
# def MassRadius_HKWX_profile_new(pressure_center,Preset_Radius_final,Preset_rtol,N,eos,omega_dimentionful=2000,l=2,r2_cen=1e-10):
#     Preset_Pressure_final=1e-8
#     
#     mr_result=MassRadiusPhi_profile(pressure_center,Preset_Pressure_final,Preset_rtol,100,eos)
#     p0=pressure_center/eos.density_s
#     x0 = np.log(r2_cen)
#     xf = np.log(Preset_Radius_final/eos.unit_radius)
#     xf_array = np.linspace(x0,xf,N)
#     
#     Phi_cen=mr_result[5][0]
#     exp_nu_sqr=np.exp(Phi_cen)
#     omega=omega_dimentionful*eos.unit_radius/c
#     eos_omega_l=[eos,omega,l]
#     y_array_list=[]
#     W_cen=1
#     for K_cen_i in [-1,1]:
#         init=list(init_HKWX(exp_nu_sqr,pressure_center,r2_cen,K_cen_i,W_cen,eos_omega_l))
#         init[1]=p0
#         #print('hello!!', omega,init,f_HKWX(x0,init,eos_omega_l))
#         #print(f_phi(x0,init[:3],eos))
#         y_array_list.append(lsoda_ode_array(f_HKWX_lnR,Preset_rtol,init,x0,xf_array,eos_omega_l))
#     y_surface=y_array_list[0][-1]*y_array_list[1][-1,-1]-y_array_list[1][-1]*y_array_list[0][-1,-1]
#     HKWX_surface=y_surface[-4:]
# # =============================================================================
# #     # runtime warning due to zeros at star center
# #     M_array=y_array[:,0]*eos.unit_mass/M_sun.value
# #     r_array=y_array[:,1]**0.5*eos.unit_radius
# #     beta_array=np.concatenate(([0],y_array[1:,0]/(r_array[1:])*eos.unit_radius))
# #     Phi_array=y_array[:,2]
# # =============================================================================
#     return  [y_array_list,HKWX_surface]
# =============================================================================

def Mass_formax(pressure_center,Preset_Pressure_final,Preset_rtol,eos):#(this function is used for finding maxmass in FindMaxmass.py ONLY!!
    if(pressure_center[0]<=0):
        return 0
    x0 = -np.log(pressure_center/eos.density_s)
    xf = x0-np.log(Preset_Pressure_final)
    r = lsoda_ode(f,Preset_rtol,[0.,0.],x0,xf,eos)
    return -r.y[0]*eos.unit_mass/M_sun.value

# =============================================================================
# def Tidal_corrected(pc,Preset_Pressure_final,beta,yR,eos):
#     radius_correction=Radius_correction_ratio(pc,Preset_Pressure_final,beta,eos)
#     a=(5-8*beta)/(1-2*beta)
#     b=(beta/(0.5-beta))**2-4*beta/(0.5-beta)
#     print '+++++++++++',a,b,radius_correction
#     zb=yR-2
#     check_solution_type=a**2-4*b
#     check_solution_type_sqrt=np.abs(check_solution_type)**0.5
#     print zb
#     zR=np.where(check_solution_type>0,
#                 ((2*check_solution_type_sqrt/(1-radius_correction**(1/check_solution_type**0.5)*(2*zb-check_solution_type_sqrt+a)/(2*zb+check_solution_type_sqrt+a)))-check_solution_type_sqrt-a)/2.,
#                 (check_solution_type_sqrt*np.tan(np.log(radius_correction)*(check_solution_type_sqrt)+np.arctan((2*zb+a)/check_solution_type_sqrt))-a)/2.)
#     yR=zR+2
#     print zR
#     beta=beta/Radius_correction_ratio(pc,Preset_Pressure_final,beta,eos)
#     tidal_R=6*beta*(2-yR+beta*(5*yR-8))+4*beta**3*(13-11*yR+beta*(3*yR-2)+2*beta**2*(1+yR))+3*(1-2*beta)**2*(2-yR+2*beta*(yR-1))*np.log(1-2*beta)
#     k2=8.0/5.0*beta**5*(1-2*beta)**2*(2-yR+2*beta*(yR-1))/tidal_R
#     tidal=2.0/3.0*(k2/beta**5)
#     return tidal
# =============================================================================

# =============================================================================
# def get_radius_corr(zR,zb,a,b):
#     return ((2*zR-(a**2-4*b)**0.5+a)/(2*zR+(a**2-4*b)**0.5+a)*(2*zb+(a**2-4*b)**0.5+a)/(2*zb-(a**2-4*b)**0.5+a))**((a**2-4*b)**0.5)
# =============================================================================


def f_momentofinertia(x, y, eos):
    p=np.exp(-x)
    p_dimentionful=p*eos.density_s
    eps=eos.eosDensity(p_dimentionful)/eos.density_s
    if(y[1]==0):
        den=p/((eps+p)*(eps/3.0+p))
        dmdx=0#np.sqrt(y[1])*eps*den
        dr2dx=0.5/np.pi*den
        djdx=0
        dzdx=(4+y[3])/(eps/p/3.0+1)
        dwdx=0
    else:
        r=y[1]**0.5
        den=p/((y[0]+4*np.pi*y[1]*r*p)*(eps+p))
        rel=1-2*y[0]/r
        dmdx=4*np.pi*eps*y[1]**2*rel*den
        dr2dx=2*y[1]*r*rel*den
        djdx=-4*np.pi*p*y[1]*r/(y[0]+4*np.pi*y[1]*r*p)
        dzdx=((4+y[3])*4*np.pi*(eps+p)*y[1]-rel*y[3]*(3+y[3]))*r*den
        dwdx=y[3]/2/y[1]*dr2dx
    return np.array([dmdx,dr2dx,djdx,dzdx,dwdx])

def MomentOfInertia_profile(pressure_center,Preset_Pressure_final,Preset_rtol,N,eos):
    x0 = -np.log(pressure_center/eos.density_s)
    xf = x0-np.log(Preset_Pressure_final)
    xf_array = np.linspace(x0,xf,N)
    p=np.exp(-xf_array)*eos.density_s
    bds=eos.eosBaryonDensity(p)*939.5654
    y_array = lsoda_ode_array(f_momentofinertia,Preset_rtol,[0.,0.,0.,0.,0.],x0,xf_array,eos)
    # runtime warning due to zeros at star center
    M_array=y_array[:,0]*eos.unit_mass/M_sun.value
    r_array=y_array[:,1]**0.5
    beta_array=y_array[:,0]/(r_array)
    R_array=r_array*eos.unit_radius
    j_array=np.exp(y_array[:,2]-y_array[-1,2])
    z_array=y_array[:,3]
    w_array=3./(3.+z_array[-1])*np.exp(y_array[:,4]-y_array[-1,4])
    I_array=r_array*y_array[:,1]*j_array*z_array*w_array/(6*y_array[-1,0]**3)
    beta=y_array[-1,0]/r_array[-1]
    I_array_Lattimer=28*np.pi*p*r_array[-1]**3*(1-1.67*beta-0.6*beta**2)*beta/(3*eos.density_s*y_array[-1,0]*(beta**2+2*p*(1+7*beta)*(1-2*beta)/bds))
    I_array_Lattimer=(1-I_array_Lattimer)*I_array[-1]
    I_array_Lattimer2=I_array[-1]-(6*np.pi*p*r_array[-1]**6/(eos.density_s*y_array[-1,0]**4))*(4+z_array[-1])/(3+z_array[-1])**2
    
    I_array_z3z=r_array**3*z_array/(z_array+3)
    I_array_test1=I_array[-1]-(0.5/(beta_array[-1]**3)-I_array)*16*np.pi*p*r_array**3/(3.*eos.density_s*y_array[:,0])
    I_array_test2=I_array[-1]-(0.5/(beta_array**3)-I_array)*16*np.pi*p*r_array**3/(3.*eos.density_s*y_array[:,0])
    I_array_test3=0.5/(beta_array**3)-(0.5/(beta_array**3)-I_array[-1])/(1-16*np.pi*p*r_array**3/(3.*eos.density_s*y_array[:,0]))
    return  [eos.density_s*np.exp(-xf_array),M_array,R_array,j_array,z_array,w_array,I_array,I_array_Lattimer,I_array_Lattimer2,I_array_z3z,I_array_test1,I_array_test2,I_array_test3]
