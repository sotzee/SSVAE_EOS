from jax import jit, lax,grad,jacrev,jacfwd
import jax.numpy as jnp
from scipy import optimize as opt
import numpy

from unitconvert import m_e_MeV,m_muon_MeV,unitMeVfm
m_p_MeV=939
m_n_MeV=939
PI2 = jnp.pi ** 2
coeff_common = 0.075 * (3 * jnp.pi ** 2) ** (2/3)

# === EDF_Skyrme Class ===
class EDF_Skyrme:
    mp = m_p_MeV
    mn = m_n_MeV
    m = 0.5 * (mp + mn)
    m_in_MeVfm2 = unitMeVfm ** (2/3) * m
    
    def __init__(self, args):
        self.args = args
        self.args_t, self.args_x = args
        self.t0, self.t1, self.t2, self.t3, self.t4 = self.args_t
        self.x0, self.x1, self.x2, self.x3, self.alpha = self.args_x
        self.Theta_s = 3 * self.t1 + self.t2 * (4 * self.x2 + 5)
        self.Theta_v = self.t1 * (self.x1 + 2) + self.t2 * (self.x2 + 2)
        # Pre-cache params for JIT functions
        self._params = [
            self.t0, self.t1, self.t2, self.t3, self.t4,
            self.x0, self.x1, self.x2, self.x3, self.alpha,
            self.Theta_s, self.Theta_v, self.m_in_MeVfm2, self.mp, self.mn
        ]

        # Warm-up JIT Compilation
        _ = eosDensity_from_n_jax(0.5, 0.5, self._params)
        _ = eosChempo_from_n_jax(0.5, 0.5, self._params)

        self.ns=self.get_ns()
        self.bulk_properties=self.compute_bulk(self.ns)
        self.bulk_key=jnp.array([self.ns,self.bulk_properties[1,0]]+list(self.bulk_properties[2:,0])+list(self.bulk_properties[2:,2])+[self.alpha])
        #self.nBcc=self.crust_core_nB()

    def get_ns(self,init_guess=0.15,tol=1e-10,max_iter=50):
        return eos_get_ns_jax(self._params, init_guess=init_guess, tol=tol, max_iter=max_iter)

    def compute_bulk(self, ns):
        self.J, self.L, self.K = Sv_L_jax(ns, self._params).transpose()

        self.delta_mass = 8 / ((self.Theta_s - 2 * self.Theta_v) * self.ns * unitMeVfm ** (2/3))
        self.mass_tmp = 8 / (self.Theta_s * self.ns * unitMeVfm ** (2/3))
    
        mp_eff_SNM = self.mp / (1 + self.mp / self.mass_tmp)
        mn_eff_SNM = self.mn / (1 + self.mn / self.mass_tmp)
        mp_eff_PNM = self.mp / (1 + self.mp * (1 / self.mass_tmp - 1 / self.delta_mass))
        mn_eff_PNM = self.mn / (1 + self.mn * (1 / self.mass_tmp + 1 / self.delta_mass))
    
        self.m_eff = jnp.array([(mp_eff_SNM + mn_eff_SNM) / 2, mp_eff_PNM, mn_eff_PNM])
    
        return jnp.array([[ns] * 3, self.m_eff, self.J, self.L, self.K])
        
    def effective_mass_n(self,nB,Y_lep):
        return 8/(8+(self.Theta_s+(self.Theta_s-2*self.Theta_v)*(1-2*Y_lep))*nB*self.m_in_MeVfm2)
    def effective_mass_p(self,nB,Y_lep):
        return 8/(8+(self.Theta_s-(self.Theta_s-2*self.Theta_v)*(1-2*Y_lep))*nB*self.m_in_MeVfm2)
        
    def eosDensity_from_n(self, n_p, n_n):
        return eosDensity_from_n_jax(n_p, n_n, self._params)

    def eosChempo_from_n(self, n_p, n_n):
        return eosChempo_from_n_jax(n_p, n_n, self._params)

    def eosPressure_from_n(self, n_p, n_n):
        return eosPressure_from_n_jax(n_p, n_n, self._params)

    def eosGamma_from_n(self, n_p, n_n):
        return eosGamma_from_n_jax(n_p, n_n, self._params)

    def eosCs2ad_from_n(self, n_p, n_n):
        return eosCs2ad_from_n_jax(n_p, n_n, self._params)

    def eos_array_from_n(self, n_p, n_n):
        return eos_array_from_n_jax(n_p, n_n, self._params)

    def eos_array_tot_from_n(self, n_p, n_n):
        eos_array_np   = eos_array_from_n_jax(n_p, n_n, self._params)
        eos_array_lep  = lepton_eos_array(eos_array_np[-1],m_e_MeV,m_muon_MeV)
        return numpy.array([eos_array_np[0],eos_array_np[1]+eos_array_lep[1],eos_array_np[2]+eos_array_lep[2]])
        
    def eosBeta_eq_from(self, nB, Yp_init=0.1):
        Ylep_tanh_init = numpy.arctanh(2*Yp_init-1)
        return eosBeta_eq_scipy(nB, self._params,Ylep_tanh_init=Ylep_tanh_init)

    def stability_beta(self,nB_list):
        return compute_criteria(nB_list[0],self.eosBeta_eq_from(nB_list[0]),self._params)
        
    def crust_core_nB(self,nB_init=0.05,tol=1e-8,max_iter=30,method='hybr'):
        sol = opt.root(self.stability_beta, [nB_init], method=method, tol=tol, options={'maxfev': max_iter})
        if not sol.success:
            raise RuntimeError(f"Root finding failed: {sol.message}")
        return  sol.x[0]
        
# === JIT-Accelerated Functions ===
from jax import config
config.update("jax_enable_x64", True)

@jit
def eosDensity_from_n_jax(n_p, n_n, params):
    t0, t1, t2, t3, t4, x0, x1, x2, x3, alpha, Theta_s, Theta_v, m_in_MeVfm2, mp, mn = params
    nB    = n_p + n_n
    n_iso = n_p - n_n
    n_p = jnp.maximum(n_p, 1e-20)
    n_n = jnp.maximum(n_n, 1e-20)
    F_5ov3 = ((n_p) ** (5/3) + (n_n) ** (5/3))
    F_8ov3 = ((n_p) ** (8/3) + (n_n) ** (8/3))
    part1 = coeff_common * (
        F_5ov3 * (4 / m_in_MeVfm2 + Theta_v * nB) +
        F_8ov3 * (Theta_s - 2 * Theta_v))
    part2 =  t0 * 0.125 * (3 * nB**2 - (2 * x0 + 1) * n_iso**2)
    part3 =  t3 / 48    * (3 * nB**2 - (2 * x3 + 1) * n_iso**2) * nB ** alpha
    return n_p * mp + n_n * mn + part1 + part2 + part3

@jit
def eosChempo_from_n_jax(n_p, n_n, params):
    dEdn_p = grad(eosDensity_from_n_jax, argnums=0)(n_p, n_n, params)
    dEdn_n = grad(eosDensity_from_n_jax, argnums=1)(n_p, n_n, params)
    return jnp.array([dEdn_p, dEdn_n])

@jit
def eosPressure_from_n_jax(n_p, n_n, params):
    density = eosDensity_from_n_jax(n_p, n_n, params)
    chempo_p, chempo_n = eosChempo_from_n_jax(n_p, n_n, params)
    return chempo_p * n_p + chempo_n * n_n - density
    
@jit
def eos_array_from_n_jax(n_p, n_n, params):
    density = eosDensity_from_n_jax(n_p, n_n, params)
    chempo_p, chempo_n = eosChempo_from_n_jax(n_p, n_n, params)
    pressure = chempo_p * n_p + chempo_n * n_n - density
    chempo_lep = chempo_n - chempo_p
    dPdnp=grad(eosPressure_from_n_jax, argnums=0)(n_p, n_n, params)
    dPdnn=grad(eosPressure_from_n_jax, argnums=1)(n_p, n_n, params)
    n3d2Edn2=dPdnp*n_p + dPdnn*n_n - 2*pressure
    return jnp.array([n_p + n_n, density, pressure, n3d2Edn2, chempo_lep])

@jit
def eosGamma_from_n_jax(n_p, n_n, params):
    dPdnp=grad(eosPressure_from_n_jax, argnums=0)(n_p, n_n, params)
    dPdnn=grad(eosPressure_from_n_jax, argnums=1)(n_p, n_n, params)
    return (dPdnp*n_p + dPdnn*n_n)/eosPressure_from_n_jax(n_p, n_n, params)

@jit
def eosCs2ad_from_n_jax(n_p, n_n, params):
    dPdnp=grad(eosPressure_from_n_jax, argnums=0)(n_p, n_n, params)
    dPdnn=grad(eosPressure_from_n_jax, argnums=1)(n_p, n_n, params)
    return (dPdnp*n_p + dPdnn*n_n)/(eosPressure_from_n_jax(n_p, n_n, params)+eosDensity_from_n_jax(n_p, n_n, params))

@jit
def Sv_L_jax(ns, params):
    # Energy per particle function
    def eos_array_beta(beta):
        n_p = 0.5* ns * (1 - beta)
        n_n = 0.5* ns * (1 + beta)
        return eos_array_from_n_jax(n_p, n_n, params)[1:4]

    # Compute direct values
    nep   = eos_array_beta(0.)*jnp.array([1., 3., 9.])/ns    # Symmetric matter (beta=0)
    nep_n = eos_array_from_n_jax(0., ns, params)[1:4]*jnp.array([1., 3., 9.])/ns    # Pure neutron matter (beta=1)
    
    # Compute derivatives at beta=0 using autodiff
    nep_det= jacrev(jacrev(eos_array_beta))(0.0)*jnp.array([1., 3., 9.])/(2*ns)
    return jnp.array([nep, nep_n-nep,nep_det])

@jit
def function_to_solve(params_to_solve, params_other, saturation_params):
    t0, t1, t2, t3, x0, x1, x2, x3 = params_to_solve
    t4, alpha, m_in_MeVfm2, mp, mn = params_other
    Theta_s = 3 * t1 + t2 * (4 * x2 + 5)
    Theta_v = t1 * (x1 + 2) + t2 * (x2 + 2)
    params = t0, t1, t2, t3, t4, x0, x1, x2, x3, alpha, Theta_s, Theta_v, m_in_MeVfm2, mp, mn

    ns=eos_get_ns_jax(params, init_guess=0.15)
    J, L, K = Sv_L_jax(ns, params).transpose()
    delta_mass = 8 / ((Theta_s - 2 * Theta_v) * ns * unitMeVfm ** (2/3))
    mass_tmp = 8 / (Theta_s * ns * unitMeVfm ** (2/3))
    return jnp.array([ns, mass_tmp, delta_mass, J[0],K[0],J[2],L[2],K[2]])-saturation_params

from scipy import optimize as opt
import numpy
def eosBeta_eq_scipy(nB, params, Ylep_tanh_init=-1.0, tol=1e-10, max_iter=100):
    sol = opt.root(equation_Ylep_scipy, [Ylep_tanh_init], args=(nB, params), method='hybr', tol=tol, options={'maxfev': max_iter})
    
    if not sol.success:
        raise RuntimeError(f"Root finding failed: {sol.message}")

    Ylep_tanh_final = sol.x[0]
    Ylep_final = 0.5 * (numpy.tanh(Ylep_tanh_final) + 1)
    return Ylep_final

#use scipy for this rootfinding because root_scalar which support jit and autodifferenciation only use newton.
#Here 'hybr' is necessary since we used Ylep_tanh
def equation_Ylep_scipy(Ylep_tanh, nB, params):
    t0, t1, t2, t3, t4, x0, x1, x2, x3, alpha, Theta_s, Theta_v, m_in_MeVfm2, mp, mn = params
    Ylep = 0.5 * (numpy.tanh(Ylep_tanh[0]) + 1)
    n_p = Ylep * nB
    n_n = nB - n_p
    chempo = eosChempo_from_n_jax(n_p, n_n, params)
    chempo_lep = chempo[1] - chempo[0]
    k_F_e    = numpy.sqrt(numpy.where(chempo_lep > m_e_MeV, chempo_lep ** 2 - m_e_MeV ** 2, 0))
    n_e = unitMeVfm * k_F_e ** 3 / (3 * PI2)
    k_F_muon = numpy.sqrt(numpy.where(chempo_lep > m_muon_MeV, chempo_lep ** 2 - m_muon_MeV ** 2, 0))
    n_muon = unitMeVfm * k_F_muon ** 3 / (3 * PI2)
    return n_p - n_e - n_muon

@jit
def eos_get_ns_jax(params, init_guess=0.16, tol=1e-10, max_iter=50):
    def body_fn(val):
        ns, err, count = val
        f = eosPressure_from_n_jax(ns / 2, ns / 2, params)
        f_prime = grad(lambda n: eosPressure_from_n_jax(n / 2, n / 2, params))(ns)
        delta = f / jnp.where(jnp.abs(f_prime) > 1e-8, f_prime, 1.0)  # Prevent div by zero
        new_ns = ns - delta
        err = jnp.abs(delta)
        count += 1
        return (new_ns, err, count)

    def cond_fn(val):
        _, err, count = val
        return jnp.logical_and(err > tol, count < max_iter)

    val_init = (init_guess, jnp.array(1e10), 0)
    final_val = lax.while_loop(cond_fn, body_fn, val_init)
    ns_final, _, _ = final_val

    return ns_final



@jit
def get_Etot(n_p, n_n, params):
    eps=eosDensity_from_n_jax(n_p, n_n, params)
    eps_lep=lepton_eos_array_nlep(n_p,m_e_MeV,m_muon_MeV)[1]
    return (eps+eps_lep)/(n_p + n_n)

# @jit
# def compute_criteria(nB, Yp, params):
#     # Define energy as a function of [nB, Yp]
#     def E_func(x):
#         nB,Yp=x
#         return get_Etot(nB*Yp, nB*(1-Yp), params)
    
#     # First derivatives
#     grad_E = grad(E_func)
    
#     # Second derivatives (Hessian)
#     hessian_E = jacfwd(grad_E)
    
#     # Compute derivatives at the given (nB, Yp)
#     hess = hessian_E(jnp.array([nB, Yp]))
    
#     d2EdnB2 = hess[0, 0]
#     d2EdYp2 = hess[1, 1]
#     d2EdnBdYp = hess[0, 1]  # Mixed derivative
    
#     # Determinant of Hessian (stability criterion)
#     criteria = d2EdnB2 * d2EdYp2 - d2EdnBdYp**2
#     return criteria
@jit
def compute_criteria(nB, Yp_nB, params):
    # Define energy density as a function of [nB, Yp*nB]; Using all extensive variables (July 2026) refer to Kubis 2008.
    def E_func(x):
        nB,Yp_nB=x
        Yp=Yp_nB/nB
        return eosDensity_from_n_jax(nB*Yp, nB*(1-Yp), params)
    
    # First derivatives
    grad_E = grad(E_func)
    
    # Second derivatives (Hessian)
    hessian_E = jacfwd(grad_E)
    
    # Compute derivatives at the given (nB, Yp*nB)
    hess = hessian_E(jnp.array([nB, Yp*nB]))
    
    # Determinant of Hessian (stability criterion)
    criteria = hess[0, 0] * hess[1, 1] - hess[0, 1]**2
    return criteria
    
@jit
def chempo_lepton(n_lep, m_e, m_muon):
    a = jnp.sqrt(m_muon**2 - m_e**2)
    b = 3 * jnp.pi**2 * n_lep / unitMeVfm

    a2, a3, a4 = a**2, a**3, a**4
    a6, a8, a12 = a**6, a**8, a**12
    b2, b4 = b**2, b**4

    # Carefully handle sqrt to avoid evaluating sqrt of negative values
    sqrt_term = 0
    c_cubic = -11 * a12 + 14 * a6 * b2 - 2 * b4 + 2 * sqrt_term
    c = jnp.sign(c_cubic) * jnp.abs(c_cubic)**(1/3)

    d = (5 * a8 - 4 * a2 * b2 + c**2) / (3 * a2 * c )
    f = (-6 * a6 + b2) / (9 * a4)
    g = (2 * b * (9 - b2 / a6)) / 27.

    d_plus_f_sqr = 1

    sqrt_inner = 0

    k_F_muon = (
        -b / a2 
        - jnp.sign(b - 3 * a3) * 3 * d_plus_f_sqr 
        + 3 * sqrt_inner
    ) / 6.0

    k_F_muon = 0
    chempo_lep_square = b**(2/3) + m_e**2
    k_F_e = jnp.sqrt(jnp.maximum(chempo_lep_square - m_e**2, 0.0))

    return jnp.sqrt(chempo_lep_square), k_F_e, k_F_muon

@jit
def lepton_kF_array(chempo_lep, m_e, m_muon):
    k_F_e = jnp.sqrt(jnp.where(chempo_lep > m_e, chempo_lep**2 - m_e**2, 0.0))
    k_F_muon = jnp.sqrt(jnp.where(chempo_lep > m_muon, chempo_lep**2 - m_muon**2, 0.0))
    return k_F_e, k_F_muon

@jit
def lepton_eos_array(chempo_lep, m_e, m_muon):
    k_F_e, k_F_muon = lepton_kF_array(chempo_lep, m_e, m_muon)
    n_e = k_F_e**3 / (3 * jnp.pi**2)
    n_lep = n_e 

    E_F_e = jnp.sqrt(k_F_e**2 + m_e**2)

    energy_density = (
        (E_F_e * k_F_e**3 + E_F_e**3 * k_F_e - m_e**4 * jnp.log((k_F_e + E_F_e) / m_e)) 
    ) / (8 * jnp.pi**2)

    pressure = chempo_lep * n_lep - energy_density
    return jnp.array([n_lep, energy_density, pressure])*unitMeVfm

@jit
def lepton_eos_array_nlep(n_lep, m_e, m_muon):
    chempo_lep, _, _ = chempo_lepton(n_lep, m_e, m_muon)
    return lepton_eos_array(chempo_lep, m_e, m_muon)

@jit
def function_to_solve_jax(params_to_solve, params_other, saturation_params):
    t0, t1, t2, t3, x0, x3, alpha = params_to_solve
    t4, x1, x2, m_in_MeVfm2, mp, mn = params_other
    Theta_s = 3 * t1 + t2 * (4 * x2 + 5)
    Theta_v = t1 * (x1 + 2) + t2 * (x2 + 2)
    params = t0, t1, t2, t3, t4, x0, x1, x2, x3, alpha, Theta_s, Theta_v, m_in_MeVfm2, mp, mn

    ns=eos_get_ns_jax(params, init_guess=0.15)
    J, L, K = Sv_L_jax(ns, params).transpose()
    delta_mass = 8 / ((Theta_s - 2 * Theta_v) * ns * unitMeVfm ** (2/3))
    mass_tmp = 8 / (Theta_s * ns * unitMeVfm ** (2/3))
    return jnp.array([ns, mass_tmp, delta_mass, J[0],K[0],J[2],L[2]])-saturation_params

# Bridge to NumPy for SciPy root
def function_wrapper_np(params_to_solve_np, params_other_np, saturation_params_np):
    params_to_solve = jnp.array(params_to_solve_np)
    params_other = jnp.array(params_other_np)
    saturation_params = jnp.array(saturation_params_np)
    result = numpy.asarray(function_to_solve_jax(params_to_solve, params_other, saturation_params))
    return result

# JAX-based Jacobian (still NumPy-wrapped)
def jacobian_wrapper_np(params_to_solve_np, params_other_np, saturation_params_np):
    params_to_solve = jnp.array(params_to_solve_np)
    params_other = jnp.array(params_other_np)
    saturation_params = jnp.array(saturation_params_np)

    jac_fn = jacfwd(function_to_solve_jax)
    jac_val = jac_fn(params_to_solve, params_other, saturation_params)
    return numpy.asarray(jac_val)

# Main root-solving function
def get_params(saturation_params, params_other, init, tol=1e-8):
    sol = opt.root(
        fun=lambda p: function_wrapper_np(p, params_other, saturation_params),
        x0=init,
        jac=lambda p: jacobian_wrapper_np(p, params_other, saturation_params),
        method='hybr',  # or 'lm' if needed
        tol=tol
    )

    if not sol.success:
        raise RuntimeError(f"Root finding failed: {sol.message}")
    return sol.x

EDF_sly4=[[-2488,486.82,-546.3,13777,123],[0.834,-0.344,-1,1.354,1/6]]
skyrme_sly4=EDF_Skyrme(EDF_sly4)
EDF_sly5=[[-2488.345,484.230,-556.690,13757.0,125.0],[0.776,-0.317,-1.000,1.263,1/6]]
skyrme_sly5=EDF_Skyrme(EDF_sly5)
EDF_NRAPR=[[-2719.70, 417.64, -66.69, 15042.00, 115.0],[ 0.16154, -0.04799, 0.027, 0.13611, 0.14416]]
skyrme_NRAPR=EDF_Skyrme(EDF_NRAPR)
EDF_skm=[[-2645,410,-135,15595,130],[0.09,0,0,0,1/6]]
skyrme_skm=EDF_Skyrme(EDF_skm)
EDF_siii=[[-1128.75,395,-95,14000,120],[0.45,0,0,1,1]]
skyrme_siii=EDF_Skyrme(EDF_siii)
EDF_SkI3=[[-1762.88, 561.608, -227.090, 8106.2, 94.254],[0.3083, -1.1722, -1.0907, 1.2926, 0.25]]#b4=94.254 ,b4'=0
skyrme_SkI3=EDF_Skyrme(EDF_SkI3)
EDF_SkI3=[[-1762.88,561.608,-227.090,8106.2,188.508],[0.308,-1.172,-1.091,1.293,0.25]]
skyrme_SkI3=EDF_Skyrme(EDF_SkI3)
EDF_SkI4=[[-1855.83, 473.829, 1006.86, 9703.61,183.097],[0.4051,  -2.8891, -1.3252,1.1452, 0.25]]#b4=183.097,b4'=-180.351
skyrme_SkI4=EDF_Skyrme(EDF_SkI4)
EDF_UNDEF0=[[-1883.68781034,277.50021224,608.43090559,13901.94834463,125.161],[0.00974375,-1.77784395,-1.67699035,-0.38079041,0.32195599]]#b4=125.161, b4'=-91.2604000
skyrme_UNDEF0=EDF_Skyrme(EDF_UNDEF0)
EDF_UNDEF1=[-2078.32802326,239.40081204,1575.11954190,14263.64624708,38.36807206],[0.05375692,-5.07723238,-1.36650561,-0.16249117,0.27001801]#b4=38.36807206, b4'=71.31652223 
skyrme_UNDEF1=EDF_Skyrme(EDF_UNDEF1)
skyrme_list=[skyrme_sly4,skyrme_sly5,skyrme_NRAPR,skyrme_skm,skyrme_siii,skyrme_SkI3,skyrme_SkI4,skyrme_UNDEF0,skyrme_UNDEF1]
skyrme_list_name=['SLy4','SLy5','NRAPR','SKM','Siii','SkI3','SkI4','UNDEF0','UNDEF1']

params_other_list=[]
params_init_list=[]
for skyrme in skyrme_list:
    params_other_list.append(numpy.array([skyrme.t4, skyrme.x1, skyrme.x2, skyrme.m_in_MeVfm2, skyrme.mp, skyrme.mn ]))
    params_init_list.append(numpy.array([skyrme.t0,skyrme.t1,skyrme.t2,skyrme.t3,skyrme.x0,skyrme.x3,skyrme.alpha]))

# === Test ===
# EDF_sly4 = [[-2488, 486.82, -546.3, 13777, 123], [0.834, -0.344, -1, 1.354, 1/6]]
# skyrme_sly4 = EDF_Skyrme(EDF_sly4)
# skyrme_sly4.bulk_properties


#print(skyrme_sly4.eos_array_from_n(0.,skyrme_sly4.ns)[1:4]*jnp.array([1., 3., 9.])/skyrme_sly4.ns)
#skyrme_sly4.compute_bulk(skyrme_sly4.ns)
# print("ns:",skyrme_sly4.ns)
# print("properties at ns:",skyrme_sly4.compute_bulk(skyrme_sly4.ns))
# print("EOS Density:", skyrme_sly4.eosDensity_from_n(0.5, 0.5))
# print("EOS Chemical Potentials:", skyrme_sly4.eosChempo_from_n(0.5, 0.5))
# print("EOS Pressure:", skyrme_sly4.eosPressure_from_n(0.5, 0.5))
# print("EOS Array:", skyrme_sly4.eos_array_from_n(0.5, 0.5))
# print("EOS Gamma:", skyrme_sly4.eosGamma_from_n(0.5, 0.5))


J, L, K = Sv_L_jax(skyrme_sly4.ns, skyrme_sly4._params).transpose()
delta_mass = 8 / ((skyrme_sly4.Theta_s - 2 * skyrme_sly4.Theta_v) * skyrme_sly4.ns * unitMeVfm ** (2/3))
mass_tmp = 8 / (skyrme_sly4.Theta_s * skyrme_sly4.ns * unitMeVfm ** (2/3))

saturation_params=numpy.array([skyrme_sly4.ns, mass_tmp, delta_mass, J[0],K[0],J[2],L[2]])
params_other=numpy.array([skyrme_sly4.t4, skyrme_sly4.x1, skyrme_sly4.x2, skyrme_sly4.m_in_MeVfm2, skyrme_sly4.mp, skyrme_sly4.mn ])
params_init =numpy.array([-2488, 486.82, -546.3, 13777, 0.834, 1.354, 1/6])
result=get_params(saturation_params, params_other, params_init)



#function_to_solve_jax(result, params_other, saturation_params)
#EDF_sly4 = [[-2488, 486.82, -546.3, 13777, 123], [0.834, -0.344, -1, 1.354, 1/6]]
#result/numpy.array([-2488, 486.82, -546.3, 13777, 0.834, -0.344, -1, 1.354])