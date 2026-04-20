# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt, pow, M_PI, fabs, log
from cython cimport boundscheck, wraparound, cdivision, language_level
from cython.parallel import prange
import time

from .single_stage_cy cimport fptd_single_cy, q_single_cy

cdef extern from "omp.h":
    int omp_get_num_threads()
    int omp_get_max_threads()

cpdef double get_addm_fptd_cy(double t, int d, double[:] mu_array, double[:] sacc_array, double sigma, double a, double b, double x0, int bdy, int trunc_num=100, double threshold=1e-20) nogil:
    """
    CYTHON VERSION, the code is written in complete C style to make it GIL-free, hence can be parallelized using OpenMP
    Compute the likelihood of a process hits the boundary `bdy` at time `t`
    The process is a Brownian motion with a piecewise constant drift rate.
    The drift rate is `mu_list` for each piecewise constant stage, and `t_list` are the times at which the drift rate changes.
    The boundaries are u(t) = a - b * t and l(t) = -a + b * t, `bdy=1` indicates the process hits u(t) and `bdy=-1` indicates  the process hits l(t).
    """
    cdef:
        double result = 0.0
        int i, j, n, order=30
        double temp
        double fptd
        double a_curr, T_curr

        double x_ref[30]
        double w_ref[30]
        double xs[30]
        double ws[30]
        double xs_prev[30]
        double pv[30]
        double ws_pv_product_prev[30]


    if d == 1:
        result = fptd_single_cy(t, mu_array[0], sigma, a, -b, -a, b, x0, bdy, trunc_num, threshold)
    else:
        # with np.printoptions(precision=18):
        #     display(np.polynomial.legendre.leggauss(10))
        x_ref[:] = [-0.9968934840746495, -0.9836681232797473, -0.9600218649683075, -0.9262000474292743, -0.8825605357920526, 
                    -0.8295657623827684, -0.7677774321048262, -0.6978504947933158, -0.6205261829892429, -0.5366241481420199,
                    -0.4470337695380892, -0.3527047255308781, -0.2546369261678899, -0.1538699136085835, -0.0514718425553177,
                    0.0514718425553177,  0.1538699136085835, 0.2546369261678899, 0.3527047255308781,  0.4470337695380892,
                    0.5366241481420199, 0.6205261829892429, 0.6978504947933158,  0.7677774321048262,  0.8295657623827684,
                    0.8825605357920526,  0.9262000474292743, 0.9600218649683075,  0.9836681232797473,  0.9968934840746495]
        w_ref[:] = [0.0079681924961695, 0.0184664683110911, 0.0287847078833229, 0.0387991925696268, 0.0484026728305944, 
                    0.0574931562176191, 0.0659742298821803, 0.0737559747377048, 0.0807558952294198, 0.0868997872010827, 
                    0.0921225222377858, 0.096368737174644, 0.0995934205867949, 0.1017623897484052, 0.1028526528935585, 
                    0.1028526528935585, 0.1017623897484052, 0.0995934205867949, 0.096368737174644 , 0.0921225222377858, 
                    0.0868997872010827, 0.0807558952294198, 0.0737559747377048, 0.0659742298821803, 0.0574931562176191, 
                    0.0484026728305944, 0.0387991925696268, 0.0287847078833229, 0.0184664683110911, 0.0079681924961695]
        for i in range(order):
            xs[i] = x_ref[i] * (a - b * sacc_array[1])
            ws[i] = w_ref[i] * (a - b * sacc_array[1])
            pv[i] = q_single_cy(xs[i], mu_array[0], sigma, a, -b, -a, b, sacc_array[1], x0, trunc_num, threshold)
            xs_prev[i] = xs[i]
            ws_pv_product_prev[i] = ws[i] * pv[i]
        for n in range(2, d):
            for i in range(order):
                xs[i] = x_ref[i] * (a - b * sacc_array[n])
                ws[i] = w_ref[i] * (a - b * sacc_array[n])
                pv[i] = 0
                a_curr = a - b * sacc_array[n-1]
                T_curr = sacc_array[n] - sacc_array[n-1]
                for j in range(order):
                    temp = q_single_cy(xs[i], mu_array[n-1], sigma, a_curr, -b, -a_curr, b, T_curr, xs_prev[j], trunc_num, threshold)
                    pv[i] += temp * ws_pv_product_prev[j]  
            for i in range(order):
                xs_prev[i] = xs[i]
                ws_pv_product_prev[i] = ws[i] * pv[i]
        a_curr = a - b * sacc_array[d-1]
        for i in range(order):
            fptd = fptd_single_cy(t - sacc_array[d-1], mu_array[d-1], sigma, a_curr, -b, -a_curr, b, xs[i], bdy, trunc_num, threshold)
            result += fptd * ws_pv_product_prev[i]
    return result


cpdef double get_addm_fptd_cy10(double t, int d, double[:] mu_array, double[:] sacc_array, double sigma, double a, double b, double x0, int bdy, int trunc_num=100, double threshold=1e-20) nogil:
    """
    CYTHON VERSION, the code is written in complete C style to make it GIL-free, hence can be parallelized using OpenMP
    Compute the likelihood of a process hits the boundary `bdy` at time `t`
    The process is a Brownian motion with a piecewise constant drift rate.
    The drift rate is `mu_list` for each piecewise constant stage, and `t_list` are the times at which the drift rate changes.
    The boundaries are u(t) = a - b * t and l(t) = -a + b * t, `bdy=1` indicates the process hits u(t) and `bdy=-1` indicates  the process hits l(t).
    """
    cdef:
        double result = 0.0
        int i, j, n, order=10
        double temp
        double fptd
        double a_curr, T_curr
        double x_ref[10]
        double w_ref[10]
        double xs[10]
        double ws[10]
        double xs_prev[10]
        double pv[10]
        double ws_pv_product_prev[10]

    if d == 1:
        result = fptd_single_cy(t, mu_array[0], sigma, a, -b, -a, b, x0, bdy, trunc_num, threshold)
    else:
        x_ref[:] = [-0.9739065285171717 , -0.8650633666889845 , -0.6794095682990244 ,
        -0.4333953941292472 , -0.14887433898163122,  0.14887433898163122,
         0.4333953941292472 ,  0.6794095682990244 ,  0.8650633666889845 ,
         0.9739065285171717 ]
        w_ref[:] = [0.06667134430868807, 0.14945134915058036, 0.219086362515982  ,
        0.2692667193099965 , 0.295524224714753  , 0.295524224714753  ,
        0.2692667193099965 , 0.219086362515982  , 0.14945134915058036,
        0.06667134430868807]
        for i in range(order):
            xs[i] = x_ref[i] * (a - b * sacc_array[1])
            ws[i] = w_ref[i] * (a - b * sacc_array[1])
            pv[i] = q_single_cy(xs[i], mu_array[0], sigma, a, -b, -a, b, sacc_array[1], x0, trunc_num, threshold)
            xs_prev[i] = xs[i]
            ws_pv_product_prev[i] = ws[i] * pv[i]
        for n in range(2, d):
            for i in range(order):
                xs[i] = x_ref[i] * (a - b * sacc_array[n])
                ws[i] = w_ref[i] * (a - b * sacc_array[n])
                pv[i] = 0
                a_curr = a - b * sacc_array[n-1]
                T_curr = sacc_array[n] - sacc_array[n-1]
                for j in range(order):
                    temp = q_single_cy(xs[i], mu_array[n-1], sigma, a_curr, -b, -a_curr, b, T_curr, xs_prev[j], trunc_num, threshold)
                    pv[i] += temp * ws_pv_product_prev[j]  
            for i in range(order):
                xs_prev[i] = xs[i]
                ws_pv_product_prev[i] = ws[i] * pv[i]
        a_curr = a - b * sacc_array[d-1]
        for i in range(order):
            fptd = fptd_single_cy(t - sacc_array[d-1], mu_array[d-1], sigma, a_curr, -b, -a_curr, b, xs[i], bdy, trunc_num, threshold)
            result += fptd * ws_pv_product_prev[i]
    return result


cpdef double get_addm_fptd_cy15(double t, int d, double[:] mu_array, double[:] sacc_array, double sigma, double a, double b, double x0, int bdy, int trunc_num=100, double threshold=1e-20) nogil:
    """
    CYTHON VERSION, the code is written in complete C style to make it GIL-free, hence can be parallelized using OpenMP
    Compute the likelihood of a process hits the boundary `bdy` at time `t`
    The process is a Brownian motion with a piecewise constant drift rate.
    The drift rate is `mu_list` for each piecewise constant stage, and `t_list` are the times at which the drift rate changes.
    The boundaries are u(t) = a - b * t and l(t) = -a + b * t, `bdy=1` indicates the process hits u(t) and `bdy=-1` indicates  the process hits l(t).
    """
    cdef:
        double result = 0.0
        int i, j, n, order=15
        double temp
        double fptd
        double a_curr, T_curr
        double x_ref[15]
        double w_ref[15]
        double xs[15]
        double ws[15]
        double xs_prev[15]
        double pv[15]
        double ws_pv_product_prev[15]

    if d == 1:
        result = fptd_single_cy(t, mu_array[0], sigma, a, -b, -a, b, x0, bdy, trunc_num, threshold)
    else:
        x_ref[:] = [-0.9879925180204854 , -0.937273392400706  , -0.8482065834104272 ,
        -0.7244177313601701 , -0.5709721726085388 , -0.3941513470775634 ,
        -0.20119409399743451,  0.                 ,  0.20119409399743451,
         0.3941513470775634 ,  0.5709721726085388 ,  0.7244177313601701 ,
         0.8482065834104272 ,  0.937273392400706  ,  0.9879925180204854 ]
        w_ref = [0.030753241996118647, 0.07036604748810807 , 0.10715922046717177 ,
        0.1395706779261539  , 0.16626920581699378 , 0.18616100001556188 ,
        0.19843148532711125 , 0.2025782419255609  , 0.19843148532711125 ,
        0.18616100001556188 , 0.16626920581699378 , 0.1395706779261539  ,
        0.10715922046717177 , 0.07036604748810807 , 0.030753241996118647]
        for i in range(order):
            xs[i] = x_ref[i] * (a - b * sacc_array[1])
            ws[i] = w_ref[i] * (a - b * sacc_array[1])
            pv[i] = q_single_cy(xs[i], mu_array[0], sigma, a, -b, -a, b, sacc_array[1], x0, trunc_num, threshold)
            xs_prev[i] = xs[i]
            ws_pv_product_prev[i] = ws[i] * pv[i]
        for n in range(2, d):
            for i in range(order):
                xs[i] = x_ref[i] * (a - b * sacc_array[n])
                ws[i] = w_ref[i] * (a - b * sacc_array[n])
                pv[i] = 0
                a_curr = a - b * sacc_array[n-1]
                T_curr = sacc_array[n] - sacc_array[n-1]
                for j in range(order):
                    temp = q_single_cy(xs[i], mu_array[n-1], sigma, a_curr, -b, -a_curr, b, T_curr, xs_prev[j], trunc_num, threshold)
                    pv[i] += temp * ws_pv_product_prev[j]  
            for i in range(order):
                xs_prev[i] = xs[i]
                ws_pv_product_prev[i] = ws[i] * pv[i]
        a_curr = a - b * sacc_array[d-1]
        for i in range(order):
            fptd = fptd_single_cy(t - sacc_array[d-1], mu_array[d-1], sigma, a_curr, -b, -a_curr, b, xs[i], bdy, trunc_num, threshold)
            result += fptd * ws_pv_product_prev[i]
    return result

cpdef double get_addm_fptd_cy20(double t, int d, double[:] mu_array, double[:] sacc_array, double sigma, double a, double b, double x0, int bdy, int trunc_num=100, double threshold=1e-20) nogil:
    """
    CYTHON VERSION, the code is written in complete C style to make it GIL-free, hence can be parallelized using OpenMP
    Compute the likelihood of a process hits the boundary `bdy` at time `t`
    The process is a Brownian motion with a piecewise constant drift rate.
    The drift rate is `mu_list` for each piecewise constant stage, and `t_list` are the times at which the drift rate changes.
    The boundaries are u(t) = a - b * t and l(t) = -a + b * t, `bdy=1` indicates the process hits u(t) and `bdy=-1` indicates  the process hits l(t).
    """
    cdef:
        double result = 0.0
        int i, j, n, order=20
        double temp
        double fptd
        double a_curr, T_curr
        double x_ref[20]
        double w_ref[20]
        double xs[20]
        double ws[20]
        double xs_prev[20]
        double pv[20]
        double ws_pv_product_prev[20]

    if d == 1:
        result = fptd_single_cy(t, mu_array[0], sigma, a, -b, -a, b, x0, bdy, trunc_num, threshold)
    else:
        x_ref[:] = [-0.9931285991850949, -0.9639719272779138, -0.9122344282513258, -0.8391169718222188, -0.7463319064601508,
                    -0.636053680726515, -0.5108670019508271, -0.37370608871541955, -0.2277858511416451, -0.07652652113349734,
                    0.07652652113349734, 0.2277858511416451, 0.37370608871541955, 0.5108670019508271, 0.636053680726515,
                    0.7463319064601508, 0.8391169718222188, 0.9122344282513258, 0.9639719272779138, 0.9931285991850949]
        w_ref[:] = [0.017614007139153273, 0.04060142980038622, 0.06267204833410944, 0.08327674157670467, 0.10193011981724026,
                    0.11819453196151825, 0.13168863844917653, 0.14209610931838187, 0.14917298647260366, 0.15275338713072578,
                    0.15275338713072578, 0.14917298647260366, 0.14209610931838187, 0.13168863844917653, 0.11819453196151825,
                    0.10193011981724026, 0.08327674157670467, 0.06267204833410944, 0.04060142980038622, 0.017614007139153273]
        for i in range(order):
            xs[i] = x_ref[i] * (a - b * sacc_array[1])
            ws[i] = w_ref[i] * (a - b * sacc_array[1])
            pv[i] = q_single_cy(xs[i], mu_array[0], sigma, a, -b, -a, b, sacc_array[1], x0, trunc_num, threshold)
            xs_prev[i] = xs[i]
            ws_pv_product_prev[i] = ws[i] * pv[i]
        for n in range(2, d):
            for i in range(order):
                xs[i] = x_ref[i] * (a - b * sacc_array[n])
                ws[i] = w_ref[i] * (a - b * sacc_array[n])
                pv[i] = 0
                a_curr = a - b * sacc_array[n-1]
                T_curr = sacc_array[n] - sacc_array[n-1]
                for j in range(order):
                    temp = q_single_cy(xs[i], mu_array[n-1], sigma, a_curr, -b, -a_curr, b, T_curr, xs_prev[j], trunc_num, threshold)
                    pv[i] += temp * ws_pv_product_prev[j]  
            for i in range(order):
                xs_prev[i] = xs[i]
                ws_pv_product_prev[i] = ws[i] * pv[i]
        a_curr = a - b * sacc_array[d-1]
        for i in range(order):
            fptd = fptd_single_cy(t - sacc_array[d-1], mu_array[d-1], sigma, a_curr, -b, -a_curr, b, xs[i], bdy, trunc_num, threshold)
            result += fptd * ws_pv_product_prev[i]
    return result

cpdef double get_addm_fptd_cy25(double t, int d, double[:] mu_array, double[:] sacc_array, double sigma, double a, double b, double x0, int bdy, int trunc_num=100, double threshold=1e-20) nogil:
    """
    CYTHON VERSION, the code is written in complete C style to make it GIL-free, hence can be parallelized using OpenMP
    Compute the likelihood of a process hits the boundary `bdy` at time `t`
    The process is a Brownian motion with a piecewise constant drift rate.
    The drift rate is `mu_list` for each piecewise constant stage, and `t_list` are the times at which the drift rate changes.
    The boundaries are u(t) = a - b * t and l(t) = -a + b * t, `bdy=1` indicates the process hits u(t) and `bdy=-1` indicates  the process hits l(t).
    """
    cdef:
        double result = 0.0
        int i, j, n, order=25
        double temp
        double fptd
        double a_curr, T_curr
        double x_ref[25]
        double w_ref[25]
        double xs[25]
        double ws[25]
        double xs_prev[25]
        double pv[25]
        double ws_pv_product_prev[25]

    if d == 1:
        result = fptd_single_cy(t, mu_array[0], sigma, a, -b, -a, b, x0, bdy, trunc_num, threshold)
    else:
        x_ref[:] = [-0.995556969790498  , -0.9766639214595175 , -0.9429745712289743 ,
        -0.8949919978782753 , -0.833442628760834  , -0.7592592630373576 ,
        -0.6735663684734684 , -0.577662930241223  , -0.473002731445715  ,
        -0.36117230580938786, -0.24386688372098841, -0.1228646926107104 ,
         0.                 ,  0.1228646926107104 ,  0.24386688372098841,
         0.36117230580938786,  0.473002731445715  ,  0.577662930241223  ,
         0.6735663684734684 ,  0.7592592630373576 ,  0.833442628760834  ,
         0.8949919978782753 ,  0.9429745712289743 ,  0.9766639214595175 ,
         0.995556969790498  ]
        w_ref = [0.011393798501027593, 0.026354986615031908, 0.0409391567013065  ,
        0.05490469597583544 , 0.06803833381235701 , 0.08014070033500098 ,
        0.09102826198296338 , 0.10053594906705049 , 0.10851962447426344 ,
        0.11485825914571146 , 0.1194557635357845  , 0.12224244299030987 ,
        0.12317605372671524 , 0.12224244299030987 , 0.1194557635357845  ,
        0.11485825914571146 , 0.10851962447426344 , 0.10053594906705049 ,
        0.09102826198296338 , 0.08014070033500098 , 0.06803833381235701 ,
        0.05490469597583544 , 0.0409391567013065  , 0.026354986615031908,
        0.011393798501027593]
        for i in range(order):
            xs[i] = x_ref[i] * (a - b * sacc_array[1])
            ws[i] = w_ref[i] * (a - b * sacc_array[1])
            pv[i] = q_single_cy(xs[i], mu_array[0], sigma, a, -b, -a, b, sacc_array[1], x0, trunc_num, threshold)
            xs_prev[i] = xs[i]
            ws_pv_product_prev[i] = ws[i] * pv[i]
        for n in range(2, d):
            for i in range(order):
                xs[i] = x_ref[i] * (a - b * sacc_array[n])
                ws[i] = w_ref[i] * (a - b * sacc_array[n])
                pv[i] = 0
                a_curr = a - b * sacc_array[n-1]
                T_curr = sacc_array[n] - sacc_array[n-1]
                for j in range(order):
                    temp = q_single_cy(xs[i], mu_array[n-1], sigma, a_curr, -b, -a_curr, b, T_curr, xs_prev[j], trunc_num, threshold)
                    pv[i] += temp * ws_pv_product_prev[j]  
            for i in range(order):
                xs_prev[i] = xs[i]
                ws_pv_product_prev[i] = ws[i] * pv[i]
        a_curr = a - b * sacc_array[d-1]
        for i in range(order):
            fptd = fptd_single_cy(t - sacc_array[d-1], mu_array[d-1], sigma, a_curr, -b, -a_curr, b, xs[i], bdy, trunc_num, threshold)
            result += fptd * ws_pv_product_prev[i]
    return result

cpdef double get_addm_fptd_cy35(double t, int d, double[:] mu_array, double[:] sacc_array, double sigma, double a, double b, double x0, int bdy, int trunc_num=100, double threshold=1e-20) nogil:
    """
    CYTHON VERSION, the code is written in complete C style to make it GIL-free, hence can be parallelized using OpenMP
    Compute the likelihood of a process hits the boundary `bdy` at time `t`
    The process is a Brownian motion with a piecewise constant drift rate.
    The drift rate is `mu_list` for each piecewise constant stage, and `t_list` are the times at which the drift rate changes.
    The boundaries are u(t) = a - b * t and l(t) = -a + b * t, `bdy=1` indicates the process hits u(t) and `bdy=-1` indicates  the process hits l(t).
    """
    cdef:
        double result = 0.0
        int i, j, n, order=35
        double temp
        double fptd
        double a_curr, T_curr
        double x_ref[35]
        double w_ref[35]
        double xs[35]
        double ws[35]
        double xs_prev[35]
        double pv[35]
        double ws_pv_product_prev[35]

    if d == 1:
        result = fptd_single_cy(t, mu_array[0], sigma, a, -b, -a, b, x0, bdy, trunc_num, threshold)
    else:
        x_ref[:] = [-0.9977065690996003 , -0.9879357644438516 , -0.9704376160392298 ,
        -0.9453451482078273 , -0.9128542613593176 , -0.8732191250252224 ,
        -0.8267498990922254 , -0.7738102522869126 , -0.7148145015566287 ,
        -0.6502243646658904 , -0.5805453447497645 , -0.5063227732414886 ,
        -0.42813754151781425, -0.346601554430814  , -0.2623529412092961 ,
        -0.17605106116598956, -0.08837134327565926,  0.                 ,
         0.08837134327565926,  0.17605106116598956,  0.2623529412092961 ,
         0.346601554430814  ,  0.42813754151781425,  0.5063227732414886 ,
         0.5805453447497645 ,  0.6502243646658904 ,  0.7148145015566287 ,
         0.7738102522869126 ,  0.8267498990922254 ,  0.8732191250252224 ,
         0.9128542613593176 ,  0.9453451482078273 ,  0.9704376160392298 ,
         0.9879357644438516 ,  0.9977065690996003 ]
        w_ref[:] = [0.00588343342044155 , 0.013650828348361326, 0.021322979911483672,
        0.028829260108894132, 0.03611011586346367 , 0.04310842232617002 ,
        0.04976937040135368 , 0.056040816212370004, 0.06187367196608038 ,
        0.06722228526908706 , 0.07204479477256023 , 0.07630345715544233 ,
        0.07996494224232442 , 0.08300059372885674 , 0.08538665339209921 ,
        0.08710444699718374 , 0.08814053043027563 , 0.08848679490710447 ,
        0.08814053043027563 , 0.08710444699718374 , 0.08538665339209921 ,
        0.08300059372885674 , 0.07996494224232442 , 0.07630345715544233 ,
        0.07204479477256023 , 0.06722228526908706 , 0.06187367196608038 ,
        0.056040816212370004, 0.04976937040135368 , 0.04310842232617002 ,
        0.03611011586346367 , 0.028829260108894132, 0.021322979911483672,
        0.013650828348361326, 0.00588343342044155 ]
        for i in range(order):
            xs[i] = x_ref[i] * (a - b * sacc_array[1])
            ws[i] = w_ref[i] * (a - b * sacc_array[1])
            pv[i] = q_single_cy(xs[i], mu_array[0], sigma, a, -b, -a, b, sacc_array[1], x0, trunc_num, threshold)
            xs_prev[i] = xs[i]
            ws_pv_product_prev[i] = ws[i] * pv[i]
        for n in range(2, d):
            for i in range(order):
                xs[i] = x_ref[i] * (a - b * sacc_array[n])
                ws[i] = w_ref[i] * (a - b * sacc_array[n])
                pv[i] = 0
                a_curr = a - b * sacc_array[n-1]
                T_curr = sacc_array[n] - sacc_array[n-1]
                for j in range(order):
                    temp = q_single_cy(xs[i], mu_array[n-1], sigma, a_curr, -b, -a_curr, b, T_curr, xs_prev[j], trunc_num, threshold)
                    pv[i] += temp * ws_pv_product_prev[j]  
            for i in range(order):
                xs_prev[i] = xs[i]
                ws_pv_product_prev[i] = ws[i] * pv[i]
        a_curr = a - b * sacc_array[d-1]
        for i in range(order):
            fptd = fptd_single_cy(t - sacc_array[d-1], mu_array[d-1], sigma, a_curr, -b, -a_curr, b, xs[i], bdy, trunc_num, threshold)
            result += fptd * ws_pv_product_prev[i]
    return result

cpdef double get_addm_fptd_cy40(double t, int d, double[:] mu_array, double[:] sacc_array, double sigma, double a, double b, double x0, int bdy, int trunc_num=100, double threshold=1e-20) nogil:
    """
    CYTHON VERSION, the code is written in complete C style to make it GIL-free, hence can be parallelized using OpenMP
    Compute the likelihood of a process hits the boundary `bdy` at time `t`
    The process is a Brownian motion with a piecewise constant drift rate.
    The drift rate is `mu_list` for each piecewise constant stage, and `t_list` are the times at which the drift rate changes.
    The boundaries are u(t) = a - b * t and l(t) = -a + b * t, `bdy=1` indicates the process hits u(t) and `bdy=-1` indicates  the process hits l(t).
    """
    cdef:
        double result = 0.0
        int i, j, n, order=40
        double temp
        double fptd
        double a_curr, T_curr
        double x_ref[40]
        double w_ref[40]
        double xs[40]
        double ws[40]
        double xs_prev[40]
        double pv[40]
        double ws_pv_product_prev[40]

    if d == 1:
        result = fptd_single_cy(t, mu_array[0], sigma, a, -b, -a, b, x0, bdy, trunc_num, threshold)
    else:
        x_ref[:] = [-0.9982377097105593  , -0.9907262386994571  ,
        -0.9772599499837743  , -0.9579168192137917  ,
        -0.9328128082786765  , -0.9020988069688743  ,
        -0.8659595032122596  , -0.8246122308333117  ,
        -0.7783056514265194  , -0.7273182551899271  ,
        -0.6719566846141796  , -0.6125538896679803  ,
        -0.5494671250951282  , -0.4830758016861787  ,
        -0.413779204371605   , -0.3419940908257585  ,
        -0.2681521850072537  , -0.1926975807013711  ,
        -0.11608407067525521 , -0.038772417506050816,
         0.038772417506050816,  0.11608407067525521 ,
         0.1926975807013711  ,  0.2681521850072537  ,
         0.3419940908257585  ,  0.413779204371605   ,
         0.4830758016861787  ,  0.5494671250951282  ,
         0.6125538896679803  ,  0.6719566846141796  ,
         0.7273182551899271  ,  0.7783056514265194  ,
         0.8246122308333117  ,  0.8659595032122596  ,
         0.9020988069688743  ,  0.9328128082786765  ,
         0.9579168192137917  ,  0.9772599499837743  ,
         0.9907262386994571  ,  0.9982377097105593  ]
        w_ref = [0.004521277098530018, 0.010498284531151609, 0.016421058381907345,
        0.022245849194166653, 0.027937006980023528, 0.03346019528254768 ,
        0.03878216797447238 , 0.043870908185673324, 0.048695807635072405,
        0.053227846983937115, 0.05743976909939189 , 0.06130624249292932 ,
        0.06480401345660149 , 0.0679120458152344  , 0.07061164739128717 ,
        0.07288658239580448 , 0.07472316905796868 , 0.07611036190062674 ,
        0.07703981816424839 , 0.07750594797842533 , 0.07750594797842533 ,
        0.07703981816424839 , 0.07611036190062674 , 0.07472316905796868 ,
        0.07288658239580448 , 0.07061164739128717 , 0.0679120458152344  ,
        0.06480401345660149 , 0.06130624249292932 , 0.05743976909939189 ,
        0.053227846983937115, 0.048695807635072405, 0.043870908185673324,
        0.03878216797447238 , 0.03346019528254768 , 0.027937006980023528,
        0.022245849194166653, 0.016421058381907345, 0.010498284531151609,
        0.004521277098530018]
        for i in range(order):
            xs[i] = x_ref[i] * (a - b * sacc_array[1])
            ws[i] = w_ref[i] * (a - b * sacc_array[1])
            pv[i] = q_single_cy(xs[i], mu_array[0], sigma, a, -b, -a, b, sacc_array[1], x0, trunc_num, threshold)
            xs_prev[i] = xs[i]
            ws_pv_product_prev[i] = ws[i] * pv[i]
        for n in range(2, d):
            for i in range(order):
                xs[i] = x_ref[i] * (a - b * sacc_array[n])
                ws[i] = w_ref[i] * (a - b * sacc_array[n])
                pv[i] = 0
                a_curr = a - b * sacc_array[n-1]
                T_curr = sacc_array[n] - sacc_array[n-1]
                for j in range(order):
                    temp = q_single_cy(xs[i], mu_array[n-1], sigma, a_curr, -b, -a_curr, b, T_curr, xs_prev[j], trunc_num, threshold)
                    pv[i] += temp * ws_pv_product_prev[j]  
            for i in range(order):
                xs_prev[i] = xs[i]
                ws_pv_product_prev[i] = ws[i] * pv[i]
        a_curr = a - b * sacc_array[d-1]
        for i in range(order):
            fptd = fptd_single_cy(t - sacc_array[d-1], mu_array[d-1], sigma, a_curr, -b, -a_curr, b, xs[i], bdy, trunc_num, threshold)
            result += fptd * ws_pv_product_prev[i]
    return result


cpdef double get_addm_fptd_cy100(double t, int d, double[:] mu_array, double[:] sacc_array, double sigma, double a, double b, double x0, int bdy, int trunc_num=100, double threshold=1e-20) nogil:
    """
    CYTHON VERSION, the code is written in complete C style to make it GIL-free, hence can be parallelized using OpenMP
    Compute the likelihood of a process hits the boundary `bdy` at time `t`
    The process is a Brownian motion with a piecewise constant drift rate.
    The drift rate is `mu_list` for each piecewise constant stage, and `t_list` are the times at which the drift rate changes.
    The boundaries are u(t) = a - b * t and l(t) = -a + b * t, `bdy=1` indicates the process hits u(t) and `bdy=-1` indicates  the process hits l(t).
    """
    cdef:
        double result = 0.0
        int i, j, n, order=100
        double temp
        double fptd
        double a_curr, T_curr
        double x_ref[100]
        double w_ref[100]
        double xs[100]
        double ws[100]
        double xs_prev[100]
        double pv[100]
        double ws_pv_product_prev[100]

    if d == 1:
        result = fptd_single_cy(t, mu_array[0], sigma, a, -b, -a, b, x0, bdy, trunc_num, threshold)
    else:
        x_ref[:] = [-0.9997137267734413  , -0.9984919506395958  ,
        -0.9962951347331251  , -0.9931249370374434  ,
        -0.9889843952429918  , -0.983877540706057   ,
        -0.9778093584869183  , -0.9707857757637064  ,
        -0.9628136542558156  , -0.9539007829254917  ,
        -0.944055870136256   , -0.9332885350430795  ,
        -0.921609298145334   , -0.9090295709825297  ,
        -0.895561644970727   , -0.8812186793850184  ,
        -0.8660146884971647  , -0.8499645278795913  ,
        -0.8330838798884008  , -0.8153892383391762  ,
        -0.7968978923903145  , -0.7776279096494955  ,
        -0.7575981185197072  , -0.7368280898020207  ,
        -0.7153381175730564  , -0.693149199355802   ,
        -0.670283015603141   , -0.6467619085141293  ,
        -0.6226088602037078  , -0.5978474702471788  ,
        -0.5725019326213812  , -0.5465970120650941  ,
        -0.520158019881763   , -0.49321078920819095 ,
        -0.465781649773358   , -0.4378974021720315  ,
        -0.40958529167830154 , -0.38087298162462996 ,
        -0.3517885263724217  , -0.32236034390052914 ,
        -0.292617188038472   , -0.2625881203715035  ,
        -0.23230248184497396 , -0.20178986409573602 ,
        -0.17108008053860327 , -0.14020313723611397 ,
        -0.10918920358006111 , -0.07806858281343663 ,
        -0.046871682421591634, -0.015628984421543084,
         0.015628984421543084,  0.046871682421591634,
         0.07806858281343663 ,  0.10918920358006111 ,
         0.14020313723611397 ,  0.17108008053860327 ,
         0.20178986409573602 ,  0.23230248184497396 ,
         0.2625881203715035  ,  0.292617188038472   ,
         0.32236034390052914 ,  0.3517885263724217  ,
         0.38087298162462996 ,  0.40958529167830154 ,
         0.4378974021720315  ,  0.465781649773358   ,
         0.49321078920819095 ,  0.520158019881763   ,
         0.5465970120650941  ,  0.5725019326213812  ,
         0.5978474702471788  ,  0.6226088602037078  ,
         0.6467619085141293  ,  0.670283015603141   ,
         0.693149199355802   ,  0.7153381175730564  ,
         0.7368280898020207  ,  0.7575981185197072  ,
         0.7776279096494955  ,  0.7968978923903145  ,
         0.8153892383391762  ,  0.8330838798884008  ,
         0.8499645278795913  ,  0.8660146884971647  ,
         0.8812186793850184  ,  0.895561644970727   ,
         0.9090295709825297  ,  0.921609298145334   ,
         0.9332885350430795  ,  0.944055870136256   ,
         0.9539007829254917  ,  0.9628136542558156  ,
         0.9707857757637064  ,  0.9778093584869183  ,
         0.983877540706057   ,  0.9889843952429918  ,
         0.9931249370374434  ,  0.9962951347331251  ,
         0.9984919506395958  ,  0.9997137267734413  ]
        w_ref[:] = [0.000734634490500881, 0.001709392653517807, 0.002683925371554019,
        0.003655961201327216, 0.004624450063421818, 0.005588428003865117,
        0.00654694845084515 , 0.007499073255464816, 0.008443871469668721,
        0.009380419653694542, 0.01030780257486916 , 0.01122511402318622 ,
        0.012131457662979251, 0.013025947892971715, 0.01390771070371885 ,
        0.014775884527441474, 0.015629621077546098, 0.01646808617614516 ,
        0.017290460568323632, 0.018095940722128407, 0.018883739613374886,
        0.01965308749443545 , 0.020403232646209593, 0.021133442112527594,
        0.02184300241624754 , 0.02253122025633626 , 0.02319742318525442 ,
        0.023840960265968263, 0.024461202707957153, 0.025057544481579718,
        0.025629402910208283, 0.02617621923954582 , 0.02669745918357113 ,
        0.02719261344657694 , 0.027661198220792507, 0.028102755659101357,
        0.028516854322395237, 0.028903089601125278, 0.029261084110638446,
        0.029590488059912694, 0.02989097959333295 , 0.03016226510516929 ,
        0.030404079526454932, 0.030616186583980524, 0.03079837903115269 ,
        0.030950478850491105, 0.03107233742756666 , 0.031163835696210035,
        0.03122488425484948 , 0.03125542345386349 , 0.03125542345386349 ,
        0.03122488425484948 , 0.031163835696210035, 0.03107233742756666 ,
        0.030950478850491105, 0.03079837903115269 , 0.030616186583980524,
        0.030404079526454932, 0.03016226510516929 , 0.02989097959333295 ,
        0.029590488059912694, 0.029261084110638446, 0.028903089601125278,
        0.028516854322395237, 0.028102755659101357, 0.027661198220792507,
        0.02719261344657694 , 0.02669745918357113 , 0.02617621923954582 ,
        0.025629402910208283, 0.025057544481579718, 0.024461202707957153,
        0.023840960265968263, 0.02319742318525442 , 0.02253122025633626 ,
        0.02184300241624754 , 0.021133442112527594, 0.020403232646209593,
        0.01965308749443545 , 0.018883739613374886, 0.018095940722128407,
        0.017290460568323632, 0.01646808617614516 , 0.015629621077546098,
        0.014775884527441474, 0.01390771070371885 , 0.013025947892971715,
        0.012131457662979251, 0.01122511402318622 , 0.01030780257486916 ,
        0.009380419653694542, 0.008443871469668721, 0.007499073255464816,
        0.00654694845084515 , 0.005588428003865117, 0.004624450063421818,
        0.003655961201327216, 0.002683925371554019, 0.001709392653517807,
        0.000734634490500881]
        for i in range(order):
            xs[i] = x_ref[i] * (a - b * sacc_array[1])
            ws[i] = w_ref[i] * (a - b * sacc_array[1])
            pv[i] = q_single_cy(xs[i], mu_array[0], sigma, a, -b, -a, b, sacc_array[1], x0, trunc_num, threshold)
            xs_prev[i] = xs[i]
            ws_pv_product_prev[i] = ws[i] * pv[i]
        for n in range(2, d):
            for i in range(order):
                xs[i] = x_ref[i] * (a - b * sacc_array[n])
                ws[i] = w_ref[i] * (a - b * sacc_array[n])
                pv[i] = 0
                a_curr = a - b * sacc_array[n-1]
                T_curr = sacc_array[n] - sacc_array[n-1]
                for j in range(order):
                    temp = q_single_cy(xs[i], mu_array[n-1], sigma, a_curr, -b, -a_curr, b, T_curr, xs_prev[j], trunc_num, threshold)
                    pv[i] += temp * ws_pv_product_prev[j]  
            for i in range(order):
                xs_prev[i] = xs[i]
                ws_pv_product_prev[i] = ws[i] * pv[i]
        a_curr = a - b * sacc_array[d-1]
        for i in range(order):
            fptd = fptd_single_cy(t - sacc_array[d-1], mu_array[d-1], sigma, a_curr, -b, -a_curr, b, xs[i], bdy, trunc_num, threshold)
            result += fptd * ws_pv_product_prev[i]
    return result


cpdef double get_addm_fptd_cy_REF(double t, int d, double[:] mu_array, double[:] sacc_array, double sigma, double a, double b, double x0, int bdy, int trunc_num=100, double threshold=1e-20):
    """
    CYTHON VERSION, the code is written in complete C style to make it GIL-free, hence can be parallelized using OpenMP
    Compute the likelihood of a process hits the boundary `bdy` at time `t`
    The process is a Brownian motion with a piecewise constant drift rate.
    The drift rate is `mu_list` for each piecewise constant stage, and `t_list` are the times at which the drift rate changes.
    The boundaries are u(t) = a - b * t and l(t) = -a + b * t, `bdy=1` indicates the process hits u(t) and `bdy=-1` indicates  the process hits l(t).
    """
    cdef:
        double result = 0.0
        int i, j, n, order=200
        double temp
        double fptd
        double a_curr, T_curr
        double xs[200]
        double ws[200]
        double xs_prev[200]
        double pv[200]
        double ws_pv_product_prev[200]
    
    threshold = 1e-100
    if d == 1:
        result = fptd_single_cy(t, mu_array[0], sigma, a, -b, -a, b, x0, bdy, trunc_num, threshold)
    else:
        x_ref, w_ref = np.polynomial.legendre.leggauss(200)
        for i in range(order):
            xs[i] = x_ref[i] * (a - b * sacc_array[1])
            ws[i] = w_ref[i] * (a - b * sacc_array[1])
            pv[i] = q_single_cy(xs[i], mu_array[0], sigma, a, -b, -a, b, sacc_array[1], x0, trunc_num, threshold)
            xs_prev[i] = xs[i]
            ws_pv_product_prev[i] = ws[i] * pv[i]
        for n in range(2, d):
            for i in range(order):
                xs[i] = x_ref[i] * (a - b * sacc_array[n])
                ws[i] = w_ref[i] * (a - b * sacc_array[n])
                pv[i] = 0
                a_curr = a - b * sacc_array[n-1]
                T_curr = sacc_array[n] - sacc_array[n-1]
                for j in range(order):
                    temp = q_single_cy(xs[i], mu_array[n-1], sigma, a_curr, -b, -a_curr, b, T_curr, xs_prev[j], trunc_num, threshold)
                    pv[i] += temp * ws_pv_product_prev[j]  
            for i in range(order):
                xs_prev[i] = xs[i]
                ws_pv_product_prev[i] = ws[i] * pv[i]
        a_curr = a - b * sacc_array[d-1]
        for i in range(order):
            fptd = fptd_single_cy(t - sacc_array[d-1], mu_array[d-1], sigma, a_curr, -b, -a_curr, b, xs[i], bdy, trunc_num, threshold)
            result += fptd * ws_pv_product_prev[i]
    return result



@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef run_timings():
    cdef:
        double sigma = 1.0
        double a = 1.5
        double b = 0.3
        double x0 = -0.5
        int d
        np.ndarray[np.double_t, ndim=1] mu_array
        np.ndarray[np.double_t, ndim=1] fixation_array
        np.ndarray[np.double_t, ndim=1] sacc_array
        np.ndarray[np.double_t, ndim=1] rt_array

    mu_array = np.array([1., -0.2, 1.5, 0.5, -1., 1., -0.2, 1.5, 0.5, -1.], dtype=np.float64)
    fixation_array = np.array([0.5, 0.75, 0.5, 0.25, 0.5, 0.5, 0.75, 0.5, 0.25, 0.5], dtype=np.float64)
    sacc_array = np.cumsum(fixation_array).astype(np.float64)
    sacc_array = np.concatenate(([0.0], sacc_array)).astype(np.float64)
    rt_array = ((sacc_array[1:] + sacc_array[:-1]) / 2).astype(np.float64)
    sacc_array = sacc_array[:-1]
    d = mu_array.shape[0]

    # Run tests
    for n in range(d):
        time_get_addm_fptd(rt_array[n], n + 1, mu_array, sacc_array, sigma, a, b, x0, 1)


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef time_get_addm_fptd(double t, int d, np.ndarray[double, ndim=1] mu_array, np.ndarray[double, ndim=1] sacc_array, double sigma, double a, double b, double x0, int bdy):
    cdef:
        int n_runs = 100
        int run
        double start, end, duration

    start = time.perf_counter()
    for run in range(n_runs):
        get_addm_fptd_cy(t, d, mu_array, sacc_array, sigma, a, b, x0, bdy)
    end = time.perf_counter()

    duration = (end - start) * 1e6
    print("%d stages, average time per run: %.2f microseconds" % (d, duration / n_runs))
    


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef get_mu_array_padded(double mu1, double mu2, int max_d, int d, int flag):
    """
    Generate an array of length `max_d`.
    The first `d` elements are alternating drift rates `mu1` and `mu2`, where `flag` determines the starting drift rate.
    `flag`=0: mu1 -> mu2 -> mu1 -> ...
    `flag`=1: mu2 -> mu1 -> mu2 -> ...
    The remaining elements are set to default(0). 
    Note that the effective length of mu_array is `d` and the effective length of sacc_array is `d-1`.
    """
    cdef np.ndarray[double, ndim=1] mu_array = np.zeros(max_d, dtype=np.float64)
    cdef double current_mu = mu2 if flag else mu1
    for i in range(d):
        mu_array[i] = current_mu
        current_mu = mu2 if current_mu == mu1 else mu1
    return mu_array

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef get_mu_array_data_padded(np.ndarray[double, ndim=1] mu1_data, \
                               np.ndarray[double, ndim=1] mu2_data, \
                               int max_d, \
                               np.ndarray[int, ndim=1] flag_data, \
                               np.ndarray[int, ndim=1] length_data):
    """
    Generate a dataset of drift rates
    each drift rate is from `get_mu_array_padded(mu1, mu2, max_d, length_data[i], flag_data[i])`
    """
    cdef int num_data = len(flag_data)
    cdef np.ndarray[double, ndim=2] mu_array_data = np.zeros((num_data, max_d), dtype=np.float64)
    for n in range(num_data):
        mu1 = mu1_data[n]
        mu2 = mu2_data[n]
        mu_array_data[n] = get_mu_array_padded(mu1, mu2, max_d, length_data[n], flag_data[n])
    return mu_array_data


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef double compute_loss_serial(np.ndarray[double, ndim=1] mu1_data, \
                               np.ndarray[double, ndim=1] mu2_data, \
                               np.ndarray[double, ndim=1] rt_data, \
                               np.ndarray[int, ndim=1] choice_data, \
                               np.ndarray[int, ndim=1] flag_data, \
                               np.ndarray[double, ndim=2] sacc_data, \
                               np.ndarray[int, ndim=1] length_data, \
                               int max_d, double sigma, double a, double b, double x0, double threshold=1e-20):
    cdef:
        int n
        double total_loss = 0.0, likelihood, loss
        int num_data = len(rt_data), num_data_effective = len(rt_data)
        np.ndarray[double, ndim=2] mu_array_data=get_mu_array_data_padded(mu1_data, mu2_data, max_d, flag_data, length_data)
    for n in range(num_data):
        likelihood = get_addm_fptd_cy(rt_data[n], length_data[n], mu_array_data[n], sacc_data[n], sigma, a, b, x0, choice_data[n], 100, threshold)
        if likelihood > 0:
            loss = -log(likelihood)
        else:
            print(f"Warning: trial {n} has 0 likelihood, skipping.")
            loss = 0
            num_data_effective -= 1
        # if n % 1 == 0:
        #     print(f"n={n}, -loglikelihood={loss:.5f}")
        total_loss += loss
    return total_loss / num_data_effective


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef np.ndarray[double, ndim=1] compute_llhds_serial10(np.ndarray[double, ndim=1] mu1_data, \
                               np.ndarray[double, ndim=1] mu2_data, \
                               np.ndarray[double, ndim=1] rt_data, \
                               np.ndarray[int, ndim=1] choice_data, \
                               np.ndarray[int, ndim=1] flag_data, \
                               np.ndarray[double, ndim=2] sacc_data, \
                               np.ndarray[int, ndim=1] length_data, \
                               int max_d, double sigma, double a, double b, double x0, double threshold=1e-20):
    """
    Returns a 1D array of per-trial likelihoods (zeros included for non-positive values).
    """
    cdef:
        int n
        int num_data = len(rt_data)
        double likelihood
        np.ndarray[double, ndim=1] likelihoods = np.empty(num_data, dtype=np.float64)
        np.ndarray[double, ndim=2] mu_array_data = get_mu_array_data_padded(
            mu1_data, mu2_data, max_d, flag_data, length_data
        )

    for n in range(num_data):
        likelihood = get_addm_fptd_cy10(
            rt_data[n], length_data[n], mu_array_data[n], sacc_data[n],
            sigma, a, b, x0, choice_data[n], 100, threshold
        )
        # Keep zeros for non-positive values
        if likelihood > 0.0:
            likelihoods[n] = likelihood
        else:
            print(f"Warning: trial {n} has 0 likelihood.")
            likelihoods[n] = 0.0
    return likelihoods

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef np.ndarray[double, ndim=1] compute_llhds_serial15(np.ndarray[double, ndim=1] mu1_data, \
                               np.ndarray[double, ndim=1] mu2_data, \
                               np.ndarray[double, ndim=1] rt_data, \
                               np.ndarray[int, ndim=1] choice_data, \
                               np.ndarray[int, ndim=1] flag_data, \
                               np.ndarray[double, ndim=2] sacc_data, \
                               np.ndarray[int, ndim=1] length_data, \
                               int max_d, double sigma, double a, double b, double x0, double threshold=1e-20):
    """
    Returns a 1D array of per-trial likelihoods (zeros included for non-positive values).
    """
    cdef:
        int n
        int num_data = len(rt_data)
        double likelihood
        np.ndarray[double, ndim=1] likelihoods = np.empty(num_data, dtype=np.float64)
        np.ndarray[double, ndim=2] mu_array_data = get_mu_array_data_padded(
            mu1_data, mu2_data, max_d, flag_data, length_data
        )

    for n in range(num_data):
        likelihood = get_addm_fptd_cy15(
            rt_data[n], length_data[n], mu_array_data[n], sacc_data[n],
            sigma, a, b, x0, choice_data[n], 100, threshold
        )
        # Keep zeros for non-positive values
        if likelihood > 0.0:
            likelihoods[n] = likelihood
        else:
            print(f"Warning: trial {n} has 0 likelihood.")
            likelihoods[n] = 0.0
    return likelihoods

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef np.ndarray[double, ndim=1] compute_llhds_serial20(np.ndarray[double, ndim=1] mu1_data, \
                               np.ndarray[double, ndim=1] mu2_data, \
                               np.ndarray[double, ndim=1] rt_data, \
                               np.ndarray[int, ndim=1] choice_data, \
                               np.ndarray[int, ndim=1] flag_data, \
                               np.ndarray[double, ndim=2] sacc_data, \
                               np.ndarray[int, ndim=1] length_data, \
                               int max_d, double sigma, double a, double b, double x0, double threshold=1e-20):
    """
    Returns a 1D array of per-trial likelihoods (zeros included for non-positive values).
    """
    cdef:
        int n
        int num_data = len(rt_data)
        double likelihood
        np.ndarray[double, ndim=1] likelihoods = np.empty(num_data, dtype=np.float64)
        np.ndarray[double, ndim=2] mu_array_data = get_mu_array_data_padded(
            mu1_data, mu2_data, max_d, flag_data, length_data
        )

    for n in range(num_data):
        likelihood = get_addm_fptd_cy20(
            rt_data[n], length_data[n], mu_array_data[n], sacc_data[n],
            sigma, a, b, x0, choice_data[n], 100, threshold
        )
        # Keep zeros for non-positive values
        if likelihood > 0.0:
            likelihoods[n] = likelihood
        else:
            print(f"Warning: trial {n} has 0 likelihood.")
            likelihoods[n] = 0.0
    return likelihoods

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef np.ndarray[double, ndim=1] compute_llhds_serial25(np.ndarray[double, ndim=1] mu1_data, \
                               np.ndarray[double, ndim=1] mu2_data, \
                               np.ndarray[double, ndim=1] rt_data, \
                               np.ndarray[int, ndim=1] choice_data, \
                               np.ndarray[int, ndim=1] flag_data, \
                               np.ndarray[double, ndim=2] sacc_data, \
                               np.ndarray[int, ndim=1] length_data, \
                               int max_d, double sigma, double a, double b, double x0, double threshold=1e-20):
    """
    Returns a 1D array of per-trial likelihoods (zeros included for non-positive values).
    """
    cdef:
        int n
        int num_data = len(rt_data)
        double likelihood
        np.ndarray[double, ndim=1] likelihoods = np.empty(num_data, dtype=np.float64)
        np.ndarray[double, ndim=2] mu_array_data = get_mu_array_data_padded(
            mu1_data, mu2_data, max_d, flag_data, length_data
        )

    for n in range(num_data):
        likelihood = get_addm_fptd_cy25(
            rt_data[n], length_data[n], mu_array_data[n], sacc_data[n],
            sigma, a, b, x0, choice_data[n], 100, threshold
        )
        # Keep zeros for non-positive values
        if likelihood > 0.0:
            likelihoods[n] = likelihood
        else:
            print(f"Warning: trial {n} has 0 likelihood.")
            likelihoods[n] = 0.0
    return likelihoods

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef np.ndarray[double, ndim=1] compute_llhds_serial30(np.ndarray[double, ndim=1] mu1_data, \
                               np.ndarray[double, ndim=1] mu2_data, \
                               np.ndarray[double, ndim=1] rt_data, \
                               np.ndarray[int, ndim=1] choice_data, \
                               np.ndarray[int, ndim=1] flag_data, \
                               np.ndarray[double, ndim=2] sacc_data, \
                               np.ndarray[int, ndim=1] length_data, \
                               int max_d, double sigma, double a, double b, double x0, double threshold=1e-20):
    """
    Returns a 1D array of per-trial likelihoods (zeros included for non-positive values).
    """
    cdef:
        int n
        int num_data = len(rt_data)
        double likelihood
        np.ndarray[double, ndim=1] likelihoods = np.empty(num_data, dtype=np.float64)
        np.ndarray[double, ndim=2] mu_array_data = get_mu_array_data_padded(
            mu1_data, mu2_data, max_d, flag_data, length_data
        )

    for n in range(num_data):
        likelihood = get_addm_fptd_cy(
            rt_data[n], length_data[n], mu_array_data[n], sacc_data[n],
            sigma, a, b, x0, choice_data[n], 100, threshold
        )
        # Keep zeros for non-positive values
        if likelihood > 0.0:
            likelihoods[n] = likelihood
        else:
            print(f"Warning: trial {n} has 0 likelihood.")
            likelihoods[n] = 0.0
    return likelihoods

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef np.ndarray[double, ndim=1] compute_llhds_serial35(np.ndarray[double, ndim=1] mu1_data, \
                               np.ndarray[double, ndim=1] mu2_data, \
                               np.ndarray[double, ndim=1] rt_data, \
                               np.ndarray[int, ndim=1] choice_data, \
                               np.ndarray[int, ndim=1] flag_data, \
                               np.ndarray[double, ndim=2] sacc_data, \
                               np.ndarray[int, ndim=1] length_data, \
                               int max_d, double sigma, double a, double b, double x0, double threshold=1e-20):
    """
    Returns a 1D array of per-trial likelihoods (zeros included for non-positive values).
    """
    cdef:
        int n
        int num_data = len(rt_data)
        double likelihood
        np.ndarray[double, ndim=1] likelihoods = np.empty(num_data, dtype=np.float64)
        np.ndarray[double, ndim=2] mu_array_data = get_mu_array_data_padded(
            mu1_data, mu2_data, max_d, flag_data, length_data
        )

    for n in range(num_data):
        likelihood = get_addm_fptd_cy35(
            rt_data[n], length_data[n], mu_array_data[n], sacc_data[n],
            sigma, a, b, x0, choice_data[n], 100, threshold
        )
        # Keep zeros for non-positive values
        if likelihood > 0.0:
            likelihoods[n] = likelihood
        else:
            print(f"Warning: trial {n} has 0 likelihood.")
            likelihoods[n] = 0.0
    return likelihoods

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef np.ndarray[double, ndim=1] compute_llhds_serial40(np.ndarray[double, ndim=1] mu1_data, \
                               np.ndarray[double, ndim=1] mu2_data, \
                               np.ndarray[double, ndim=1] rt_data, \
                               np.ndarray[int, ndim=1] choice_data, \
                               np.ndarray[int, ndim=1] flag_data, \
                               np.ndarray[double, ndim=2] sacc_data, \
                               np.ndarray[int, ndim=1] length_data, \
                               int max_d, double sigma, double a, double b, double x0, double threshold=1e-20):
    """
    Returns a 1D array of per-trial likelihoods (zeros included for non-positive values).
    """
    cdef:
        int n
        int num_data = len(rt_data)
        double likelihood
        np.ndarray[double, ndim=1] likelihoods = np.empty(num_data, dtype=np.float64)
        np.ndarray[double, ndim=2] mu_array_data = get_mu_array_data_padded(
            mu1_data, mu2_data, max_d, flag_data, length_data
        )

    for n in range(num_data):
        likelihood = get_addm_fptd_cy40(
            rt_data[n], length_data[n], mu_array_data[n], sacc_data[n],
            sigma, a, b, x0, choice_data[n], 100, threshold
        )
        # Keep zeros for non-positive values
        if likelihood > 0.0:
            likelihoods[n] = likelihood
        else:
            print(f"Warning: trial {n} has 0 likelihood.")
            likelihoods[n] = 0.0
    return likelihoods

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef np.ndarray[double, ndim=1] compute_llhds_serial100(np.ndarray[double, ndim=1] mu1_data, \
                               np.ndarray[double, ndim=1] mu2_data, \
                               np.ndarray[double, ndim=1] rt_data, \
                               np.ndarray[int, ndim=1] choice_data, \
                               np.ndarray[int, ndim=1] flag_data, \
                               np.ndarray[double, ndim=2] sacc_data, \
                               np.ndarray[int, ndim=1] length_data, \
                               int max_d, double sigma, double a, double b, double x0, double threshold=1e-20):
    """
    Returns a 1D array of per-trial likelihoods (zeros included for non-positive values).
    """
    cdef:
        int n
        int num_data = len(rt_data)
        double likelihood
        np.ndarray[double, ndim=1] likelihoods = np.empty(num_data, dtype=np.float64)
        np.ndarray[double, ndim=2] mu_array_data = get_mu_array_data_padded(
            mu1_data, mu2_data, max_d, flag_data, length_data
        )

    for n in range(num_data):
        likelihood = get_addm_fptd_cy100(
            rt_data[n], length_data[n], mu_array_data[n], sacc_data[n],
            sigma, a, b, x0, choice_data[n], 100, threshold
        )
        # Keep zeros for non-positive values
        if likelihood > 0.0:
            likelihoods[n] = likelihood
        else:
            print(f"Warning: trial {n} has 0 likelihood.")
            likelihoods[n] = 0.0
    return likelihoods


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef np.ndarray[double, ndim=1] compute_llhds_serial_REF(np.ndarray[double, ndim=1] mu1_data, \
                               np.ndarray[double, ndim=1] mu2_data, \
                               np.ndarray[double, ndim=1] rt_data, \
                               np.ndarray[int, ndim=1] choice_data, \
                               np.ndarray[int, ndim=1] flag_data, \
                               np.ndarray[double, ndim=2] sacc_data, \
                               np.ndarray[int, ndim=1] length_data, \
                               int max_d, double sigma, double a, double b, double x0, double threshold=1e-20):
    """
    Returns a 1D array of per-trial likelihoods (zeros included for non-positive values).
    """
    cdef:
        int n
        int num_data = len(rt_data)
        double likelihood
        np.ndarray[double, ndim=1] likelihoods = np.empty(num_data, dtype=np.float64)
        np.ndarray[double, ndim=2] mu_array_data = get_mu_array_data_padded(
            mu1_data, mu2_data, max_d, flag_data, length_data
        )

    for n in range(num_data):
        likelihood = get_addm_fptd_cy_REF(
            rt_data[n], length_data[n], mu_array_data[n], sacc_data[n],
            sigma, a, b, x0, choice_data[n], 100, threshold
        )
        # Keep zeros for non-positive values
        if likelihood > 0.0:
            likelihoods[n] = likelihood
        else:
            print(f"Warning: trial {n} has 0 likelihood.")
            likelihoods[n] = 0.0
    return likelihoods



@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef double compute_loss_parallel(np.ndarray[double, ndim=1] mu1_data, \
                               np.ndarray[double, ndim=1] mu2_data, \
                               np.ndarray[double, ndim=1] rt_data, \
                               np.ndarray[int, ndim=1] choice_data, \
                               np.ndarray[int, ndim=1] flag_data, \
                               np.ndarray[double, ndim=2] sacc_data, \
                               np.ndarray[int, ndim=1] length_data, \
                               int max_d, double sigma, double a, double b, double x0, double threshold=1e-20, int num_threads=-1):
    cdef:
        int n
        double total_loss = 0.0, likelihood, loss
        int num_data = len(rt_data), num_data_effective = len(rt_data)
        np.ndarray[double, ndim=2] mu_data=get_mu_array_data_padded(mu1_data, mu2_data, max_d, flag_data, length_data)
        double[:, :] mu_data_view = mu_data
        double[:, :] sacc_data_view = sacc_data
    if num_threads <= 0:
        num_threads = omp_get_max_threads()
    for n in prange(num_data, nogil=True, schedule='dynamic', num_threads=num_threads):
        likelihood = get_addm_fptd_cy(rt_data[n], length_data[n], mu_data_view[n], sacc_data_view[n], sigma, a, b, x0, choice_data[n], 100, threshold)
        if likelihood > 0:
            loss = -log(likelihood)
        else:
            loss = 0
            num_data_effective -= 1
        with gil:
            total_loss += loss
    # print("num_data_effective:", num_data_effective)
    return total_loss / num_data_effective

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef double compute_glamloss_parallel(np.ndarray[double, ndim=1] mu1_data, \
                                       np.ndarray[double, ndim=1] mu2_data, \
                                       np.ndarray[double, ndim=1] rt_data, \
                                       np.ndarray[int, ndim=1] choice_data, \
                                       np.ndarray[int, ndim=1] flag_data, \
                                       np.ndarray[double, ndim=2] sacc_data, \
                                       np.ndarray[int, ndim=1] length_data, \
                                       int max_d, double sigma, double a, double b, double x0, double threshold=1e-20):
    cdef:
        int n, i, L
        double total_loss = 0.0, likelihood, loss, mu_sum
        int num_data = len(rt_data), num_data_effective = len(rt_data)
        np.ndarray[double, ndim=2] mu_data=get_mu_array_data_padded(mu1_data, mu2_data, max_d, flag_data, length_data)
        double[:, :] mu_data_view = mu_data
        double[:, :] sacc_data_view = sacc_data
        double[:] mu_glam_data = np.zeros(num_data)
    for n in range(num_data):
        L = length_data[n] # effective length for this trial
        mu_sum = 0.0
        for i in range(L - 1):
            mu_sum += mu_data_view[n, i] * (sacc_data_view[n, i + 1] - sacc_data_view[n, i])
        mu_sum += mu_data_view[n, L - 1] * (rt_data[n] - sacc_data_view[n, L - 1])
        mu_glam_data[n] = mu_sum / rt_data[n]
    max_num_threads = omp_get_max_threads()
    for n in prange(num_data, nogil=True, schedule='dynamic', num_threads=max_num_threads):
        likelihood = fptd_single_cy(rt_data[n], mu_glam_data[n], sigma, a, -b, -a, b, x0, choice_data[n], 100, threshold)
        if likelihood > 0:
            loss = -log(likelihood)
        else:
            loss = 0
            num_data_effective -= 1
        with gil:
            total_loss += loss
    return total_loss / num_data_effective

cpdef print_num_threads():
    print("Number of available threads:", omp_get_max_threads())


