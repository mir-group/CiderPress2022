import numpy as np

Sqrt = np.sqrt
Pi = np.pi
Log = np.log

fac = (6*np.pi**2)**(2.0/3)/(16*np.pi)

def desc_and_ddesc(x):

    """
    gammax = 0.1838
    gamma1 = 0.0440
    gamma2 = 0.0161
    gamma0a = 0.4658
    gamma0b = 0.8318
    gamma0c = 0.3535

    gammax = 0.03396679161527282
    gamma1 = 0.025525996367805805
    gamma2 = 0.015353511288718948
    gamma0a = 0.47032113660384833
    gamma0b = 1.1014410669536636
    gamma0c = 0.37588448262415936
    center0a = 0.48470667994514244
    center0b = 0.8980790815244916
    center0c = 0.15820823165989775
    """
    gammax = 0.12011685392376696
    gamma1 = 0.025802574385367972
    gamma2 = 0.01654121930252892
    gamma0a = 0.4891146891376963
    gamma0b = 0.8342344450082123
    gamma0c = 0.41209749093646153
    center0a = 0.4944974475751677
    center0b = 0.8696877487558009
    center0c = 0.17084611732524574

    p = x[0]
    alpha = x[1]

    matrix = np.zeros((11, x.shape[-1]))
    dmatrix = np.zeros((11, 11, x.shape[-1]))

    scale = Sqrt(1 + 0.6*(-1 + alpha)*fac + fac*p)
    dscaledp = fac/(2.*Sqrt(1 + 0.6*(-1 + alpha)*fac + fac*p))
    dscaledalpha = (0.3*fac)/Sqrt(1 + 0.6*(-1 + alpha)*fac + fac*p)

    matrix[0] = (gammax*x[0])/(1 + gammax*x[0])
    matrix[1] = -1 + 2/(1 + x[1]**2)
    matrix[2] = -center0a + (gamma0a*x[2])/(1 + gamma0a*x[2])
    matrix[3] = (gamma1*x[3])/(1 + gamma1*x[3])
    matrix[4] = (gamma2*x[4])/(1 + gamma2*x[4])
    matrix[5] = (gammax*np.sqrt(gamma2/(1 + gamma2*x[4]))*x[5])/(1 + gammax*x[0])
    matrix[6] = np.sqrt(gammax/(1 + gammax*x[0]))*np.sqrt(gamma1/(1 + gamma1*x[3]))*x[6]
    matrix[7] = -center0b + (gamma0b*x[7])/(1 + gamma0b*x[7])
    matrix[8] = -center0c + (gamma0c*x[8])/(1 + gamma0c*x[8])
    matrix[9] = np.sqrt(gammax/(1 + gammax*x[0]))*x[9]*np.sqrt(gamma1/(1 + gamma1*x[3]))*np.sqrt(gamma2/(1 + gamma2*x[4]))

    dmatrix[0,0] = gammax/(1 + gammax*x[0])**2
    dmatrix[0,1] = 0
    dmatrix[0,2] = 0
    dmatrix[0,3] = 0
    dmatrix[0,4] = 0
    dmatrix[0,5] = 0
    dmatrix[0,6] = 0
    dmatrix[0,7] = 0
    dmatrix[0,8] = 0
    dmatrix[0,9] = 0
    dmatrix[1,0] = 0
    dmatrix[1,1] = (-4*x[1])/(1 + x[1]**2)**2
    dmatrix[1,2] = 0
    dmatrix[1,3] = 0
    dmatrix[1,4] = 0
    dmatrix[1,5] = 0
    dmatrix[1,6] = 0
    dmatrix[1,7] = 0
    dmatrix[1,8] = 0
    dmatrix[1,9] = 0
    dmatrix[2,0] = 0
    dmatrix[2,1] = 0
    dmatrix[2,2] = gamma0a/(1 + gamma0a*x[2])**2
    dmatrix[2,3] = 0
    dmatrix[2,4] = 0
    dmatrix[2,5] = 0
    dmatrix[2,6] = 0
    dmatrix[2,7] = 0
    dmatrix[2,8] = 0
    dmatrix[2,9] = 0
    dmatrix[3,0] = 0
    dmatrix[3,1] = 0
    dmatrix[3,2] = 0
    dmatrix[3,3] = gamma1/(1 + gamma1*x[3])**2
    dmatrix[3,4] = 0
    dmatrix[3,5] = 0
    dmatrix[3,6] = 0
    dmatrix[3,7] = 0
    dmatrix[3,8] = 0
    dmatrix[3,9] = 0
    dmatrix[4,0] = 0
    dmatrix[4,1] = 0
    dmatrix[4,2] = 0
    dmatrix[4,3] = 0
    dmatrix[4,4] = gamma2/(1 + gamma2*x[4])**2
    dmatrix[4,5] = 0
    dmatrix[4,6] = 0
    dmatrix[4,7] = 0
    dmatrix[4,8] = 0
    dmatrix[4,9] = 0
    dmatrix[5,0] = -((gammax**2*np.sqrt(gamma2/(1 + gamma2*x[4]))*x[5])/(1 + gammax*x[0])**2)
    dmatrix[5,1] = 0
    dmatrix[5,2] = 0
    dmatrix[5,3] = 0
    dmatrix[5,4] = -(gammax*(gamma2/(1 + gamma2*x[4]))**1.5*x[5])/(2.*(1 + gammax*x[0]))
    dmatrix[5,5] = (gammax*np.sqrt(gamma2/(1 + gamma2*x[4])))/(1 + gammax*x[0])
    dmatrix[5,6] = 0
    dmatrix[5,7] = 0
    dmatrix[5,8] = 0
    dmatrix[5,9] = 0
    dmatrix[6,0] = -((gammax*np.sqrt(gammax/(1 + gammax*x[0]))*np.sqrt(gamma1/(1 + gamma1*x[3]))*x[6])/(2 + 2*gammax*x[0]))
    dmatrix[6,1] = 0
    dmatrix[6,2] = 0
    dmatrix[6,3] = -((gamma1*np.sqrt(gammax/(1 + gammax*x[0]))*np.sqrt(gamma1/(1 + gamma1*x[3]))*x[6])/(2 + 2*gamma1*x[3]))
    dmatrix[6,4] = 0
    dmatrix[6,5] = 0
    dmatrix[6,6] = np.sqrt(gammax/(1 + gammax*x[0]))*np.sqrt(gamma1/(1 + gamma1*x[3]))
    dmatrix[6,7] = 0
    dmatrix[6,8] = 0
    dmatrix[6,9] = 0
    dmatrix[7,0] = 0
    dmatrix[7,1] = 0
    dmatrix[7,2] = 0
    dmatrix[7,3] = 0
    dmatrix[7,4] = 0
    dmatrix[7,5] = 0
    dmatrix[7,6] = 0
    dmatrix[7,7] = gamma0b/(1 + gamma0b*x[7])**2
    dmatrix[7,8] = 0
    dmatrix[7,9] = 0
    dmatrix[8,0] = 0
    dmatrix[8,1] = 0
    dmatrix[8,2] = 0
    dmatrix[8,3] = 0
    dmatrix[8,4] = 0
    dmatrix[8,5] = 0
    dmatrix[8,6] = 0
    dmatrix[8,7] = 0
    dmatrix[8,8] = gamma0c/(1 + gamma0c*x[8])**2
    dmatrix[8,9] = 0
    dmatrix[9,0] = -((gammax*np.sqrt(gammax/(1 + gammax*x[0]))*x[9]*np.sqrt(gamma1/(1 + gamma1*x[3]))*np.sqrt(gamma2/(1 + gamma2*x[4])))/(2 + 2*gammax*x[0]))
    dmatrix[9,1] = 0
    dmatrix[9,2] = 0
    dmatrix[9,3] = -((gamma1*np.sqrt(gammax/(1 + gammax*x[0]))*x[9]*np.sqrt(gamma1/(1 + gamma1*x[3]))*np.sqrt(gamma2/(1 + gamma2*x[4])))/(2 + 2*gamma1*x[3]))
    dmatrix[9,4] = -((gamma2*np.sqrt(gammax/(1 + gammax*x[0]))*x[9]*np.sqrt(gamma1/(1 + gamma1*x[3]))*np.sqrt(gamma2/(1 + gamma2*x[4])))/(2 + 2*gamma2*x[4]))
    dmatrix[9,5] = 0
    dmatrix[9,6] = 0
    dmatrix[9,7] = 0
    dmatrix[9,8] = 0
    dmatrix[9,9] = np.sqrt(gammax/(1 + gammax*x[0]))*np.sqrt(gamma1/(1 + gamma1*x[3]))*np.sqrt(gamma2/(1 + gamma2*x[4]))

    return matrix, dmatrix, scale, dscaledp, dscaledalpha
