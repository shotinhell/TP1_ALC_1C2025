# Matriz A de ejemplo
#A_ejemplo = np.array([
#    [0, 1, 1, 1, 0, 0, 0, 0],
#    [1, 0, 1, 1, 0, 0, 0, 0],
#    [1, 1, 0, 1, 0, 1, 0, 0],
#    [1, 1, 1, 0, 1, 0, 0, 0],
#    [0, 0, 0, 1, 0, 1, 1, 1],
#    [0, 0, 1, 0, 1, 0, 1, 1],
#    [0, 0, 0, 0, 1, 1, 0, 1],
#    [0, 0, 0, 0, 1, 1, 1, 0]
#])
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Para leer archivos
import geopandas as gpd # Para hacer cosas geográficas
import seaborn as sns # Para hacer plots lindos
import networkx as nx # Construcción de la red en NetworkX
import scipy
import template_funciones as func


def calcula_L(A):
    # La función recibe la matriz de adyacencia A y calcula la matriz laplaciana
    K = np.diag(np.sum(A, axis=1))
    L = K - A
    return L

def calcula_R(A):
    # La funcion recibe la matriz de adyacencia A y calcula la matriz de modularidad
    DosE = np.sum(A)  # Calculo 2*E
    K = np.sum(A, axis=1)
    P = np.outer(K, K) / DosE
    R = A - P
    return R

def calcula_lambda(L,v):
    s = np.sign(v)
    lambdon = (1/4) * np.dot(s.T, np.dot(L, s))  # Cálculo del corte mínimo
    return lambdon

def calcula_Q(R,v):
    # La funcion recibe R y s y retorna la modularidad (a menos de un factor 2E)
    s = np.sign(v)
    DosE = np.sum(R)
    Q = (1 / (2 * DosE)) * np.dot(s.T, np.dot(R, s))
    return Q

def metpot1(A,tol=1e-8,maxrep=np.inf):
   # Recibe una matriz A y calcula su autovalor de mayor módulo, con un error relativo menor a tol y-o haciendo como mucho maxrep repeticiones
   v = np.random.uniform(-1, 1, A.shape[0]) # Generamos un vector de partida aleatorio, entre -1 y 1
   v = v / np.linalg.norm(v) # Lo normalizamos
   v1 = np.dot(A, v) # Aplicamos la matriz una vez
   v1 = v1 / np.linalg.norm(v1) # normalizamos
   l = np.dot(v.T, np.dot(A, v)) # Calculamos el autovector estimado
   l1 = np.dot(v1.T, np.dot(A, v1)) # Y el estimado en el siguiente paso
   nrep = 0 # Contador
   while np.abs(l1-l)/np.abs(l) > tol and nrep < maxrep: # Si estamos por debajo de la tolerancia buscada
      v = v1 # actualizamos v y repetimos
      l = l1
      v1 = np.dot(A, v) # Calculo nuevo v1
      v1 = v1 / np.linalg.norm(v1) # Normalizo
      l1 = np.dot(v1.T, np.dot(A, v1)) # Calculo autovector
      nrep += 1 # Un pasito mas
   if not nrep < maxrep:
      print('MaxRep alcanzado')
   l = np.dot(v1.T, np.dot(A, v1)) # Calculamos el autovalor
   return v1,l,nrep<maxrep

def deflaciona(A,tol=1e-8,maxrep=np.inf):
    # Recibe la matriz A, una tolerancia para el método de la potencia, y un número máximo de repeticiones
    v1,l1,_ = metpot1(A,tol,maxrep) # Buscamos primer autovector con método de la potencia
    deflA = A - l1 * np.outer(v1, v1) / np.dot(v1.T, v1) # Sugerencia, usar la funcion outer de numpy
    return deflA

def metpot2(A,v1,l1,tol=1e-8,maxrep=np.inf):
   # La funcion aplica el metodo de la potencia para buscar el segundo autovalor de A, suponiendo que sus autovectores son ortogonales
   # v1 y l1 son los primeors autovectores y autovalores de A}
   deflA = deflaciona(A, tol, maxrep)
   return metpot1(deflA,tol,maxrep)


def metpotI(A,mu,tol=1e-8,maxrep=np.inf):
    # Retorna el primer autovalor de la inversa de A + mu * I, junto a su autovector y si el método convergió.
    X = A + mu * np.eye(A.shape[0])
    return metpot1(func.inversa(X),tol=tol,maxrep=maxrep)

def metpotI2(A,mu,tol=1e-8,maxrep=np.inf):
   # Recibe la matriz A, y un valor mu y retorna el segundo autovalor y autovector de la matriz A, 
   # suponiendo que sus autovalores son positivos excepto por el menor que es igual a 0
   # Retorna el segundo autovector, su autovalor, y si el metodo llegó a converger.
   X = A + mu * np.eye(A.shape[0]) # Calculamos la matriz A shifteada en mu
   iX = func.inversa(X) # La invertimos
   defliX = deflaciona(iX) # La deflacionamos
   v,l,_ =  metpot1(defliX,0,0) # Buscamos su segundo autovector
   l = 1/l # Reobtenemos el autovalor correcto
   l -= mu
   return v,l,_


def laplaciano_iterativo(A,niveles,nombres_s=None):
    # Recibe una matriz A, una cantidad de niveles sobre los que hacer cortes, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.
    # La función debe, recursivamente, ir realizando cortes y reduciendo en 1 el número de niveles hasta llegar a 0 y retornar.
    if nombres_s is None: # Si no se proveyeron nombres, los asignamos poniendo del 0 al N-1
        nombres_s = range(A.shape[0])
    if A.shape[0] == 1 or niveles == 0: # Si llegamos al último paso, retornamos los nombres en una lista
        return([nombres_s])
    else: # Sino:
        L = calcula_L(A) # Recalculamos el L
        v,l,_ = metpot2(A,0,0) # Encontramos el segundo autovector de L
        # Recortamos A en dos partes, la que está asociada a el signo positivo de v y la que está asociada al negativo
        indices_positivos = [ni for ni, vi in zip(nombres_s, v) if vi > 0]
        indices_negativos = [ni for ni, vi in zip(nombres_s, v) if vi < 0]
        Ap = A[np.ix_([nombres_s.index(i) for i in indices_positivos], [nombres_s.index(i) for i in indices_positivos])] # Asociado al signo positivo
        Am = A[np.ix_([nombres_s.index(i) for i in indices_negativos], [nombres_s.index(i) for i in indices_negativos])] # Asociado al signo negativo

        return(
                laplaciano_iterativo(Ap,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi>0]) +
                laplaciano_iterativo(Am,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi<0])
                )


def modularidad_iterativo(A=None,R=None,nombres_s=None):
    # Recibe una matriz A, una matriz R de modularidad, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.

    if A is None and R is None:
        print('Dame una matriz')
        return(np.nan)
    if R is None:
        R = calcula_R(A)
    if nombres_s is None:
        nombres_s = range(R.shape[0])
    # Acá empieza lo bueno
    if R.shape[0] == 1: # Si llegamos al último nivel
        return[nombres_s]
    else:
        v,l,_ = metpot1(R) # Primer autovector y autovalor de R
        # Modularidad Actual:
        Q0 = np.sum(R[v>0,:][:,v>0]) + np.sum(R[v<0,:][:,v<0])
        if Q0<=0 or all(v>0) or all(v<0): # Si la modularidad actual es menor a cero, o no se propone una partición, terminamos
            return [nombres_s]#([[ni for ni,vi in zip(nombres_s,v) if vi>0],[ni for ni,vi in zip(nombres_s,v) if vi<0]])
        else:
            ## Hacemos como con L, pero usando directamente R para poder mantener siempre la misma matriz de modularidad
            Rp = R[np.ix_([ni for ni, vi in enumerate(v) if vi > 0], [ni for ni, vi in enumerate(v) if vi > 0])] # Parte de R asociada a los valores positivos de v
            Rm = R[np.ix_([ni for ni, vi in enumerate(v) if vi < 0], [ni for ni, vi in enumerate(v) if vi < 0])] # Parte asociada a los valores negativos de v
            vp,lp,_ = metpot1(Rp)  # autovector principal de Rp
            vm,lm,_ = metpot1(Rm) # autovector principal de Rm

            # Calculamos el cambio en Q que se produciría al hacer esta partición
            Q1 = 0
            if not all(vp>0) or all(vp<0):
               Q1 = np.sum(Rp[vp>0,:][:,vp>0]) + np.sum(Rp[vp<0,:][:,vp<0])
            if not all(vm>0) or all(vm<0):
                Q1 += np.sum(Rm[vm>0,:][:,vm>0]) + np.sum(Rm[vm<0,:][:,vm<0])
            if Q0 >= Q1: # Si al partir obtuvimos un Q menor, devolvemos la última partición que hicimos
                return([[ni for ni,vi in zip(nombres_s,v) if vi>0],[ni for ni,vi in zip(nombres_s,v) if vi<0]])
            else:
                # Sino, repetimos para los subniveles
                return(modularidad_iterativo(A, Rp, nombres_s=[ni for ni, vi in zip(nombres_s, v) if vi > 0]) +
                       modularidad_iterativo(A, Rm, nombres_s=[ni for ni, vi in zip(nombres_s, v) if vi < 0])
                )

def graficar_grafo(museos,barrios,escala,A):

    G = nx.from_numpy_array(A) # Construimos la red a partir de la matriz de adyacencia
    # Construimos un layout a partir de las coordenadas geográficas
    G_layout = {i:v for i,v in enumerate(zip(museos.to_crs("EPSG:22184").get_coordinates()['x'],museos.to_crs("EPSG:22184").get_coordinates()['y']))}
    fig, ax = plt.subplots(figsize=(15*escala, 15*escala)) # Visualización de la red en el mapa
    barrios.to_crs("EPSG:22184").boundary.plot(color='gray',ax=ax) # Graficamos Los barrios
    factor_escala = 30*escala # Escalamos los nodos 10 mil veces para que sean bien visibles
    nx.draw_networkx(G,G_layout,node_size = factor_escala, ax=ax,with_labels=False) # Graficamos red
