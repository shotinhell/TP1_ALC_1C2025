# Carga de paquetes necesarios para graficar
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Para leer archivos
import geopandas as gpd # Para hacer cosas geográficas
import seaborn as sns # Para hacer plots lindos
import networkx as nx # Construcción de la red en NetworkX
import scipy

# Leemos el archivo, retenemos aquellos museos que están en CABA, y descartamos aquellos que no tienen latitud y longitud
museos = gpd.read_file('https://raw.githubusercontent.com/MuseosAbiertos/Leaflet-museums-OpenStreetMap/refs/heads/principal/data/export.geojson')
barrios = gpd.read_file('https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-educacion/barrios/barrios.geojson')

def construye_adyacencia(D,m): 
    # Función que construye la matriz de adyacencia del grafo de museos
    # D matriz de distancias, m cantidad de links por nodo
    # Retorna la matriz de adyacencia como un numpy.
    D = D.copy()
    l = [] # Lista para guardar las filas
    for fila in D: # recorriendo las filas, anexamos vectores lógicos
        l.append(fila<=fila[np.argsort(fila)[m]] ) # En realidad, elegimos todos los nodos que estén a una distancia menor o igual a la del m-esimo más cercano
    A = np.asarray(l).astype(int) # Convertimos a entero
    np.fill_diagonal(A,0) # Borramos diagonal para eliminar autolinks
    return(A)

def calculaLU(matriz):
    n = matriz.shape[0]
    m = matriz.shape[1]
    L = np.eye(n,n)
    U = matriz.copy()
    if m != n:
        print('La matriz no es cuadrada')
        return
    else:
        for j in range(n):
            for i in range(j + 1, n):
                L[i, j] = U[i, j] / U[j, j]
                U[i, :] = U[i, :] - L[i, j] * U[j, :]
    return L, U

def determinante(matriz):
    _, U = calculaLU(matriz)
    res = 1
    for i in range(len(U)):
        res *= U[i][i]
    return res
    #caso base 2x2
    m = matriz.shape[1]
    if m == 2:
        return 
    
def inversa(matriz):
    L, U = calculaLU(matriz)
    n = matriz.shape[0]
    I = np.eye(n)
    res = np.zeros_like(matriz)
    for i in range(n):
       e = I[:, i]
       y = scipy.linalg.solve_triangular(L, e,)
       x = scipy.linalg.solve_triangular(U, y)
       res[:, i] = x
       
    return res
    
def calcula_matriz_C(A): 
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C
    K = np.zeros_like(A) # Inicializa matriz K con ceros, de la misma forma que A
    for i in range(len(A)):
        K[i][i] = sum(A[i][:]) # Suma los elementos de la fila i-ésima de A y lo asigna a la diagonal de K
    Kinv = inversa(K) # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de A
    C = Kinv @ A # Calcula C multiplicando Kinv y A
     # Calcula C multiplicando Kinv y A
    return C

    
def calcula_pagerank(A,alfa):
    # Función para calcular PageRank usando LU
    # A: Matriz de adyacencia
    # d: coeficientes de damping
    # Retorna: Un vector p con los coeficientes de page rank de cada museo
    C = calcula_matriz_C(A)
    N = len(A) # Obtenemos el número de museos N a partir de la estructura de la matriz A
    M = (1 - alfa) * C
    L, U = calculaLU(M) # Calculamos descomposición LU a partir de C y d
    b = np.ones(N)
    b = (alfa/N) * b # Vector de 1s, multiplicado por el coeficiente correspondiente usando d y N.
    Up = scipy.linalg.solve_triangular(L,b,lower=True) # Primera inversión usando L
    p = scipy.linalg.solve_triangular(U,Up) # Segunda inversión usando U
    return p

def calcula_matriz_C_continua(D): 
    # Función para calcular la matriz de trancisiones C
    # D: Matriz de adyacencia
    # Retorna la matriz C en versión continua
    D = D.copy()
    F = 1/D
    np.fill_diagonal(F,0)
    K = np.zeros_like(F) # Inicializa matriz K con ceros, de la misma forma que F
    for i in range(len(F)):
        K[i][i] = sum(F[i][:])
    # Suma los elementos de la fila i-ésima de F y lo asigna a la diagonal de K
    Kinv = inversa(K) # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de F 
    C = Kinv @ F # Calcula C multiplicando Kinv y F
    return C

def calcula_B(C,cantidad_de_visitas):
    # Recibe la matriz T de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matirz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    B = np.eye(C.shape[0])
    for i in range(cantidad_de_visitas-1):
        B += np.linalg.matrix_power(C, i) # Sumamos las matrices de transición para cada cantidad de pasos
    return B
