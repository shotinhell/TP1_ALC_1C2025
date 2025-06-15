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

def matrizPermutacion(matriz):
    #el objetivo el elemento en la diagonal en la columna i a partir de la fila i sea mas grande que todos
    #por debajo de la fila i
    matriz = np.array(matriz, dtype=float)
    n = matriz.shape[0]
    res = np.eye(n)
    
    for i in range(n):
        #busco posición del elemento con mayor módulo en la columna i a partir de la fila i
        fila_maximo = np.argmax(np.abs(matriz[i:n, i])) + i
        #si el máximo no está en la fila i, cambio las filas de lugar
        if i != fila_maximo:
            matriz[[i, fila_maximo], :] = matriz[[fila_maximo, i], :]
            res[[i, fila_maximo], :] = res[[fila_maximo, i], :]
            
    return res
        
def calculaLU(matriz):
    #utilizo la función anterior para permutar la matriz
    matriz = np.array(matriz, dtype=float)
    P = matrizPermutacion(matriz)
    matriz_permutada = P @ matriz
    n = matriz.shape[0]
    m = matriz.shape[1]
    L = np.eye(n,n)
    U = matriz_permutada.copy()
    if m != n:
        print('La matriz no es cuadrada')
        return
    else:
        for j in range(n):
            for i in range(j + 1, n):
                L[i, j] = U[i, j] / U[j, j]
                U[i, :] = U[i, :] - L[i, j] * U[j, :]
    return L, U, P

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
    L, U, P = calculaLU(matriz)
    n = matriz.shape[0]
    I = np.eye(n)
    res = np.zeros_like(matriz, dtype=float)
    for i in range(n):
       e = I[:, i]  
       y = scipy.linalg.solve_triangular(L, P @ e, lower=True)
       x = scipy.linalg.solve_triangular(U, y, lower=False)
       res[:, i] = x
       
    return res
    
def calcula_matriz_C(A): 
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C
    K = np.zeros(A.shape) # Inicializamos la matriz K con ceros
    K = np.diag(A.sum(axis = 1))
    for i in range(len(K)):
      if K[i][i] == 0:
          K[i][i] = len(A)

    Kinv = inversa(K) # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de A
    C = Kinv @ A.T # Calcula C multiplicando Kinv y A
     # Calcula C multiplicando Kinv y A
    return C
    
def calcula_pagerank(A,alfa):
    # Función para calcular PageRank usando LU
    # A: Matriz de adyacencia
    # d: coeficientes de damping
    # Retorna: Un vector p con los coeficientes de page rank de cada museo
    C = calcula_matriz_C(A)
    N = len(A) # Obtenemos el número de museos N a partir de la estructura de la matriz A
    I = np.eye(N) # Matriz identidad de tamaño N
    M = I - (1-alfa) * C
    L, U, P = calculaLU(M) # Calculamos descomposición LU a partir de C y d
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
    K = np.diag(F.sum(axis = 1))

    # Suma los elementos de la fila i-ésima de F y lo asigna a la diagonal de K
    Kinv = inversa(K) # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de F
    C = F @ Kinv # Calcula C multiplicando Kinv y F
    return C

def calcula_B(C,cantidad_de_visitas):
    # Recibe la matriz T de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matirz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    B = np.eye(C.shape[0])
    for i in range(1, cantidad_de_visitas):
        B += np.linalg.matrix_power(C, i) # Sumamos las matrices de transición para cada cantidad de pasos
    return B

def norma1_matriz(A):
  return np.max(np.sum(np.abs(A), axis=0))   #valor absoluto de todo A, suma las columnas y toma el max

def calcula_x(A, b): #resuelve Ax=b con LU
    L, U, P= calculaLU(A)
    y = scipy.linalg.solve_triangular(L, b, lower=True)
    x = scipy.linalg.solve_triangular(U, y, lower=False)
    return x

def cond1(B):
  #calculo la norma de B, la de su inversa, las multiplico y devuelvo el res
  norma1_B = norma1_matriz(B)
  B_inv = inversa(B)
  norma1_B_inv = norma1_matriz(B_inv)
  res = norma1_B * norma1_B_inv
  return res

def graficar_pagerank(pr,museos,barrios,escala,A):
    pr_copy = pr
    pr_ratio = 3 # la relacion de tamaño
    pr_min = pr_copy.min()
    pr_max = pr_copy.max()
    pr_diff = pr_max-pr_min
    #ajustamos para que la diferencia sea mayor
    pr_copy = (pr_copy - pr_min)*pr_ratio + pr_min

    pr_copy = 2* pr_copy/pr.sum() # Normalizamos para que sume 1

    G = nx.from_numpy_array(A) # Construimos la red a partir de la matriz de adyacencia
    # Construimos un layout a partir de las coordenadas geográficas
    G_layout = {i:v for i,v in enumerate(zip(museos.to_crs("EPSG:22184").get_coordinates()['x'],museos.to_crs("EPSG:22184").get_coordinates()['y']))}
    fig, ax = plt.subplots(figsize=(15*escala, 15*escala)) # Visualización de la red en el mapa
    barrios.to_crs("EPSG:22184").boundary.plot(color='gray',ax=ax) # Graficamos Los barrios
    factor_escala = 1e4*escala # Escalamos los nodos 10 mil veces para que sean bien visibles
    nx.draw_networkx(G,G_layout,node_size = pr_copy*factor_escala, ax=ax,with_labels=False) # Graficamos red
