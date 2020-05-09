"""
DEFINE GOAL
"""
#libraries
import numpy as np

# define gamma and alpha parameters for the Q-Learning algorithm configuration
gamma = 0.75
alpha = 0.9

# STEP 1
# Define states
location_to_state = {
  "A":0,
  "B":1,
  "C":2,
  "D":3,
  "E":4, 
  "F":5,
  "G":6,
  "H":7,
  "I":8,
  "J":9,
  "K":10,
  "L":11
}

# Define actions
# that would be the same than;
# actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
actions = [i for i in range(12)]

# How can we add rewards?
# we have to create a matrix [states * actions]
R = np.array([
#  A,B,C,D,E,F,G,H,I,J,K,L
  [0,-1000,0,0,0,0,   0,0,0,0,0,0], # A
  [1,    0,1,0,0,1,   0,0,0,0,0,0], # B
  [0,-1000,0,0,0,0,   1,0,0,0,0,0], # C
  [0,    0,0,0,0,0,   0,1,0,0,0,0], # D
  [0,    0,0,0,0,0,   0,0,1,0,0,0], # E
  [0,-1000,0,0,0,0,   0,0,0,1,0,0], # F
  [0,    0,1,0,0,0,1000,1,0,0,0,0], # G  
  [0,    0,0,1,0,0,   1,0,0,0,0,1], # H
  [0,    0,0,0,1,0,   0,0,0,1,0,0], # I 
  [0,    0,0,0,0,1,   0,0,1,0,1,0], # J
  [0,    0,0,0,0,0,   0,0,0,1,0,1], # K
  [0,    0,0,0,0,0,   0,1,0,0,1,0]  # L
])

# STEP 2
# Building the solution with Q-Learning

#initialization of Q values
# inicializaremos una matriz con valores 0 para los valores de Q
Q = np.array(np.zeros([12, 12])) # definimos las dimensiones2

# implemntacion del proceso de Q-Learning
# 1. Seleccionamos un estado aleatorio st de nuestros 12 estados posibles 
for i in range(1000):  # haremos un total de 1000 iteraciones
  current_state = np.random.randint(0,12) # uniformemente distribuido

  # 2. Llevar a cabo una accion aleatoria at, que pueda conducir al siguiente 
  # estado posible
  # y seleccionar una solucion posible 
  # course solution: 
  # debemos definir a las playable actions, lo mismo que possible actions
  playable_actions = []
  num_possible_actions = len(R[current_state]) 
  for j in range(num_possible_actions):
    if R[current_state, j] != 0: # opcion positiva, por lo tanto ejecutable
      playable_actions.append(j) # obtenemos las acciones ejecutables
  
  # 3. Llegamos al siguiente estado y obtenemos la recompensa
  next_state = np.random.choice(playable_actions)
  
  # 4. Calcular la diferencia temporal. La usamos para maximazar la rempensa en
  # casa estado
  TD = R[current_state, next_state] + gamma*Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
  # argmax devuelve la posicion del indicado 
  
  # 5. Actualizamos el valor Q aplicando la ecuacion de Bellman
  Q[current_state, next_state] += alpha*TD

# mostrar la matrix en formato integer: print(Q.astype(int))
# con la matrix Q ahora podemos entender el plan que nuestro AI ha obtenido. 
# 
# Poner el modelo en produccion


