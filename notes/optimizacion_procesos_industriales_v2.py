"""
DEFINE GOAL
"""
#libraries
import numpy as np

# define gamma and alpha parameters for the Q-Learning algorithm configuration
gamma = 0.75
alpha = 0.9

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

# actions
actions = [i for i in range(12)]

R = np.array([
#  A,B,C,D,E,F,   G,H,I,J,K,L
  [0,1,0,0,0,0,   0,0,0,0,0,0], # A
  [1,0,1,0,0,1,   0,0,0,0,0,0], # B
  [0,1,0,0,0,0,   1,0,0,0,0,0], # C
  [0,0,0,0,0,0,   0,1,0,0,0,0], # D
  [0,0,0,0,0,0,   0,0,1,0,0,0], # E
  [0,1,0,0,0,0,   0,0,0,1,0,0], # F
  [0,0,1,0,0,0,1000,1,0,0,0,0], # G 
  [0,0,0,1,0,0,   1,0,0,0,0,1], # H
  [0,0,0,0,1,0,   0,0,0,1,0,0], # I 
  [0,0,0,0,0,1,   0,0,1,0,1,0], # J
  [0,0,0,0,0,0,   0,0,0,1,0,1], # K
  [0,0,0,0,0,0,   0,1,0,0,1,0]  # L
])

Q = np.array(np.zeros([12, 12])) # definimos las dimensiones2

for i in range(1000):
  current_state = np.random.randint(0,12) 
  playable_actions = []
  num_possible_actions = len(R[current_state]) 
  for j in range(num_possible_actions):
    if R[current_state, j] != 0: 
      playable_actions.append(j) 
  next_state = np.random.choice(playable_actions)
  TD = R[current_state, next_state] + gamma*Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
  Q[current_state, next_state] += alpha*TD
# Poner el modelo en produccion
# Transformacion inversa diccionario, de location to state a state to location
state_to_location = {state : location for location, state in location_to_state.items()}
# vamos a crear una funcion donde un modelo nos devolvera la ruta más óptima en 
# base a un punto de inicio A a un punto de inicio B
# los valores entrados en la funcion no seran numericos, sino caracteres
def route(starting_location, ending_location): 
  # check if the start or end location exist
  if starting_location not in location_to_state or ending_location not in location_to_state:
    print("Start or end point doesn't exist!")
    return False
  # anadiremos las localizaciones para crear la ruta
  route = [starting_location]
  # declarar next_location
  next_location = starting_location
  # check ending location exista en el diccionario
  while next_location != ending_location:
    starting_state = location_to_state[next_location]
    next_state = np.argmax(Q[starting_state, ]) # we have already calculated the Q matrix
    next_location = state_to_location[next_state]
    route.append(next_location)
  return route

print("La ruta elegida es:")
print(route(starting_location="E", ending_location="G"))
