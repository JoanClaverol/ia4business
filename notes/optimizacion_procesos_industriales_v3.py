"""
DEFINE GOAL: 
Mejora de nuestro algoritmo que tenga en cuenta tareas intermedias; 
A. Si hay un punto intermedio interesante, se puede dar una recompensa positiva
    al caer en este. 
B. Si hay un sitio mejor que otro, dar a la opcion que queremos evitar con un 
    castigo negativo.
C. Si hay rutas mas importantes a nivel de prioridad, que pase por ellas antes 
    de llegar a nuestro objetivo final. 
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

# matriz de adyacencias
R = np.array([
#  A,B,C,D,E,F,G,H,I,J,K,L
  [0,1,0,0,0,0,0,0,0,0,0,0], # A
  [1,0,1,0,0,1,0,0,0,0,0,0], # B
  [0,1,0,0,0,0,1,0,0,0,0,0], # C
  [0,0,0,0,0,0,0,1,0,0,0,0], # D
  [0,0,0,0,0,0,0,0,1,0,0,0], # E
  [0,1,0,0,0,0,0,0,0,1,0,0], # F
  [0,0,1,0,0,0,0,1,0,0,0,0], # G 
  [0,0,0,1,0,0,1,0,0,0,0,1], # H
  [0,0,0,0,1,0,0,0,0,1,0,0], # I 
  [0,0,0,0,0,1,0,0,1,0,1,0], # J
  [0,0,0,0,0,0,0,0,0,1,0,1], # K
  [0,0,0,0,0,0,0,1,0,0,1,0]  # L
])

# Transformacion inversa diccionario, de location to state a state to location
state_to_location = {state : location for location, state in location_to_state.items()}

# define ranks
priority_rank = {
  "G":1,
  "K":2,
  "L":3,
  "J":4,
  "A":5,
  "I":6,
  "H":7,
  "C":8,
  "B":9,
  "D":10,
  "F":11,
  "E":12
}
# and create weights to the ranks 
weights = [weight + 1 for weight in range(R.shape[1])]
reversed_rank = [i for i in reversed(list(priority_rank.keys()))]
weights_rank = dict(zip(reversed_rank, weights))
# add rewards based on the priority ranks
for col in range(R.shape[1]):
  location = state_to_location[col]
  weight = weights_rank[location]
  R[:,col] = np.where(R[:,col] == 1, weight, R[:,col])


# function to create the Q matrix
def Q_values(starting_location, ending_location): 
  # check if the locations exists
  if starting_location not in location_to_state or ending_location not in location_to_state:
    print("Start or end point doesn't exist!")
    return False
  # create the Q-learning matrix
  R_new = R.copy()
  # add the location to go
  ending_state = location_to_state[ending_location]
  R_new[ending_state, ending_state] = 1000
  Q = np.array(np.zeros([R.shape[0], R.shape[1]])) 
  for i in range(1000):
    current_state = np.random.randint(0,R.shape[1]) 
    playable_actions = []
    num_possible_actions = len(R_new[current_state]) 
    for j in range(num_possible_actions):
      if R_new[current_state, j] != 0: 
        playable_actions.append(j)
    next_state = np.random.choice(playable_actions)
    TD = R_new[current_state, next_state] + gamma*Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
    Q[current_state, next_state] += alpha*TD
  return Q


# funcion donde se dirigira de la forma mas optima de un punto inicial a un punto
# final, pasando por una lista de puntos intermedios. 
def best_route(starting_location, ending_location, middle_location=[]):#, middle_location): 
  # anadiremos las localizaciones para crear la ruta
  route = [starting_location]
  # paramos antes al punto medio
  while middle_location: 
    Q = Q_values(starting_location, middle_location[0])
    next_location=route[-1]
    while next_location != middle_location[0]:
      starting_state = location_to_state[next_location]
      next_state = np.argmax(Q[starting_state, ])
      next_location = state_to_location[next_state]
      route.append(next_location)
    middle_location = middle_location[1:]
  # ahora nos dirigimos al punto final, calculamos de nuevo los valores de Q
  # para el punto final
  Q = Q_values(starting_location, ending_location)
  next_location=route[-1]
  while next_location != ending_location:
    starting_state = location_to_state[next_location]
    next_state = np.argmax(Q[starting_state, ])
    next_location = state_to_location[next_state]
    route.append(next_location)
  return route


print("La ruta elegida es:")
print(best_route(starting_location="E",ending_location="L",middle_location=["B","E"]))
