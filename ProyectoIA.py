import random
import numpy
#importando librerias de deap
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import matplotlib.pyplot as plt

print("cruza 1 punto, cruza 80%, mutacion 1%, seleccion por ruleta,representacion Entera \n")
#Parámetros del problema 
mejorbeneficio = 0
mejoresinversiones = [0,0,0,0]
# FUNCIÓN DE FITNESS
# se crea funcion que se utlizara para generar la poblacion
def CrearIversiones():
    bandera = True
    while(bandera):
        #asegurar que los 4 numeros sumados den una suma de 10
        iteraciones = 0
        inversiontotal = 0
        inversiones = [0,0,0,0]
        while(iteraciones<4):
            #se generan 4 numeros random
            inversionporzona = random.randint(0,10)
            inversiontotal += inversionporzona
            inversiones[iteraciones] = inversionporzona
            iteraciones = iteraciones+1
        if(inversiontotal == 10):
            bandera = False
    return inversiones
    #funcion que servira para evaluar los individuos
def EvaluarAptitud(inversiones):    
    global mejorbeneficio
    global mejoresinversiones
    tabladebeneficios = [[0,0,0,0],   #invirtiendo 0 millon
        [.28,.25,.15,.20], #invirtiendo 1 millon
        [.45,.41,.25,.33], #invirtiendo 2 millon
        [.65,.55,.40,.42], #invirtiendo 3 millon
        [.78,.65,.50,.48], #invirtiendo 4 millon
        [.90,.75,.62,.53], #invirtiendo 5 millon
        [1.02,.80,.73,.56],#invirtiendo 6 millon
        [1.13,.85,.82,.58],#invirtiendo 7 millon
        [1.23,.88,.90,.60],#invirtiendo 8 millon
        [1.32,.90,.96,.60],#invirtiendo 9 millon
        [1.38,.90,1,.60] ]  #invirtiendo 10 millon

    sumadordeinversiones = 0
    for x in inversiones:
        sumadordeinversiones = sumadordeinversiones+x #se suman las inverciones de las 4 zonas
    sumadordebeneficios = 0
    if(sumadordeinversiones > 10):#si es mayor a 10 se aplica funcion de aptidud penalizada
        sumadordebeneficios = tabladebeneficios[round(inversiones[0])][0] + tabladebeneficios[round(inversiones[1])][1] + tabladebeneficios[round(inversiones[2])][2] + tabladebeneficios[round(inversiones[3])][3]
        v= 10 - sumadordebeneficios   
        funciondeaptitud = sumadordebeneficios/((500*v)+1)
        sumadordebeneficios = funciondeaptitud
    else:    #si no es mayor a 10 solo obtenemos el beneficio por zona
        sumadordebeneficios = tabladebeneficios[round(inversiones[0])][0] + tabladebeneficios[round(inversiones[1])][1] + tabladebeneficios[round(inversiones[2])][2] + tabladebeneficios[round(inversiones[3])][3]
    if(sumadordebeneficios > mejorbeneficio):#capturamos el mejor beneficio obtenido
        mejorbeneficio = sumadordebeneficios
        mejoresinversiones = inversiones
    return sumadordebeneficios,


# DEFINICIÓN DEL PROBLEMA
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# REGISTRO DE FUNCIONES QUE SON NECESARIAS -- CAJA DE HERRAMIENTAS
#metodos de la libreria que vienen en las diapositivas
toolbox = base.Toolbox()
toolbox.register("permutation", CrearIversiones)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.permutation)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", EvaluarAptitud)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1)
toolbox.register("select", tools.selRoulette)

def main():
    seed=0
    random.seed(seed)

    pop = toolbox.population(n=200)
    stats = tools.Statistics(lambda ind: ind.fitness.values) # calcular estadísticas
    stats.register("Avg", numpy.mean)
    stats.register("Std", numpy.std)
    stats.register("Min", numpy.min)
    stats.register("Max", numpy.max)
    

    algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.01, ngen=100, stats=stats,
                        verbose=True)

    return pop, stats

if __name__ == "__main__":
    pop, stats = main()
    print ("el mejor individuo es"+str(mejoresinversiones)+" con el beneficio "+str(mejorbeneficio))

   