#!/usr/bin/env python
# coding: utf-8

# # Funciones para el análisis de relaciones de parentesco usando perfiles genéticos

# In[2]:


import pandas as pd
import numpy as np

# para construir tablas para las dpc
from itertools import product

# para enlistar genotipos ordenados
from itertools import permutations
import functools

# Modulos de pgmpy para redes bayesianas
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference.EliminationOrder import MinFill, MinNeighbors, MinWeight, WeightedMinFill 
from pgmpy.inference import VariableElimination


# In[31]:


__author__= 'Rolando Corona Jiménez'
__version__ = '1.0'
__email__= 'rolando.corona@cimat.mx'
__date__ = '2021-05-19'


# ## Modelo de herencia

# In[3]:


'''
Funciones para el modelo de herencia
'''

def modelo_herencia_mendeliano(alelo, alelos_progenitor):
    '''
    Calcula la probabilidad que se transmita un alelo dado el genotipo del progenitor según un modelo mendeliano (sin mutaciones).

    Parameters
    -------------
    alelo: object
        El alelo transmitido
    alelos_progenitor: tuple
        El par de alelos del progenitor (genotipo)
    
    Returns
    -------------
    La probabilidad que se herede el alelo 'alelo' dado que el genotipo es 'alelos_progenitor', Pr(alelo|alelos_progenitor)
    '''
    
    # si el alelo no es compatible con el genotipo del progenitor
    if not(alelo in set(alelos_progenitor)):
        return 0
    # si el progenitor es homocigoto
    elif alelo in set(alelos_progenitor) and len(set(alelos_progenitor)) == 1:
        return 1
    # si el progenitor es heterocigoto
    elif alelo in set(alelos_progenitor) and len(set(alelos_progenitor)) == 2:
        return .5


# In[33]:


def crear_modelo_mutacion_constante(mu, n_alelos):
    '''
    Crea un modelo de mutación mutación constante a partir de la tasa de mutación y el número de alelos del locus.
    
    Parameters
    -------------
    mu: float
        La tasa de mutación
    n_alelos: int
        El número de alelos del locus
        
    Returns
    -------------
    Una función que calcula las entradas de la matriz de mutación. f(a,b) = Pr(a sea transmitido como b)
    '''
    
    return lambda alelo_original, alelo_transmitido: (1-mu) if alelo_transmitido == alelo_original else mu/(n_alelos-1)


# In[5]:


def crear_modelo_herencia(modelo_mutacion):
    '''
    Crea una función que calcula la probabilidad que se transmita un alelo dado el genotipo del progenitor según un modelo de mutación arbitrario.
    
    Parameters
    -------------
    alelo: function
        Función para las probabilidades de mutación, f(a,b) = Pr(a sea transmitido como b).
        
    Returns
    -------------
    Una función que calcula la probabilidad que se transmita un alelo dados los de su progenitor, Pr(alelo|alelos_progenitor).
    '''

    return lambda alelo, alelos_progenitor: 0.5*modelo_mutacion(alelo, alelos_progenitor[0]) + 0.5*modelo_mutacion(alelo, alelos_progenitor[1])


# In[6]:


def tabla_herencia(alelos, modelo_herencia):
    '''
    Calcula una tabla que respresenta distribución de probabilidad condicional del modelo de herencia Pr(alelo|alelos_progenitor).
    
    Parameters
    -------------
    alelos: list
        Nombres de los alelos del locus
    modelo_herencia: function
        Modelo de herencia para calcular Pr(alelo|alelos_progenitor)

    Returns
    -------------
        Un dataframe que tiene como índices a los alelos y como columnas todos los posibles genotipos del progenitor (alelos_progenitor).
        Las entradas representan Pr(alelo|alelos_progenitor).
    '''

    # se obtienen los posibles genotipos del progenitor
    pares_alelos = pd.Series(list(product(alelos, repeat = 2)))
    # se calculan las probabilidades Pr(alelo|alelos_progenitor) y se guardan en un dataframe
    df_tabla_herencia = pd.Series(alelos).apply(lambda alelo: pares_alelos.apply(lambda alelos_progenitor: modelo_herencia(alelo, alelos_progenitor)))
    df_tabla_herencia.index  = alelos
    df_tabla_herencia.columns = pares_alelos
    return df_tabla_herencia


# ## Modelo observacional

# In[ ]:


'''
Funciones para el modelo observacional
'''

#esta asociado a los genotipos en orden
def etiquetar_genotipos(genotipos, personas):
    '''
    Función auxiliar para agregar etiquetas a los genotipos ordenados que se calculan con la función 'enlistar_genotipos_ordenados'.
    
    Parameters
    -------------
    genotipos: tuple
        Tupla que contiene todos los genotipos ordenados. Lo genotipos están anidados en otras tuplas.
    personas: list
        Nombres de las personas
    
    Returns
    -------------
        Una serie que tiene como índices los alelos de las personas.
        Se agregan dos registros por cada persona, si se llama "Persona", se agregan "Persona_0" y "Persona_1" para los alelos materno y paterno, respectivamente.
        
    genotipos es un conjunto de tuplas anidadas de la forma
    ((('a_1', 'a_3'), ('a_1', 'a_4')), ('a_1', 'a_3'))
    
    Lo que hace la función es agregar la etiqueta de persona y el tipo de alelo (paterno, materno).
    Por ejemplo, si las personas son P1, P2 y P3, 
    sus genotipos son 
    ('a_1', 'a_3'), ('a_1', 'a_4') y ('a_1', 'a_3')
    
    Para P1 = ('a_1', 'a_3'), la primera entrada 'a_1' es el alelo paterno y la segunda 'a_3', el materno.
    En ese caso se agregan dos registros
    P1_0 = 'a_3'
    P1_1 = 'a_1'
    
    La salida es 
    
    P1_0 = 'a_3'
    P1_1 = 'a_1'
    P2_0 = 'a_4'
    P2_1 = 'a_1'
    P3_0 = 'a_3'
    P4_1 = 'a_1'
    '''

    n_personas = len(personas)
    genotipos_etiquetados = pd.Series([], dtype = 'object')
    for i in range(n_personas):
        n = n_personas - (i+1)
        if n == 0:
            genotipos_etiquetados.loc[personas[n]+'_0'] = genotipos[1]
            genotipos_etiquetados.loc[personas[n]+'_1'] = genotipos[0]
        else:
            genotipos_etiquetados.loc[personas[n]+'_0'] = genotipos[1][1]
            genotipos_etiquetados.loc[personas[n]+'_1'] = genotipos[1][0]    
        genotipos = genotipos[0]
    return genotipos_etiquetados 


# In[37]:


def enlistar_genotipos_ordenados(df_genotipos_observados, personas):
    '''
    Enlista todos los posibles órdenes para los genotipos observados.

    Parameters
    -------------
    df_genotipos_observados: pandas.DataFrame
        Dataframe cuyas filas son los genotipos observados (tiene dos columnas).
    personas: list
        Nombres de las personas

    Returns
    -------------
        Un datafrme cuyas columnas son todos los genotipos ordenados posibles según los observados.
        Por cada persona se tienen dos filas para los alelos materno y paterno.
    '''
    
    # se obtienen las permutaciones de cada genotipo
    df_permutaciones_genotipos = df_genotipos_observados.apply(lambda genotipo: permutations(genotipo), axis = 1)
    # se obtienen todas las posibles combinaciones de genotipos ordenados
    producto_permutaciones = functools.reduce(product, df_permutaciones_genotipos)
    df_genotipos_ordenados = pd.DataFrame(map(lambda genotipos: etiquetar_genotipos(genotipos, personas), producto_permutaciones)).T
    # se eliminan los duplicados que se introducen por los homocigotos
    df_genotipos_ordenados_unicos = df_genotipos_ordenados.loc[:,~df_genotipos_ordenados.T.duplicated(keep='first')]
    return df_genotipos_ordenados_unicos


# ## Representación de la red de alelos

# In[9]:


'''
Funciones para definir la red de alelos
'''

def crear_aristas_red_alelos(dict_relaciones_persona_padres):
    '''
    Crea las aristas para la red de alelos a partir de las relaciones de parentesco entre las personas del pedigrí.

    Parameters
    -------------
    dict_relaciones_persona_padres: dict
        Diccionario con las relaciones de la forma Persona:(Padre,Madre)
        
    Returns
    -------------
        Una lista de tuplas de la forma (Alelo_Progenitor, Alelo_Persona). Por cada relación Persona:(Padre,Madre) se agregan cuatro aristas.
        (Madre_0, Persona_0), (Madre_1, Persona_0) (Padre_0, Persona_0), (Padre, Persona_0)
    
    Diccionario de relaciones persona-padres
    {P3: (P1, P2), P6: (P3, P4), P7: (P5, P6)}
    
    Para la terna
    P3: (P1, P2)
    
    Se agregan las aristas
    P1_0 -> P3_1 (del alelo paterno de 1 (padre) al alelo que 3 recibe de 1)
    P1_1 -> P3_1 (del alelo materno de 1 (padre) al alelo que 3 recibe de 1)
    P2_0 -> P3_0 (del alelo paterno de 2 (madre) al alelo que 3 recibe de 2)
    P2_1 -> P3_0 (del alelo materno de 2 (madre) al alelo que 3 recibe de 2)
    '''
    lista_relaciones_pares_anidada = [[(dict_relaciones_persona_padres[persona][0]+'_1', persona+'_1'), 
                                   (dict_relaciones_persona_padres[persona][0]+'_0', persona+'_1'),
                                   (dict_relaciones_persona_padres[persona][1]+'_1', persona+'_0'),
                                   (dict_relaciones_persona_padres[persona][1]+'_0', persona+'_0')] for persona in dict_relaciones_persona_padres]
    lista_relaciones_pares = [relacion for sublist in lista_relaciones_pares_anidada for relacion in sublist]
    return lista_relaciones_pares


# In[10]:


def construir_cpds(persona, fundador_descendiente, dict_relaciones_persona_padres, n_alelos, cpd):
    '''
    Construye los objetos de la clase TabularCPD que corresponden a las distribuciones de probabilidad condicional asociados a cada nodo en la red de alelos.
    
    Parameters
    -------------
    persona: str
        Nombre de la persona
    fundador_descendiente: str
        Indica si la persona es fundador o descendiente ("fundador", "descendiente")
    dict_relaciones_persona_padres: dict
        Diccionario con las relaciones de la forma Persona:(Padre,Madre)
    n_alelos: int
        numero de alelos
    cpd: numpy.array
        Arreglo con los valores de la tabla que representa a la distribución de probabilidad condicional
        
    Returns
    -------------
    Un diccionario que con las dos cpds (pgmpy.factors.discrete.CPD.TabularCPD) asociadas a los alelos de la persona.
    '''
    
    #si es fundador
    if fundador_descendiente == 'fundador':
        #alelo de la madre
        cpd_f_0 = TabularCPD(variable=persona+'_0', variable_card=n_alelos, values=cpd)
        #alelo del padre
        cpd_f_1 = TabularCPD(variable=persona+'_1', variable_card=n_alelos, values=cpd)
        return {'cpd_'+persona+'_0':cpd_f_0, 'cpd_'+persona+'_1':cpd_f_1}
    elif fundador_descendiente == 'descendiente':
        #herencia de la madre
        cpd_d_0 = TabularCPD(variable=persona+'_0', variable_card=n_alelos, 
                             values=cpd,
                             evidence=[dict_relaciones_persona_padres[persona][1] + '_1', dict_relaciones_persona_padres[persona][1] + '_0'],
                             evidence_card=[n_alelos, n_alelos])

        #herencia del padre
        cpd_d_1 = TabularCPD(variable=persona+'_1', variable_card=n_alelos, 
                             values=cpd,
                             evidence=[dict_relaciones_persona_padres[persona][0] + '_1', dict_relaciones_persona_padres[persona][0] + '_0'],
                             evidence_card=[n_alelos, n_alelos])
        return {'cpd_'+persona+'_0':cpd_d_0, 'cpd_'+persona+'_1':cpd_d_1}


# ## Inferencia en la red de alelos

# In[11]:


'''
Funciones para hacer inferencias en la red de alelos
'''

def distribucion_marginal_pedigri(fundadores, descendientes,
                                  dict_relaciones_persona_padres,
                                  personas_observadas, personas_no_observadas,
                                  alelos, frecuencias_alelos, 
                                  modelo_herencia = modelo_herencia_mendeliano, 
                                  metodo_eliminacion = MinNeighbors,
                                  mostrar_model_check = False, mostrar_orden_eliminacion = False, 
                                  mostrar_avance_ve = False, mostrar_avance_inferencia = False):
    
    '''
    Calcula la distribución marginal sobre los alelos observados en un conjunto de personas que forman parte de un pedigri.
    Se requiere la información sobre las relaciones de parentesco así como de los parámetros de la población y del modelo de herencia.
    La distribución marginal se calcula usando el método de eliminación de variables.
    
    Parameters
    -------------
    fundadores: list
        Nombres de los fundadores
    descendientes: list
        Nombres de los descendientes
    dict_relaciones_persona_padres: dict
        Diccionario con las relaciones de la forma Persona:(Padre,Madre)
    personas_observadas: list
        Personas de las que se conoce su genotipo
    personas_no_observadas: list
        Personas sin información
    alelos: list
        Nombres de los alelos del locus
    frecuencias_alelos: list
        Frecuencias de los alelos del locus
    modelo_herencia: function
        Modelo de herencia para calcular Pr(alelo|alelos_progenitor). Por defecto modelo_herencia_mendeliano.
    metodo_eliminacion: pgmpy.inference.EliminationOrder
        Método de eliminación de variables. Por defecto MinNeighbors. Otros métodos son MinFill, MinWeight y WeightedMinFill.
    mostrar_model_check: bool 
        Mostrar validación del modelo. Por defecto False.
    mostrar_orden_eliminacion = False.
        Mostrar el orden de eliminación. Por defecto False.
    mostrar_avance_ve
        Mostrar el progreso en el cálculo del orden de eliminación. Por defecto False.
    mostrar_avance_inferencia
        Mostrar el progreso en el cálculo de la distribución marginal. Por defecto False.

    Returns
    -------------
    La distribución marginal de los genotipos observados. Un objeto de la clase pgmpy.factors.discrete.DiscreteFactor.
    Las variables de esta distribución tienen identificadores de la forma Persona_0, Persona_1.
    Los valores que toma la marginal son valores númericos 0,...,n-1 que corresponden a los nombres de los alelos en el orden que aparecen en la lista 'alelos'
    Si alelos = [a_1, a_2, a_3, a_4, a_5], los valores en la red de alelos se codifican como [0,1,2,3,4]
    '''
    
    n_alelos = len(alelos)
    df_tabla_herencia = tabla_herencia(alelos, modelo_herencia)
    #arrays para las CPD
    fundadores_cpd = np.array(frecuencias_alelos).reshape(n_alelos, 1)
    descendientes_cpd = df_tabla_herencia.values
    
    #aristas de la red de alelos
    relaciones_pares = crear_aristas_red_alelos(dict_relaciones_persona_padres)
    #definicion de la red de alelos
    model = BayesianModel(relaciones_pares)
    cpds_anidada = []
    
    #cpds fundadores
    for persona in fundadores:
        # 2 cpds por persona
        cpds_persona = construir_cpds(persona, 'fundador', dict_relaciones_persona_padres, n_alelos, fundadores_cpd)
        cpds_anidada.append(list(cpds_persona.values()))
        
    #cpds descendientes
    for persona in descendientes:
        # 2 cpds por persona
        cpds_persona = construir_cpds(persona, 'descendiente', dict_relaciones_persona_padres, n_alelos, descendientes_cpd)
        cpds_anidada.append(list(cpds_persona.values()))
    
    #se extraen los cpd de la lista anidada
    cpds = [cpd for sublist in cpds_anidada for cpd in sublist]
    
    #se agregan las cdps al modelo
    for cpd in cpds:
        model.add_cpds(cpd)
    if mostrar_model_check:
        print("Modelo válido :", model.check_model())

    # se generan los nombres de los nodos
    personas_marginal = personas_observadas
    nodos_marginal_anidada = [[persona +'_1', persona +'_0'] for persona in personas_marginal]
    nodos_marginal = [nodo_red for sublist in nodos_marginal_anidada for nodo_red in sublist]
    personas_eliminar = personas_no_observadas
    nodos_eliminar_anidada = [[persona +'_1', persona +'_0'] for persona in personas_eliminar]
    nodos_eliminar = [nodo_red for sublist in nodos_eliminar_anidada for nodo_red in sublist]
    # se calcula el orden de eliminación
    orden_eliminacion = metodo_eliminacion(model).get_elimination_order(nodos_eliminar, show_progress= mostrar_avance_ve)
    if mostrar_orden_eliminacion:
        print("Orden de eliminación: ", orden_eliminacion)
    inference = VariableElimination(model)
    # se calcula la distribución marginal
    distribucion_marginal = inference.query(nodos_marginal, evidence=None, elimination_order = orden_eliminacion , joint=True, show_progress= mostrar_avance_inferencia)
    return distribucion_marginal


# In[12]:


def evaluar_marginal_pedigri(distribucion_marginal, alelos_ordenados, nombres_alelos):
    '''
    Evalua la distribución marginal en sobre un conjunto de genotipos ordenados.
    Las etiquetas de los alelos ordenados deben coincidir con los identificadores de las variables de la distribución marginal.
    
    Parameters
    -------------
    distribucion_marginal: pgmpy.factors.discrete.DiscreteFactor
        Distribución marginal definida sobre algunos nodos de la red de alelos
    alelos_ordenados: pd.Series
        
    
    Returns
    -------------
    La evaluación de la marginal en los genotipos ordenados dados.
    
    Si los genotipos ordenados son
    P2 = (a_1,a_3)
    P5 = (a_1,a_4)
    P7 = (a_1,a_3)
    
    Los alelos ordenados deben venir etiquetados como
    P2_0    a_3
    P2_1    a_1
    P5_0    a_4
    P5_1    a_1
    P7_0    a_3
    P7_1    a_1
    '''
    # se codifican los valores de los alelos como números
    dict_nombres_numero = dict(zip(nombres_alelos, range(len(nombres_alelos))))
    alelos_ordenados = alelos_ordenados.replace(dict_nombres_numero)
    return distribucion_marginal.get_value(**dict(alelos_ordenados))


# In[13]:


#opcion de regresar todos los calculos
def verosimilitud_pedigri(fundadores, descendientes,
                          dict_relaciones_persona_padres,
                          personas_observadas, personas_no_observadas,
                          alelos, frecuencias_alelos,
                          df_genotipos_observados,
                          modelo_herencia = modelo_herencia_mendeliano, 
                          metodo_eliminacion = MinNeighbors,
                          return_genotipos_ordenados = False,
                          mostrar_model_check = False, mostrar_orden_eliminacion = False, 
                          mostrar_avance_ve = False, mostrar_avance_inferencia = False):
    
    '''
    Calcula la verosimilitud de un pedigrí a partir de los genotipos obsevados.
    La verosimilitud del pedigrí es la suma de las evaluaciones de la distribución marginal sobre todos los genotipos ordenados.
    Se requiere la información sobre las relaciones de parentesco, los genotipos observados, así como de los parámetros de la población y del modelo de herencia.
    
    Parameters
    -------------
    fundadores: list
        Nombres de los fundadores
    descendientes: list
        Nombres de los descendientes
    dict_relaciones_persona_padres: dict
        Diccionario con las relaciones de la forma Persona:(Padre,Madre)
    personas_observadas: list
        Personas de las que se conoce su genotipo
    personas_no_observadas: list
        Personas sin información
    alelos: list
        Nombres de los alelos del locus
    frecuencias_alelos: list
        Frecuencias de los alelos del locus
    df_genotipos_observados: pandas.DataFrame
        Dataframe con la información de los genotipos observados (2 columnas) que tiene como indices los nombres de las personas.
    modelo_herencia: function
        Modelo de herencia para calcular Pr(alelo|alelos_progenitor). Por defecto modelo_herencia_mendeliano.
    metodo_eliminacion: pgmpy.inference.EliminationOrder
        Método de eliminación de variables. Por defecto MinNeighbors. Otros métodos son MinFill, MinWeight y WeightedMinFill.
    return_genotipos_ordenados: bool
        Regresar un dataframe con todos los genotipos ordenados generados con el cálculo de la marginal en cada uno.
    mostrar_model_check: bool 
        Mostrar validación del modelo. Por defecto False.
    mostrar_orden_eliminacion = False.
        Mostrar el orden de eliminación. Por defecto False.
    mostrar_avance_ve
        Mostrar el progreso en el cálculo del orden de eliminación. Por defecto False.
    mostrar_avance_inferencia
        Mostrar el progreso en el cálculo de la distribución marginal. Por defecto False.

    Returns
    -------------
    La verosimilitud del pedigrí para los genotipos observados.
    Si return_genotipos_ordenados = True, entonces se regresa un dataframe con todos los genotipos ordenados y la probabilidad marginal evaluada en cada uno.
    '''
    
    # se obtiene la distribución marginal
    distribucion_marginal = distribucion_marginal_pedigri(fundadores, descendientes,
                                                          dict_relaciones_persona_padres,
                                                          personas_observadas, personas_no_observadas,
                                                          alelos, frecuencias_alelos, 
                                                          modelo_herencia, 
                                                          metodo_eliminacion,
                                                          mostrar_model_check, mostrar_orden_eliminacion, 
                                                          mostrar_avance_ve, mostrar_avance_inferencia)
    
    # se obtienen los genotipos ordenados
    df_genotipos_ordenados = enlistar_genotipos_ordenados(df_genotipos_observados, df_genotipos_observados.index)

    # se evalua la marginal en cada uno de los genotipos ordenados y se regresa la suma
    probabilidades_genotipos = df_genotipos_ordenados.apply(lambda alelos_ordenados: evaluar_marginal_pedigri(distribucion_marginal, alelos_ordenados = alelos_ordenados, nombres_alelos = alelos))
    verosimilitud = probabilidades_genotipos.sum()
    if return_genotipos_ordenados:
        df_resumen = df_genotipos_ordenados.copy()
        df_resumen.loc['Probabilidad'] = probabilidades_genotipos
        return df_resumen
    return verosimilitud

